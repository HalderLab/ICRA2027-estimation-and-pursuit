#!/usr/bin/env python3
"""NumPy port of growMHE(8).m; no MATLAB, ROS, or external solver required.

    from growMHE import growMHE
    estimator = growMHE()
    out = estimator.step(t, rho, alpha2, vSelf, uSelf)
    out = estimator.predict(t, vSelf, uSelf)

All times are seconds on ONE nondecreasing timeline (prefer elapsed time).
step timestamps must be strictly increasing; a step may not precede a predict.
alpha2 ALWAYS means the local measured bearing, irrespective of robot identity.
The MMC adapter maps these self/other outputs to global agent 1/2 coordinates.
Each instance owns independent buffers. Call methods serially, not concurrently.

The continuous costate RK4 and trapezoidal PMP residual deliberately reproduce
MATLAB, rather than substituting an exact discrete-RK4 adjoint or another solver.
Default parameters and warm starts match the supplied MATLAB source. Config is
copied at construction and must not be changed while an estimator is running.
"""
from dataclasses import dataclass, field
from copy import deepcopy
import math
import numpy as np


def wrap_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class MHEConfig:
    Nh: int = 30
    minSamples: int = 5
    nSub: int = 5
    qx: float = 20.0
    qy: float = 20.0
    chi_v: float = 2.0
    chi_u: float = 2.0
    Pa: np.ndarray = field(default_factory=lambda: np.diag(
        [0.05**2, 0.05**2, (30*math.pi/180)**2, 0.05**2, 0.05**2]))
    PfInv: np.ndarray = field(default_factory=lambda: np.zeros((5, 5)))
    terminalReference: np.ndarray = field(default_factory=lambda: np.zeros(5))
    initialThetaGuess: float = 0.20
    initialV1Guess: float = 0.10
    initialU1Guess: float = 0.00
    maxIterBoot: int = 120
    maxIterRT: int = 10
    relCostTol: float = 1e-6
    armijoC: float = 1e-4
    maxLineSearch: int = 20
    predictionMaxStep: float = 0.01


def _interp_linear(t, values, query):
    """MATLAB interp1(..., 'linear', 'extrap'), including outside endpoints."""
    t, values, query = np.asarray(t), np.asarray(values), np.asarray(query)
    idx = np.clip(np.searchsorted(t, query, side='right')-1, 0, len(t)-2)
    return values[idx] + (query-t[idx])*(values[idx+1]-values[idx])/(t[idx+1]-t[idx])


class growMHE:
    def __init__(self, cfg=None):
        self.cfg = deepcopy(cfg) if cfg is not None else MHEConfig()
        c = self.cfg
        for name in ('Nh', 'minSamples', 'nSub', 'maxIterBoot', 'maxIterRT', 'maxLineSearch'):
            value = getattr(c, name)
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if not 2 <= c.minSamples <= c.Nh+1:
            raise ValueError('minSamples must satisfy 2 <= minSamples <= Nh+1.')
        c.Pa = np.array(c.Pa, dtype=float, copy=True)
        c.PfInv = np.array(c.PfInv, dtype=float, copy=True)
        c.terminalReference = np.array(c.terminalReference, dtype=float, copy=True).reshape(5)
        if c.Pa.shape != (5, 5) or c.PfInv.shape != (5, 5):
            raise ValueError('Pa and PfInv must be 5-by-5 matrices.')
        if not np.array_equal(c.Pa, np.diag(np.diag(c.Pa))) or np.any(np.diag(c.Pa) <= 0):
            raise ValueError('Pa must be diagonal and strictly positive, as in MATLAB.')
        scalars = [c.qx, c.qy, c.chi_v, c.chi_u, c.initialThetaGuess,
                   c.initialV1Guess, c.initialU1Guess, c.relCostTol,
                   c.armijoC, c.predictionMaxStep]
        if not (np.isfinite(scalars).all() and np.isfinite(c.Pa).all()
                and np.isfinite(c.PfInv).all() and np.isfinite(c.terminalReference).all()):
            raise ValueError('Configuration values must be finite.')
        if min(c.chi_v, c.chi_u, c.predictionMaxStep) <= 0:
            raise ValueError('chi_v, chi_u and predictionMaxStep must be positive.')
        self.reset()

    def reset(self):
        self.tBuf, self.rhoBuf, self.alpha2Buf = [], [], []
        self.controlTBuf, self.controlVBuf, self.controlUBuf = [], [], []
        self.prevSolution = None
        self.zPred = None
        self.tPred = self.vSelfPred = self.uSelfPred = math.nan
        self.lastOutput = self.empty_output()

    def step(self, t, rho, alpha2, vSelf, uSelf):
        t, rho, alpha2, vSelf, uSelf = map(float, (t, rho, alpha2, vSelf, uSelf))
        self.validate_sample(t, rho, alpha2, vSelf, uSelf)
        self.record_control_sample(t, vSelf, uSelf)
        self.tBuf.append(t)
        self.rhoBuf.append(rho)
        self.alpha2Buf.append(wrap_pi(alpha2))
        maxNodes = self.cfg.Nh + 1
        windowShifted = len(self.tBuf) > maxNodes
        if windowShifted:
            self.tBuf = self.tBuf[-maxNodes:]
            self.rhoBuf = self.rhoBuf[-maxNodes:]
            self.alpha2Buf = self.alpha2Buf[-maxNodes:]
        self.prune_control_history()
        nSamples = len(self.tBuf)
        if nSamples < self.cfg.minSamples:
            self.lastOutput = self.bootstrap_output(t, rho, alpha2)
            return self.lastOutput.copy()
        Y = np.array([np.array(self.rhoBuf)*np.cos(self.alpha2Buf),
                      np.array(self.rhoBuf)*np.sin(self.alpha2Buf)])
        currentM = (nSamples-1)*self.cfg.nSub
        prev = self.prevSolution
        if prev is None:
            zbar = np.array([Y[0, 0], Y[1, 0], self.cfg.initialThetaGuess,
                             self.cfg.initialV1Guess, self.cfg.initialU1Guess])
            z0 = zbar.copy()
            W = np.zeros((2, currentM))
            maxIter = self.cfg.maxIterBoot
        elif not windowShifted:
            zbar = prev['zbar'].copy()
            z0 = prev['Z'][:, 0].copy()
            nAppend = currentM-prev['W'].shape[1]
            if nAppend < 0:
                raise RuntimeError('Internal growing-horizon warm-start size mismatch.')
            W = np.concatenate((prev['W'], np.repeat(prev['W'][:, -1:], nAppend, axis=1)), axis=1)
            maxIter = self.cfg.maxIterRT
        else:
            zbar = prev['Z'][:, self.cfg.nSub].copy()
            z0 = zbar.copy()
            W = np.concatenate((prev['W'][:, self.cfg.nSub:],
                                np.repeat(prev['W'][:, -1:], self.cfg.nSub, axis=1)), axis=1)
            if W.shape[1] != currentM:
                raise RuntimeError('Internal receding-horizon warm-start size mismatch.')
            maxIter = self.cfg.maxIterRT
        z0, W, Z, info = self.solve_window(z0, W, Y, self.tBuf,
            self.controlTBuf, self.controlVBuf, self.controlUBuf,
            zbar, self.cfg.terminalReference, maxIter)
        self.prevSolution = dict(Z=Z, W=W, lambdaPlus=info['lambdaPlus'],
            lambdaMinus=info['lambdaMinus'], zbar=zbar, tMeas=np.array(self.tBuf), info=info)
        self.zPred = Z[:, -1].copy()
        self.tPred, self.vSelfPred, self.uSelfPred = t, vSelf, uSelf
        self.lastOutput = self.output_from_state(self.zPred, t, info['cost'],
            info['iter'], False, t, nSamples, self.tBuf[-1]-self.tBuf[0])
        return self.lastOutput.copy()

    def predict(self, t, vSelf, uSelf):
        t, vSelf, uSelf = map(float, (t, vSelf, uSelf))
        self.validate_prediction_input(t, vSelf, uSelf)
        self.record_control_sample(t, vSelf, uSelf)
        if self.zPred is None:
            self.lastOutput = dict(self.lastOutput, t=t, isPrediction=True)
            return self.lastOutput.copy()
        if t-self.tPred > 0:
            self.zPred = self.propagate_prediction(self.zPred, self.tPred, t,
                self.vSelfPred, self.uSelfPred, vSelf, uSelf)
            self.tPred = t
        self.vSelfPred, self.uSelfPred = vSelf, uSelf
        self.lastOutput = self.output_from_state(self.zPred, t,
            self.lastOutput['cost'], 0, True, self.tBuf[-1], len(self.tBuf),
            self.tBuf[-1]-self.tBuf[0])
        return self.lastOutput.copy()

    def getLastEstimate(self):
        return self.lastOutput.copy()

    def solve_window(self, z0, W, Y, tMeas, controlT, controlV, controlU,
                     zbar, zfbar, maxIter):
        c = self.cfg
        PaInv = np.diag(1/np.diag(c.Pa))
        tFine, measNodes = self.make_fine_grid(tMeas)
        # Inputs and grid do not depend on optimization variables; cache once.
        known = self.known_controls(controlT, controlV, controlU, tFine)
        J, Z = self.forward_cost(z0, W, Y, zbar, zfbar, tFine, measNodes, PaInv, known)
        history = np.full(maxIter+1, np.nan)
        history[0] = J
        iteration = 0
        for it in range(1, maxIter+1):
            iteration = it
            _, _, g0, gW = self.backward_sweep(z0, W, Z, Y, zbar, zfbar,
                                              tFine, measNodes, known, PaInv)
            dz = -c.Pa @ g0
            dW = -gW / (np.array([c.chi_v, c.chi_u])[:, None]*np.diff(tFine))
            directionalDerivative = float(g0 @ dz + np.sum(gW*dW))
            if directionalDerivative >= -1e-14:
                break
            step = 1.0
            accepted = False
            for _ in range(c.maxLineSearch):
                zTry, WTry = z0+step*dz, W+step*dW
                JTry, ZTry = self.forward_cost(zTry, WTry, Y, zbar, zfbar,
                                               tFine, measNodes, PaInv, known)
                if JTry <= J+c.armijoC*step*directionalDerivative:
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
            relDecrease = abs(J-JTry)/max(1.0, abs(J))
            z0, W, Z, J = zTry, WTry, ZTry, JTry
            history[it] = J
            if relDecrease < c.relCostTol:
                break
        lp, lm, _, _ = self.backward_sweep(z0, W, Z, Y, zbar, zfbar,
                                           tFine, measNodes, known, PaInv)
        return z0, W, Z, dict(iter=iteration, cost=J,
            costHistory=history[:iteration+1], lambdaPlus=lp, lambdaMinus=lm)

    @staticmethod
    def known_controls(controlT, controlV, controlU, tFine):
        known = {}
        for suffix, times in [('L', tFine[:-1]), ('R', tFine[1:]),
                              ('M', 0.5*(tFine[:-1]+tFine[1:]))]:
            known['v'+suffix] = _interp_linear(controlT, controlV, times)
            known['u'+suffix] = _interp_linear(controlT, controlU, times)
        return known

    def forward_cost(self, z0, W, Y, zbar, zfbar, tFine, measNodes, PaInv, known):
        M = len(tFine)-1
        Z = np.zeros((5, M+1))
        Z[:, 0] = z0
        for j, h in enumerate(np.diff(tFine)):
            Z[:, j+1] = self.rk4(Z[:, j], W[:, j], h,
                known['vL'][j], known['uL'][j], known['vM'][j], known['uM'][j],
                known['vR'][j], known['uR'][j])
        dz0 = z0-zbar
        J = 0.5*float(dz0 @ PaInv @ dz0)
        for k, node in enumerate(measNodes):
            ex, ey = Z[:2, node]-Y[:, k]
            J += 0.5*self.cfg.qx*ex**2 + 0.5*self.cfg.qy*ey**2
        for j, h in enumerate(np.diff(tFine)):
            J += 0.5*h*(self.cfg.chi_v*W[0, j]**2+self.cfg.chi_u*W[1, j]**2)
        dzf = Z[:, -1]-zfbar
        J += 0.5*float(dzf @ self.cfg.PfInv @ dzf)
        return float(J), Z

    def backward_sweep(self, z0, W, Z, Y, zbar, zfbar, tFine, measNodes, known, PaInv):
        M = len(tFine)-1
        lp, lm = np.zeros((5, M+1)), np.zeros((5, M+1))
        measAtNode = np.full(M+1, -1, dtype=int)
        measAtNode[measNodes] = np.arange(len(measNodes))
        lp[:, -1] = self.cfg.PfInv @ (Z[:, -1]-zfbar)
        lm[:, -1] = lp[:, -1]+self.measurement_gradient(Z[:, -1], Y[:, measAtNode[-1]])
        for j in range(M-1, -1, -1):
            h = -(tFine[j+1]-tFine[j])
            lamR = lm[:, j+1]
            zR, zL = Z[:, j+1], Z[:, j]
            zM = 0.5*(zL+zR)
            k1 = self.costate_rhs(zR, lamR, known['uR'][j])
            k2 = self.costate_rhs(zM, lamR+0.5*h*k1, known['uM'][j])
            k3 = self.costate_rhs(zM, lamR+0.5*h*k2, known['uM'][j])
            k4 = self.costate_rhs(zL, lamR+h*k3, known['uL'][j])
            lp[:, j] = lamR+(h/6)*(k1+2*k2+2*k3+k4)
            kMeas = measAtNode[j]
            lm[:, j] = lp[:, j]
            if kMeas >= 0:
                lm[:, j] += self.measurement_gradient(Z[:, j], Y[:, kMeas])
        g0 = PaInv @ (z0-zbar)+lm[:, 0]
        lamAvg = 0.5*(lp[:, :-1]+lm[:, 1:])
        gW = np.diff(tFine)*(np.array([self.cfg.chi_v, self.cfg.chi_u])[:, None]*W+lamAvg[3:5])
        return lp, lm, g0, gW

    def measurement_gradient(self, z, yMeas):
        return np.array([self.cfg.qx*(z[0]-yMeas[0]),
                         self.cfg.qy*(z[1]-yMeas[1]), 0., 0., 0.])

    @staticmethod
    def relative_dynamics(z, w, vSelf, uSelf):
        x, y, theta, v1, u1 = z
        return np.array([v1*np.cos(theta)-vSelf+uSelf*y,
                         v1*np.sin(theta)-uSelf*x, u1-uSelf, w[0], w[1]])

    @staticmethod
    def costate_rhs(z, lam, uSelf):
        theta, v1 = z[2:4]
        lx, ly, ltheta = lam[:3]
        return np.array([uSelf*ly, -uSelf*lx,
            v1*lx*np.sin(theta)-v1*ly*np.cos(theta),
            -lx*np.cos(theta)-ly*np.sin(theta), -ltheta])

    def make_fine_grid(self, tMeas):
        fine = [tMeas[0]]
        nodes = [0]
        for left, right in zip(tMeas[:-1], tMeas[1:]):
            fine.extend(np.linspace(left, right, self.cfg.nSub+1)[1:])
            nodes.append(len(fine)-1)
        return np.array(fine), np.array(nodes, dtype=int)

    def rk4(self, z, w, h, v0, u0, vm, um, v1, u1):
        k1 = self.relative_dynamics(z, w, v0, u0)
        k2 = self.relative_dynamics(z+0.5*h*k1, w, vm, um)
        k3 = self.relative_dynamics(z+0.5*h*k2, w, vm, um)
        k4 = self.relative_dynamics(z+h*k3, w, v1, u1)
        return z+(h/6)*(k1+2*k2+2*k3+k4)

    def propagate_prediction(self, z, t0, t1, vSelf0, uSelf0, vSelf1, uSelf1):
        totalDt = t1-t0
        if totalDt <= 0:
            return z.copy()
        n = max(1, math.ceil(totalDt/self.cfg.predictionMaxStep))
        h = totalDt/n
        z, w = z.copy(), np.zeros(2)
        for j in range(n):
            tau0 = j*h
            s0, sM, s1 = tau0/totalDt, (tau0+0.5*h)/totalDt, (tau0+h)/totalDt
            v0, vm, v1 = [(1-s)*vSelf0+s*vSelf1 for s in (s0, sM, s1)]
            u0, um, u1 = [(1-s)*uSelf0+s*uSelf1 for s in (s0, sM, s1)]
            z = self.rk4(z, w, h, v0, u0, vm, um, v1, u1)
        z[2] = wrap_pi(z[2])
        return z

    def record_control_sample(self, t, vSelf, uSelf):
        if self.controlTBuf:
            tol = 1e-12*max(1, abs(t))
            if t < self.controlTBuf[-1]-tol:
                raise ValueError('Control/prediction timestamps must be nondecreasing; delayed samples are unsupported.')
            if abs(t-self.controlTBuf[-1]) <= tol:
                self.controlTBuf[-1], self.controlVBuf[-1], self.controlUBuf[-1] = t, vSelf, uSelf
                return
        self.controlTBuf.append(t)
        self.controlVBuf.append(vSelf)
        self.controlUBuf.append(uSelf)

    def prune_control_history(self):
        if not self.tBuf or not self.controlTBuf:
            return
        idx = int(np.searchsorted(self.controlTBuf, self.tBuf[0], side='right'))-1
        if idx > 0:
            self.controlTBuf = self.controlTBuf[idx:]
            self.controlVBuf = self.controlVBuf[idx:]
            self.controlUBuf = self.controlUBuf[idx:]

    def output_from_state(self, z, t, cost, iterations, isPrediction,
                          lastMeasurementTime, horizonSamples, horizonDuration):
        alpha2 = math.atan2(z[1], z[0])
        return dict(ready=True, t=t, x=float(z[0]), y=float(z[1]), theta=wrap_pi(z[2]),
            v1=float(z[3]), u1=float(z[4]), rho=math.hypot(z[0], z[1]),
            alpha2=wrap_pi(alpha2), alpha1=wrap_pi(alpha2+math.pi-z[2]),
            cost=float(cost), iterations=iterations, isPrediction=isPrediction,
            lastMeasurementTime=lastMeasurementTime, predictionAge=t-lastMeasurementTime,
            horizonSamples=horizonSamples, horizonDuration=horizonDuration,
            windowFull=horizonSamples >= self.cfg.Nh+1)

    def validate_prediction_input(self, t, vSelf, uSelf):
        if not np.isfinite([t, vSelf, uSelf]).all():
            raise ValueError('Prediction inputs must be finite.')
        if self.zPred is not None and t < self.tPred-1e-12*max(1, abs(t)):
            raise ValueError('predict() timestamps must be nondecreasing.')

    def validate_sample(self, t, rho, alpha2, vSelf, uSelf):
        if not np.isfinite([t, rho, alpha2, vSelf, uSelf]).all():
            raise ValueError('All estimator inputs must be finite.')
        if rho <= 0:
            raise ValueError('rho must be strictly positive.')
        if self.tBuf and t <= self.tBuf[-1]:
            raise ValueError('Measurement timestamps must be strictly increasing.')

    def bootstrap_output(self, t, rho, alpha2):
        out = self.empty_output()
        out.update(t=t, x=rho*math.cos(alpha2), y=rho*math.sin(alpha2), rho=rho,
            alpha2=wrap_pi(alpha2), isPrediction=False, lastMeasurementTime=t,
            predictionAge=0., horizonSamples=len(self.tBuf),
            horizonDuration=self.tBuf[-1]-self.tBuf[0], windowFull=False)
        return out

    @staticmethod
    def empty_output():
        out = dict.fromkeys(('t', 'x', 'y', 'theta', 'v1', 'u1', 'rho', 'alpha2',
                             'alpha1', 'cost', 'lastMeasurementTime', 'predictionAge'), math.nan)
        out.update(ready=False, iterations=0, isPrediction=False,
                   horizonSamples=0, horizonDuration=0., windowFull=False)
        return out


GrowMHE = growMHE
