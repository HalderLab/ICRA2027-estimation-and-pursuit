#!/usr/bin/env python3
"""Numba-accelerated growMHE. Same step/predict API and MHEConfig defaults.

The unmodified Python reference supplies buffering, warm starts and outputs.
Only solve_window and prediction dispatch to compiled float64 kernels.
Numba is REQUIRED: no silent fallback to a backend known to lag on the Pi.
Call warmup() before reporting robot readiness. warmup uses a throwaway instance,
so no synthetic samples are retained in the real estimator.
"""
import numpy as np
from growMHE_reference import growMHE as ReferenceGrowMHE, MHEConfig, wrap_pi, _interp_linear
try:
    from mhe_numba_kernels import solve, prediction
except ImportError as exc:
    raise ImportError('Numba backend unavailable. Install compatible numpy, scipy, numba and llvmlite in this Python environment; see README_NUMBA_zh.md.') from exc


def f64(x):
    return np.ascontiguousarray(x,dtype=np.float64)


class growMHE(ReferenceGrowMHE):
    backend = 'numba'

    def solve_window(self,z0,W,Y,tMeas,controlT,controlV,controlU,zbar,zfbar,maxIter):
        c = self.cfg
        tf,nodes = self.make_fine_grid(tMeas)
        known = self.known_controls(controlT,controlV,controlU,tf)
        packed = f64([known[k] for k in ('vL','uL','vM','uM','vR','uR')])
        z0,W,Z,it,J,hist,lp,lm = solve(f64(z0),f64(W),f64(Y),f64(zbar),f64(zfbar),
            f64(np.diff(tf)),np.ascontiguousarray(nodes,dtype=np.int64),f64(c.Pa),
            f64(np.diag(1/np.diag(c.Pa))),f64(c.PfInv),f64([c.qx,c.qy,c.chi_v,c.chi_u]),
            packed,int(maxIter),int(c.maxLineSearch),float(c.armijoC),float(c.relCostTol))
        return z0,W,Z,dict(iter=it,cost=J,costHistory=hist,lambdaPlus=lp,lambdaMinus=lm)

    def propagate_prediction(self,z,t0,t1,vSelf0,uSelf0,vSelf1,uSelf1):
        return prediction(f64(z),float(t0),float(t1),float(vSelf0),float(uSelf0),
                          float(vSelf1),float(uSelf1),float(self.cfg.predictionMaxStep))

    def warmup(self):
        dummy = growMHE(self.cfg)
        for i in range(dummy.cfg.minSamples):
            dummy.step(i*.1,1.,.4,.15,.3)
        dummy.predict((dummy.cfg.minSamples-1)*.1+.03,.15,.3)
        # Warm start memory layout differs from initial layout; wrappers normalize
        # every solver argument, so growing/full horizons reuse these signatures.
        if not solve.nopython_signatures or not prediction.nopython_signatures:
            raise RuntimeError('Numba nopython compilation is not active')


GrowMHE = growMHE
