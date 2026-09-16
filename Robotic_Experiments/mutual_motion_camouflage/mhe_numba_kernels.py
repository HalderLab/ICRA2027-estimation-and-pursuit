"""Compiled continuous-PMP/RK4 kernels. Float64; fastmath disabled.
Scratch arrays are reused within each sweep; accepted/trial trajectories swap.
No ROS, input dropping, solver parameter reduction, or timing-policy changes.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def dynamics_into(z, a, omega, vself, uself, out):
    theta, v = z[2], z[3]
    out[0] = v*math.cos(theta)-vself+uself*z[1]
    out[1] = v*math.sin(theta)-uself*z[0]
    out[2] = z[4]-uself
    out[3], out[4] = a, omega


@njit(cache=True)
def rk4_into(z, a, omega, h, vl, ul, vm, um, vr, ur, out, k, tmp):
    dynamics_into(z, a, omega, vl, ul, k[0])
    for i in range(5): tmp[i] = z[i]+.5*h*k[0,i]
    dynamics_into(tmp, a, omega, vm, um, k[1])
    for i in range(5): tmp[i] = z[i]+.5*h*k[1,i]
    dynamics_into(tmp, a, omega, vm, um, k[2])
    for i in range(5): tmp[i] = z[i]+h*k[2,i]
    dynamics_into(tmp, a, omega, vr, ur, k[3])
    for i in range(5):
        out[i] = z[i]+(h/6)*(k[0,i]+2*k[1,i]+2*k[2,i]+k[3,i])


@njit(cache=True)
def costate_into(z, lam, uself, out):
    theta, v = z[2], z[3]
    lx, ly, lt = lam[0], lam[1], lam[2]
    out[0], out[1] = uself*ly, -uself*lx
    out[2] = v*lx*math.sin(theta)-v*ly*math.cos(theta)
    out[3] = -lx*math.cos(theta)-ly*math.sin(theta)
    out[4] = -lt


@njit(cache=True)
def forward_into(z0, W, Y, zbar, zfbar, h, nodes, PaInv, PfInv, weights, known, Z):
    M = len(h)
    k, tmp = np.empty((4,5)), np.empty(5)
    Z[:,0] = z0
    for j in range(M):
        rk4_into(Z[:,j],W[0,j],W[1,j],h[j],known[0,j],known[1,j],
            known[2,j],known[3,j],known[4,j],known[5,j],Z[:,j+1],k,tmp)
    # General matrix products retained, including optional terminal cross terms.
    dz = z0-zbar
    J = .5*np.dot(np.dot(dz,PaInv),dz)
    for kmeas in range(len(nodes)):
        node = nodes[kmeas]
        ex,ey = Z[0,node]-Y[0,kmeas], Z[1,node]-Y[1,kmeas]
        J += .5*weights[0]*ex**2+.5*weights[1]*ey**2
    for j in range(M):
        J += .5*h[j]*(weights[2]*W[0,j]**2+weights[3]*W[1,j]**2)
    dzf = Z[:,-1]-zfbar
    J += .5*np.dot(np.dot(dzf,PfInv),dzf)
    return J


@njit(cache=True)
def backward_into(z0,W,Z,Y,zbar,zfbar,h,meas_at,PaInv,PfInv,weights,known,lp,lm,g0,gW):
    M = len(h)
    lp[:,-1] = np.dot(PfInv,Z[:,-1]-zfbar)
    lm[:,-1] = lp[:,-1]
    km = meas_at[-1]
    lm[0,-1] += weights[0]*(Z[0,-1]-Y[0,km])
    lm[1,-1] += weights[1]*(Z[1,-1]-Y[1,km])
    k, tmp, zm = np.empty((4,5)), np.empty(5), np.empty(5)
    for j in range(M-1,-1,-1):
        dt = -h[j]
        for i in range(5): zm[i] = .5*(Z[i,j]+Z[i,j+1])
        costate_into(Z[:,j+1],lm[:,j+1],known[5,j],k[0])
        for i in range(5): tmp[i] = lm[i,j+1]+.5*dt*k[0,i]
        costate_into(zm,tmp,known[3,j],k[1])
        for i in range(5): tmp[i] = lm[i,j+1]+.5*dt*k[1,i]
        costate_into(zm,tmp,known[3,j],k[2])
        for i in range(5): tmp[i] = lm[i,j+1]+dt*k[2,i]
        costate_into(Z[:,j],tmp,known[1,j],k[3])
        for i in range(5):
            lp[i,j] = lm[i,j+1]+(dt/6)*(k[0,i]+2*k[1,i]+2*k[2,i]+k[3,i])
            lm[i,j] = lp[i,j]
        km = meas_at[j]
        if km >= 0:
            lm[0,j] += weights[0]*(Z[0,j]-Y[0,km])
            lm[1,j] += weights[1]*(Z[1,j]-Y[1,km])
    g0[:] = np.dot(PaInv,z0-zbar)+lm[:,0]
    for j in range(M):
        gW[0,j] = h[j]*(weights[2]*W[0,j]+.5*(lp[3,j]+lm[3,j+1]))
        gW[1,j] = h[j]*(weights[3]*W[1,j]+.5*(lp[4,j]+lm[4,j+1]))


@njit(cache=True)
def solve(z0in,Win,Y,zbar,zfbar,h,nodes,Pa,PaInv,PfInv,weights,known,
          max_iter,max_ls,armijo_c,rel_tol):
    z0, W = z0in.copy(), Win.copy()
    M = len(h)
    Z, Ztry = np.empty((5,M+1)), np.empty((5,M+1))
    lp, lm = np.empty((5,M+1)), np.empty((5,M+1))
    g0, dz, ztry = np.empty(5), np.empty(5), np.empty(5)
    gW, dW, Wtry = np.empty((2,M)), np.empty((2,M)), np.empty((2,M))
    meas_at = np.full(M+1,-1,np.int64)
    for k in range(len(nodes)): meas_at[nodes[k]] = k
    history = np.full(max_iter+1,np.nan)
    J = forward_into(z0,W,Y,zbar,zfbar,h,nodes,PaInv,PfInv,weights,known,Z)
    history[0] = J
    iteration = 0
    for it in range(1,max_iter+1):
        iteration = it
        backward_into(z0,W,Z,Y,zbar,zfbar,h,meas_at,PaInv,PfInv,weights,known,lp,lm,g0,gW)
        dz[:] = -np.dot(Pa,g0)
        for j in range(M):
            dW[0,j] = -gW[0,j]/(h[j]*weights[2])
            dW[1,j] = -gW[1,j]/(h[j]*weights[3])
        # Match row-major residual order. No fastmath/reassociation requested.
        directional = np.dot(g0,dz)+np.sum(gW*dW)
        if directional >= -1e-14: break
        step,accepted = 1.,False
        for ls in range(max_ls):
            for i in range(5): ztry[i] = z0[i]+step*dz[i]
            for i in range(2):
                for j in range(M): Wtry[i,j] = W[i,j]+step*dW[i,j]
            Jtry = forward_into(ztry,Wtry,Y,zbar,zfbar,h,nodes,PaInv,PfInv,weights,known,Ztry)
            if Jtry <= J+armijo_c*step*directional:
                accepted = True
                break
            step *= .5
        if not accepted: break
        rel = abs(J-Jtry)/max(1.,abs(J))
        z0,ztry = ztry,z0
        W,Wtry = Wtry,W
        Z,Ztry = Ztry,Z
        J = Jtry
        history[it] = J
        if rel < rel_tol: break
    backward_into(z0,W,Z,Y,zbar,zfbar,h,meas_at,PaInv,PfInv,weights,known,lp,lm,g0,gW)
    return z0,W,Z,iteration,J,history[:iteration+1],lp,lm


@njit(cache=True)
def prediction(z_in,t0,t1,vs0,us0,vs1,us1,max_step):
    z = z_in.copy()
    total = t1-t0
    if total <= 0: return z
    n = max(1,math.ceil(total/max_step))
    h = total/n
    k,tmp,out = np.empty((4,5)), np.empty(5), np.empty(5)
    for j in range(n):
        tau = j*h
        s0,sm,s1 = tau/total,(tau+.5*h)/total,(tau+h)/total
        vl,vm,vr = (1-s0)*vs0+s0*vs1,(1-sm)*vs0+sm*vs1,(1-s1)*vs0+s1*vs1
        ul,um,ur = (1-s0)*us0+s0*us1,(1-sm)*us0+sm*us1,(1-s1)*us0+s1*us1
        rk4_into(z,0.,0.,h,vl,ul,vm,um,vr,ur,out,k,tmp)
        z,out = out,z
    z[2] = math.atan2(math.sin(z[2]),math.cos(z[2]))
    return z
