#!/usr/bin/env python3
"""ROS-free synthetic full-window benchmark and optional reference comparison."""
import argparse
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import time
import numpy as np
from growMHE import growMHE
from growMHE_reference import growMHE as Reference


def replay(cls, count):
    mhe = cls()
    times, outputs = [], []
    for i in range(count):
        t = i*.13
        rho = 1.+.13*np.sin(.7*t)
        angle = 2.9+.45*np.sin(.4*t)
        v,u = .15+.02*np.sin(t), .3*np.cos(.5*t)
        began = time.perf_counter()
        out = mhe.step(t,rho,angle,v,u)
        times.append((time.perf_counter()-began)*1000)
        outputs.append(out)
        for j in range(1,4):
            mhe.predict(t+j*.03, v+.001*j, u-.002*j)
    return mhe, times, outputs


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compare',action='store_true')
    ap.add_argument('--samples',type=int,default=65)
    args=ap.parse_args()
    if args.samples < 40: ap.error('--samples must be >=40 to cover a full moving window')
    began=time.perf_counter(); growMHE().warmup()
    print('Precompile/cache load: %.2f s'%(time.perf_counter()-began),flush=True)
    fast,dt,outs=replay(growMHE,args.samples)
    for name,values in [('growing',dt[4:30]),('full window',dt[30:])]:
        print('%s: median %.3f ms; p95 %.3f ms; max %.3f ms'%(name,np.median(values),np.percentile(values,95),max(values)),flush=True)
    if args.compare:
        ref,rdt,routs=replay(Reference,args.samples)
        keys=('x','y','theta','v1','u1','cost')
        errors=[]
        for a,b in zip(outs,routs):
            assert a['ready']==b['ready']
            if a['ready']:
                av,bv=[a[k] for k in keys],[b[k] for k in keys]
                np.testing.assert_allclose(av,bv,rtol=1e-8,atol=1e-9)
                errors.append(np.max(np.abs(np.array(av)-bv)))
        print('Reference full-window median %.3f ms; median speedup %.1fx'%(np.median(rdt[30:]),np.median(rdt[30:])/np.median(dt[30:])))
        print('Numerical comparison PASS; max output/cost absolute difference %.3g'%max(errors))
    print('Synthetic benchmark only; actual Pi sensor load and estimate age still need verification.')

if __name__=='__main__': main()
