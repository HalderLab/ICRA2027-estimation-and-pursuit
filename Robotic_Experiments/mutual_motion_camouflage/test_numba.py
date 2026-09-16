"""Compiled/reference trajectory and adjoint parity, including terminal cross terms."""
import unittest
import numpy as np
from growMHE import growMHE, MHEConfig
from growMHE_reference import growMHE as Reference
from mhe_numba_kernels import solve, prediction

class Parity(unittest.TestCase):
    def test_full_solver_and_costates(self):
        rng=np.random.default_rng(7)
        c=MHEConfig()
        a=rng.normal(size=(5,5))*.1
        c.PfInv=a.T@a
        fast,ref=growMHE(c),Reference(c)
        for n in (5,31):
            t=np.cumsum(rng.uniform(.1,.15,n))
            ct=np.linspace(t[0]-.01,t[-1]+.01,4*n)
            cv=.15+.02*np.sin(ct); cu=.3*np.cos(ct)
            Y=rng.normal(size=(2,n))*.1+np.array([[1.],[.2]])
            z=np.array([1.,.2,.3,.1,.02]); W=rng.normal(size=(2,(n-1)*c.nSub))*.01
            args=(z,W,Y,t,ct,cv,cu,z+.01,np.zeros(5),c.maxIterRT)
            x,y=fast.solve_window(*args),ref.solve_window(*args)
            for i in range(3): np.testing.assert_allclose(x[i],y[i],atol=1e-10,rtol=1e-9)
            for k in ('cost','costHistory','lambdaPlus','lambdaMinus'):
                np.testing.assert_allclose(x[3][k],y[3][k],atol=1e-10,rtol=1e-9)
            self.assertEqual(x[3]['iter'],y[3]['iter'])

    def test_precompile_isolation_and_prediction(self):
        m=growMHE(); m.warmup()
        self.assertEqual(m.tBuf,[]); self.assertIsNone(m.prevSolution)
        z=np.array([1.,.2,3.14,.1,.3])
        np.testing.assert_allclose(m.propagate_prediction(z,0.,.5,.15,.2,.17,-.3),
            Reference().propagate_prediction(z,0.,.5,.15,.2,.17,-.3),atol=1e-12)
        self.assertTrue(solve.nopython_signatures)
        self.assertTrue(prediction.nopython_signatures)

if __name__=='__main__': unittest.main(verbosity=2)
