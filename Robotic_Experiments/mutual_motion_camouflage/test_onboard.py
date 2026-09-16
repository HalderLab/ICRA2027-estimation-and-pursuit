"""Offline tests; no ROS import, no cmd_vel publishers, no robot motion."""
import math
import multiprocessing as mp
from queue import Empty
import time
import unittest
import numpy as np
from mmc_protocol import LocalSession, Coordinator, ROBOTS
from mmc_math import MMCMath, wrap_pi
from mmc_worker import mhe_worker


class Simulation:
    def __init__(self):
        self.nodes = {r:LocalSession() for r in ROBOTS}
        self.coord = Coordinator('test-run')
        self.time = 0.
        self.statuses = {}
        self.last = {}
        self.seq = 0
        self.starts = {}
        self.runs = {}

    def advance(self, seconds, commands=True, drop_to=None, missing_status=None, ready=True, mhe=True):
        for _ in range(round(seconds/.01)):
            self.time = round(self.time+.01, 8)
            t, wall = self.time, 1000.+self.time
            if round(t*100) % 5 == 0:
                for r,n in self.nodes.items():
                    if r == missing_status:
                        continue
                    p = n.status(r, ready, mhe and n.start_mono is not None and t-n.start_mono>.45)
                    p['config_signature'] = 'matching'
                    self.statuses[r] = (p,t)
                    self.last[r] = (p,t)
                self.coord.update(t, wall, self.statuses)
                if commands and self.coord.phase != 'WAIT':
                    packet = self.coord.packet(wall, self.seq); self.seq += 1
                    for r,n in self.nodes.items():
                        if r != drop_to:
                            n.receive(packet,t,wall,ready)
            for r,n in self.nodes.items():
                peer = self.last.get('burger3' if r=='burger1' else 'burger1', (None,-math.inf))
                n.tick(t,ready,mhe and n.start_mono is not None and t-n.start_mono>.45,peer[0],t-peer[1])
                if n.just_started:
                    self.starts[r] = t
                if n.state == 'RUN' and r not in self.runs:
                    self.runs[r] = t


class ProtocolTests(unittest.TestCase):
    def test_no_start_without_command(self):
        s=Simulation(); s.advance(5,commands=False)
        self.assertEqual([n.state for n in s.nodes.values()], ['WAIT','WAIT'])
        self.assertFalse(s.starts)

    def test_complete_start_and_switch(self):
        s=Simulation(); s.advance(5)
        self.assertEqual([n.state for n in s.nodes.values()], ['RUN','RUN'])
        self.assertAlmostEqual(s.starts['burger1'],s.starts['burger3'])
        self.assertAlmostEqual(s.runs['burger1'],s.runs['burger3'])
        self.assertGreater(s.runs['burger1'],s.starts['burger1']+.45)

    def test_coordinator_loss_latches_stop(self):
        s=Simulation(); s.advance(5); s.advance(.8,commands=False)
        self.assertTrue(all(n.state=='STOPPED' for n in s.nodes.values()))
        s.advance(1)
        self.assertTrue(all(n.state=='STOPPED' for n in s.nodes.values()))

    def test_peer_loss(self):
        s=Simulation(); s.advance(5); s.advance(.8,missing_status='burger3')
        self.assertTrue(all(n.state=='STOPPED' for n in s.nodes.values()))

    def test_one_sided_start_delivery(self):
        s=Simulation(); s.advance(5,drop_to='burger3')
        self.assertFalse(s.starts)
        self.assertEqual(s.nodes['burger1'].state,'STOPPED')
        self.assertEqual(s.nodes['burger3'].state,'WAIT')

    def test_mhe_never_ready(self):
        s=Simulation(); s.advance(7,mhe=False)
        self.assertTrue(all(n.state=='STOPPED' for n in s.nodes.values()))
        self.assertFalse(s.runs)

    def test_sensor_loss_in_run(self):
        s=Simulation(); s.advance(5); s.advance(.1,ready=False)
        self.assertTrue(all(n.state=='STOPPED' for n in s.nodes.values()))

    def test_late_replay_and_clock(self):
        n=LocalSession()
        p=dict(action='PREPARE',run_id='r',seq=1,stamp=1000.,start_at=1002.)
        n.receive(p,0,1001.,True)
        self.assertEqual(n.state,'WAIT')
        n.receive(p,0,1000.,True)
        self.assertEqual(n.state,'ARMED')
        n.receive(dict(p,action='START'),.1,1000.1,True)
        self.assertEqual(n.state,'ARMED') # duplicate sequence cannot commit
        n.receive(dict(p,action='START',seq=2,stamp=1002.1),2.1,1002.1,True)
        self.assertEqual(n.state,'STOPPED')

    def test_different_run_rejected(self):
        n=LocalSession()
        p=dict(action='PREPARE',run_id='r',seq=1,stamp=1000.,start_at=1002.)
        n.receive(p,0,1000.,True)
        n.receive(dict(p,run_id='other',seq=2),0,1000.,True)
        self.assertEqual(n.state,'STOPPED')

    def test_shared_parameter_mismatch(self):
        c=Coordinator('r')
        p=dict(state='WAIT',ready=True,run_id='',mhe_ready=False)
        c.update(0,1000,{'burger1':(dict(p,config_signature='a'),0),
                         'burger3':(dict(p,config_signature='b'),0)})
        self.assertEqual(c.phase,'STOP')


class NumericalTests(unittest.TestCase):
    def test_mmc_equations_and_agent_signs(self):
        m=MMCMath()
        m.rho_min=.03; m.v1_nom=m.v3_nom=.15; m.mu=1.46225; m.E0=.00284423
        m.kd=1000.; m.use_dissipative=True; m.theta_diff_d=math.pi
        m.sync_gain_final=.1; m.sync_initial_fraction=.01
        m.sync_ramp_start=15.; m.sync_ramp_duration=5.; m.u_max=2.5
        for rho,a1,a2 in [(1.,1.2,1.5),(.3,-.2,.4),(1.4,2.,-1.)]:
            q=m.shape_to_mmc(np.array([rho,a1,a2]))
            lam=-.15*(math.sin(a1)+math.sin(a2))
            gamma=-.15*(math.cos(a1)+math.cos(a2))
            E=rho**2*lam**2*math.exp(-2*m.mu*rho)
            nominal=-m.mu*lam+1000*lam*gamma*(E-m.E0)
            err=wrap_pi(math.pi-a1+a2-math.pi)
            for robot,sign in [('burger1',-1),('burger3',1)]:
                got=m.compute_local_command(q,21.,robot)
                raw=nominal+sign*.5*.1*err
                self.assertAlmostEqual(got['u_self_cmd'],max(-2.5,min(2.5,raw)))
        self.assertAlmostEqual(m.sync_gain_effective(0),.001)
        self.assertAlmostEqual(m.sync_gain_effective(17.5),.0505)

    def test_real_worker_process(self):
        ctx=mp.get_context('spawn'); iq=ctx.Queue(64); oq=ctx.Queue(64)
        proc=ctx.Process(target=mhe_worker,args=(iq,oq)); proc.start()
        try:
            self.assertTrue(oq.get(timeout=120)['worker_ready'])
            iq.put(('reset',0,0,0,0,0))
            for k in range(8):
                t=.1*k
                if k: iq.put(('predict',t-.05,.15,0,0,0))
                iq.put(('step',t,.15,0,1.-.03*t,.4))
            outputs=[]
            limit=time.monotonic()+15
            while len(outputs)<15 and time.monotonic()<limit:
                outputs.append(oq.get(timeout=5))
            self.assertEqual(len(outputs),15)
            final=outputs[-1]['output']
            self.assertTrue(final['ready'])
            self.assertEqual(final['horizonSamples'],8)
            self.assertTrue(math.isfinite(final['rho']))
            iq.put(None); proc.join(timeout=5)
            self.assertEqual(proc.exitcode,0)
        finally:
            if proc.is_alive(): proc.terminate(); proc.join()
            iq.close(); oq.close()


if __name__=='__main__':
    unittest.main(verbosity=2)
