"""ROS-independent MHE process entry point."""
from queue import Empty, Full
import time

def mhe_worker(inbox, outbox):
    """Serializes all MHE history updates outside the ROS process."""
    def publish(value):
        # Latest outputs may replace old outputs, but input samples are never dropped.
        try:
            outbox.put_nowait(value)
        except Full:
            try:
                outbox.get_nowait()
                outbox.put_nowait(value)
            except (Empty, Full):
                pass
    try:
        from growMHE import growMHE
        mhe = growMHE()
        compile_started = time.perf_counter()
        mhe.warmup()
        publish({'worker_ready': True, 'compile_seconds': time.perf_counter()-compile_started})
        while True:
            event = inbox.get()
            if event is None:
                return
            kind, t, v, u, rho, bearing = event
            if kind == 'reset':
                mhe.reset()
                continue
            began = time.perf_counter()
            if kind == 'step':
                out = mhe.step(t, rho, bearing, v, u)
            else:
                out = mhe.predict(t, v, u)
            publish(dict(output=out, event_time=t, kind=kind,
                         runtime=time.perf_counter()-began))
    except BaseException as exc:
        publish({'error': f'{type(exc).__name__}: {exc}'})
        raise

