import time, numpy as np, onnxruntime as ort
from engine import prep, make_loader

def ort_session(path, threads=2):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

def ort_scores(path, ds):
    s, out = ort_session(path), []
    for x, _, _ in make_loader(ds, False):
        z = s.run(None, {"mel": prep(x).numpy()})[0]
        out.append(1 / (1 + np.exp(-z)))
    return np.concatenate(out)

def robust_ms(fn, blocks=5, runs=40, warmup=10):
    for _ in range(warmup): fn()
    meds = []
    for _ in range(blocks):
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        meds.append(np.median(ts) * 1000)
    return round(min(meds), 1), round(float(np.median(meds)), 1), round(max(meds), 1)

def ort_latency(path, batch, runs):
    s = ort_session(path); x = np.random.randn(batch, 1, 128, 313).astype(np.float32)
    return robust_ms(lambda: s.run(None, {"mel": x}), blocks=5, runs=max(4, runs // 2))[0]
