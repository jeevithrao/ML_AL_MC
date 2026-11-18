import copy, io, os, time, numpy as np, torch
from torch.utils.flop_counter import FlopCounterMode

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def size_mb(m):
    buf = io.BytesIO(); torch.save(m.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6

def gflops(m, shape=(1, 1, 128, 313)):
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(torch.randn(*shape))
    return fc.get_total_flops() / 1e9

@torch.no_grad()
def cpu_latency_ms(m, batch, runs, warmup=10):
    x = torch.randn(batch, 1, 128, 313)
    for _ in range(warmup): m(x)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter(); m(x); ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1000)

def mel_ms(n=30):
    import librosa
    from audio_config import SR, CHUNK_SEC, N_FFT, HOP, N_MELS, FMIN, FMAX, TOP_DB
    y = (np.random.randn(SR * CHUNK_SEC) * 0.1).astype(np.float32)
    def one():
        m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                           n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
        return librosa.power_to_db(m, ref=np.max, top_db=TOP_DB)
    for _ in range(3): one()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); one(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1000)

def profile(name, model, extra=None, threads=2):
    m = copy.deepcopy(model).eval().cpu()
    torch.set_num_threads(threads)
    mel, l1, l32 = mel_ms(), cpu_latency_ms(m, 1, 100), cpu_latency_ms(m, 32, 20)
    row = dict(model=name, params_M=round(count_params(m) / 1e6, 3), size_MB=round(size_mb(m), 2),
               gflops=round(gflops(m), 3), lat_b1_ms=round(l1, 1), lat_b32_ms=round(l32, 1),
               mel_ms=round(mel, 1), threads=threads,
               win_per_s_b1=round(1000 / (mel + l1), 1),
               win_per_s_b32=round(32 * 1000 / (32 * mel + l32), 1))
    if extra: row.update(extra)
    return row

def log_row(row, path="results/efficiency.csv"):
    import pandas as pd
    df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    if len(df): df = df[df.model != row["model"]]
    pd.concat([df, pd.DataFrame([row])]).to_csv(path, index=False)
