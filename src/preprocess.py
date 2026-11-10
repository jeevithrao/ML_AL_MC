import numpy as np, soundfile as sf, librosa
from audio_config import *

def _load(path):
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        y = f.read(frames=int(MAX_READ_SEC * sr), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y

def _windows(y):
    L = SR * CHUNK_SEC
    n_full = len(y) // L
    wins = [y[i*L:(i+1)*L] for i in range(n_full)]
    starts = [i * CHUNK_SEC for i in range(n_full)]
    rem = y[n_full * L:]
    if n_full == 0 or len(rem) >= SR * MIN_TAIL_SEC:
        if len(rem) == 0:
            rem = np.zeros(L, dtype=np.float32)
        wins.append(np.resize(rem, L))        # repeat the audio to fill the window
        starts.append(n_full * CHUNK_SEC)
    return wins, starts

def clip_to_mels(path):
    y = _load(path)
    wins, starts = _windows(y)
    rms = np.array([np.sqrt(np.mean(w.astype(np.float64) ** 2)) for w in wins])
    k = min(N_CHUNKS, len(wins))
    keep = np.sort(np.argsort(rms)[::-1][:k])  # loudest k, back in time order
    mels = []
    for j in keep:
        m = librosa.feature.melspectrogram(
            y=wins[j], sr=SR, n_fft=N_FFT, hop_length=HOP,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
        mels.append(librosa.power_to_db(m, ref=np.max, top_db=TOP_DB).astype(np.float16))
    return np.stack(mels), np.array(starts)[keep], rms[keep].astype(np.float32)
