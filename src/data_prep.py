import os, json, numpy as np, pandas as pd
from tqdm.auto import tqdm
from data_utils import build_targets

MEL_DIR = "/content/data/mels"
ALL = "/content/data/mels_all.npy"

def consolidate(idx, mel_dir=MEL_DIR, out=ALL):
    if os.path.exists(out):
        return out
    tmp = out + ".tmp"
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(idx), 128, 313))
    for fn, g in tqdm(idx.groupby("filename", sort=False), total=idx.filename.nunique()):
        z = np.load(f"{mel_dir}/{fn.replace('/', '__')}.npz")["mel"]
        lo, hi = g.index[0], g.index[-1] + 1
        assert hi - lo == len(g) == len(z), fn
        mm[lo:hi] = z
    mm.flush(); del mm
    os.replace(tmp, out)
    return out

def prepare(results_dir="results"):
    idx = pd.read_csv(f"{results_dir}/chunk_index.csv")
    classes = json.load(open(f"{results_dir}/classes.json"))
    path = consolidate(idx)
    Y = build_targets(idx, classes)
    rows = {s: np.where(idx.split == s)[0] for s in ["train", "val", "test"]}
    files = {s: set(idx.filename.values[r]) for s, r in rows.items()}
    assert not (files["train"] & files["val"]) and not (files["train"] & files["test"]) \
        and not (files["val"] & files["test"]), "split leakage"
    return idx, Y, rows, classes, path
