import ast, numpy as np, torch
from torch.utils.data import Dataset

def build_targets(idx, classes):
    cid = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(idx), len(classes)), dtype=np.float32)
    for r, (p, s) in enumerate(zip(idx.primary_label, idx.secondary_labels)):
        Y[r, cid[p]] = 1.0
        if isinstance(s, str):
            for c in ast.literal_eval(s):
                if c in cid:
                    Y[r, cid[c]] = 1.0
    return Y

def sqrt_inv_weights(labels):
    import pandas as pd
    s = pd.Series(labels)
    cnt = s.map(s.value_counts()).values
    return (1.0 / np.sqrt(cnt)).astype(np.float64)

class MelDataset(Dataset):
    def __init__(self, path, rows, Y):
        self.path, self.rows, self.Y = path, np.asarray(rows), Y
        self.mm = None
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        if self.mm is None:                      # opened lazily, once per worker
            self.mm = np.load(self.path, mmap_mode="r")
        r = self.rows[i]
        x = torch.from_numpy(np.array(self.mm[r], dtype=np.float32))
        return x, torch.from_numpy(self.Y[r]), int(r)
