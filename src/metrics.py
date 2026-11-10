import numpy as np
from sklearn.metrics import roc_auc_score

def per_class_auc(Y, P):
    out = np.full(Y.shape[1], np.nan)
    for c in range(Y.shape[1]):
        pos = Y[:, c].sum()
        if 0 < pos < len(Y):
            out[c] = roc_auc_score(Y[:, c], P[:, c])
    return out

def macro_auc(Y, P):
    a = per_class_auc(Y, P)
    return float(np.nanmean(a)), int(np.sum(~np.isnan(a)))
