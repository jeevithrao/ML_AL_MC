import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_fscore_support, roc_curve)

def per_class_auc(Y, P):
    out = np.full(Y.shape[1], np.nan)
    for c in range(Y.shape[1]):
        pos = Y[:, c].sum()
        if 0 < pos < len(Y):
            out[c] = roc_auc_score(Y[:, c], P[:, c])
    return out

def metric_row(Y, P, thr):
    Yi, pred = Y.astype(int), (P > thr)
    n = np.arange(len(Y))
    top1 = P.argmax(1)
    top5 = np.argsort(-P, 1)[:, :5]
    auc = per_class_auc(Y, P)
    ap = [average_precision_score(Yi[:, c], P[:, c]) for c in range(Y.shape[1]) if Yi[:, c].sum() > 0]
    mp, mr, mf, _ = precision_recall_fscore_support(Yi, pred, average="macro", zero_division=0)
    _, _, uf, _ = precision_recall_fscore_support(Yi, pred, average="micro", zero_division=0)
    return {"threshold": round(float(thr), 2), "macro ROC-AUC": np.nanmean(auc), "macro mAP": np.mean(ap),
            "top-1 acc": Y[n, top1].mean(), "top-5 acc": Y[n[:, None], top5].max(1).mean(),
            "macro precision": mp, "macro recall": mr, "macro F1": mf, "micro F1": uf}

def best_threshold(Yv, Pv, grid=np.arange(0.05, 0.96, 0.05)):
    f = [precision_recall_fscore_support(Yv.astype(int), Pv > t, average="macro", zero_division=0)[2] for t in grid]
    return round(float(grid[int(np.argmax(f))]), 2), f, grid

def confusion_summary(prim_idx, P, Y, names, k_pairs=12, k_heat=15):
    top1 = P.argmax(1)
    wrong = Y[np.arange(len(Y)), top1] == 0          # top-1 is not any labelled species
    C = np.zeros((P.shape[1], P.shape[1]), int)
    np.add.at(C, (prim_idx[wrong], top1[wrong]), 1)
    pairs = pd.DataFrame([(names[i], names[j], int(C[i, j]), round(C[i, j] / max(1, (prim_idx == i).sum()), 3))
                          for i, j in zip(*np.nonzero(C))], columns=["true species", "predicted instead", "chunks", "share of true species' chunks"])
    pairs = pairs.sort_values("chunks", ascending=False).head(k_pairs).reset_index(drop=True)
    inv = C.sum(1) + C.sum(0)
    sel = np.argsort(-inv)[:k_heat]
    cnt = np.bincount(prim_idx, minlength=P.shape[1])
    M = C[np.ix_(sel, sel)] / np.maximum(cnt[sel][:, None], 1)
    return pairs, sel, M, float(wrong.mean())


def roc_figure(models, names, path, title="ROC curves on the test split"):
    """models: {label: (Y, P)}. One panel per model: micro-average plus the worst two, median and best species."""
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 5), squeeze=False)
    for ax, (label, (Y, P)) in zip(axes[0], models.items()):
        auc = per_class_auc(Y, P)
        valid = np.where(~np.isnan(auc))[0]; srt = valid[np.argsort(auc[valid])]
        picks = [srt[0], srt[1], srt[len(srt) // 2], srt[-1]]
        fpr, tpr, _ = roc_curve(Y.ravel().astype(int), P.ravel())
        ax.plot(fpr, tpr, "k--", lw=1.6, label=f"micro-average (AUC {roc_auc_score(Y.ravel().astype(int), P.ravel()):.3f})")
        for c in picks:
            f, t, _ = roc_curve(Y[:, c].astype(int), P[:, c]); ax.plot(f, t, lw=1.3, label=f"{names[c]} (AUC {auc[c]:.3f})")
        ax.plot([0, 1], [0, 1], ":", c="gray", lw=1)
        ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
        ax.set_title(f"{label}\nmacro AUC {np.nanmean(auc):.4f}", fontsize=10); ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=.3)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=140); return fig

def confusion_figure(M, sel, names, path, title="Most-confused species (test split, row-normalised)"):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(0.3, float(M.max())))
    lab = [names[i] for i in sel]
    ax.set_xticks(range(len(sel))); ax.set_xticklabels(lab, rotation=75, ha="right", fontsize=7)
    ax.set_yticks(range(len(sel))); ax.set_yticklabels(lab, fontsize=7)
    ax.set_xlabel("predicted (top-1)"); ax.set_ylabel("true primary species"); ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="share of the true species' chunks"); fig.tight_layout(); fig.savefig(path, dpi=140); return fig

def f1_figure(Y, P, thr, ntr, path):
    _, _, f1, sup = precision_recall_fscore_support(Y.astype(int), P > thr, average=None, zero_division=0)
    rare = ntr <= np.quantile(ntr, 0.25)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(ntr[~rare], f1[~rare], s=14, label="other species"); ax.scatter(ntr[rare], f1[rare], s=18, c="C3", label="rare (bottom 25% by training chunks)")
    ax.set_xscale("log"); ax.set_xlabel("training chunks per species"); ax.set_ylabel(f"test F1 at threshold {thr:.2f}")
    ax.set_title("Per-species F1 vs amount of training data", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=140)
    return dict(f1_rare=float(f1[rare].mean()), f1_other=float(f1[~rare].mean()), n_rare=int(rare.sum()))
