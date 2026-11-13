import os, gc, json, time, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from engine import DEV, prep, time_shift, spec_augment, mixup
from train_config import *
from al_config import *
from data_utils import MelDataset, sqrt_inv_weights
from metrics import per_class_auc
from model import make_model

def _loader(ds, train=False, weights=None):
    if train:
        smp = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
        return DataLoader(ds, batch_size=BATCH, sampler=smp, num_workers=NUM_WORKERS,
                          pin_memory=True, drop_last=True)
    return DataLoader(ds, batch_size=BATCH * 2, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)

def build_context(idx, Y, rows, classes, path):
    tr = rows["train"]
    fid, files = pd.factorize(idx.filename.values[tr])
    prim = idx.primary_label.values[tr]
    file_prim = pd.Series(prim).groupby(fid).first().values
    n_train = pd.Series(prim).value_counts().reindex(classes).values
    rare = n_train <= np.quantile(n_train, RARE_QUANTILE)
    return dict(path=path, Y=Y, TR=tr, tr_fid=fid, tr_prim=prim, n_files=len(files),
                file_prim=file_prim, rare=rare, rare_names=list(np.array(classes)[rare]),
                classes=classes, n_train=n_train,
                pool_ds=MelDataset(path, tr, Y),
                val_ds=MelDataset(path, rows["val"], Y), Yv=Y[rows["val"]])

def make_seed(file_prim, seed):
    rng = np.random.RandomState(seed)
    lab = np.zeros(len(file_prim), bool)
    for c in np.unique(file_prim):                       # one file per species first
        lab[rng.choice(np.where(file_prim == c)[0])] = True
    need = int(round(SEED_FRAC * len(file_prim))) - int(lab.sum())
    if need > 0:                                         # then fill randomly
        lab[rng.choice(np.where(~lab)[0], need, replace=False)] = True
    return lab

def train_round(rows, prim, Y, path, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    ds = MelDataset(path, rows, Y)
    tl = _loader(ds, True, sqrt_inv_weights(prim))
    spe = len(tl)
    epochs = int(max(AL_EPOCHS, np.ceil(MIN_STEPS / spe)))
    model = make_model().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs * spe, pct_start=PCT_START)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        for x, y, _ in tl:
            x = prep(x.to(DEV, non_blocking=True)); y = y.to(DEV, non_blocking=True)
            x = spec_augment(time_shift(x))
            if np.random.rand() < MIXUP_P:
                x, y = mixup(x, y)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            loss = crit(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
    return model, epochs

@torch.no_grad()
def predict_ds(model, ds):
    model.eval(); out = []
    for x, _, _ in _loader(ds):
        x = prep(x.to(DEV, non_blocking=True))
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(model(x).float()).cpu())
    return torch.cat(out).numpy()

@torch.no_grad()
def score_pool(model, ds):
    model.eval(); P, E = [], []
    clf = model.get_classifier()
    for x, _, _ in _loader(ds):
        x = prep(x.to(DEV, non_blocking=True))
        with torch.autocast("cuda", dtype=torch.float16):
            z = model.forward_head(model.forward_features(x), pre_logits=True)
            lg = clf(z)
        P.append(torch.sigmoid(lg.float()).cpu())
        E.append(nn.functional.normalize(z.float(), dim=1).cpu())
    return torch.cat(P).numpy(), torch.cat(E).numpy()

def chunk_uncertainty(P, kind):
    if kind == "entropy":
        p = np.clip(P, 1e-6, 1 - 1e-6)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p)).mean(1)
    if kind == "margin":
        top2 = np.partition(P, -2, axis=1)[:, -2:]
        return -(top2.max(1) - top2.min(1))              # small gap -> high score
    raise ValueError(kind)

def select_uncertain(score, tr_fid, lab_f, k):
    fs = pd.Series(score).groupby(tr_fid).max().reindex(range(len(lab_f))).values
    fs = np.where(lab_f, -np.inf, fs)
    return np.argsort(-fs)[:k]

def select_coreset(E, tr_fid, lab_f, k):
    F = len(lab_f)
    fid = torch.from_numpy(tr_fid).to(DEV)
    FE = torch.zeros(F, E.shape[1], device=DEV).index_add_(0, fid, torch.from_numpy(E).to(DEV))
    FE = nn.functional.normalize(FE, dim=1)
    lab = torch.from_numpy(np.where(lab_f)[0]).to(DEV)
    mind = torch.full((F,), float("inf"), device=DEV)
    for s in range(0, len(lab), 1024):
        mind = torch.minimum(mind, torch.cdist(FE, FE[lab[s:s + 1024]]).min(1).values)
    mind[lab] = -1.0
    picked = []
    for _ in range(k):
        j = int(torch.argmax(mind)); picked.append(j)
        mind = torch.minimum(mind, torch.cdist(FE, FE[j:j + 1]).squeeze(1))
        mind[j] = -1.0
    return np.array(picked)

def run_strategy(name, seed, ctx, out_dir, rounds=N_ROUNDS):
    assert name in ("random", "entropy", "margin", "coreset")
    os.makedirs(out_dir, exist_ok=True)
    sp = f"{out_dir}/{name}_s{seed}.json"
    F = ctx["n_files"]; k = int(round(QUERY_FRAC * F))
    if os.path.exists(sp):
        st = json.load(open(sp))
        lab_f = np.zeros(F, bool); lab_f[st["labeled"]] = True; hist = st["hist"]
        print(f"[{name} s{seed}] resuming after {len(hist)} finished rounds")
    else:
        lab_f, hist = make_seed(ctx["file_prim"], seed), []
    for r in range(len(hist), rounds + 1):
        m = lab_f[ctx["tr_fid"]]
        t0 = time.time()
        model, ep = train_round(ctx["TR"][m], ctx["tr_prim"][m], ctx["Y"], ctx["path"], seed * 1000 + r)
        t_train = time.time() - t0
        a = per_class_auc(ctx["Yv"], predict_ds(model, ctx["val_ds"]))
        rec = dict(round=r, n_files=int(lab_f.sum()), n_chunks=int(m.sum()), frac=float(lab_f.mean()),
                   epochs=ep, macro_auc=float(np.nanmean(a)), rare_auc=float(np.nanmean(a[ctx["rare"]])),
                   rare_files=int(np.isin(ctx["file_prim"][lab_f], ctx["rare_names"]).sum()),
                   train_sec=round(t_train))
        if r < rounds:
            t0 = time.time()
            if name == "random":
                pick = np.random.RandomState(seed * 1000 + r).choice(np.where(~lab_f)[0], k, replace=False)
            else:
                P, E = score_pool(model, ctx["pool_ds"])
                pick = (select_coreset(E, ctx["tr_fid"], lab_f, k) if name == "coreset"
                        else select_uncertain(chunk_uncertainty(P, name), ctx["tr_fid"], lab_f, k))
            lab_f[pick] = True
            rec["select_sec"] = round(time.time() - t0)
        hist.append(rec)
        with open(sp + ".tmp", "w") as f:
            json.dump(dict(labeled=np.where(lab_f)[0].tolist(), hist=hist), f)
        os.replace(sp + ".tmp", sp)
        print(f"[{name} s{seed}] round {r}: {rec['n_files']} files ({rec['frac']:.0%}) | "
              f"macro {rec['macro_auc']:.4f} | rare {rec['rare_auc']:.4f} | {t_train:.0f}s train")
        del model; gc.collect(); torch.cuda.empty_cache()
    return hist
