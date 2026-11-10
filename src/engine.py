import os, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from train_config import *
from metrics import macro_auc

DEV = "cuda"

def prep(x):
    # chunks are in [-80, 0] dB; map to roughly [-1, 1]
    return ((x + 40.0) / 40.0).unsqueeze(1)

def time_shift(x):
    return torch.roll(x, shifts=int(np.random.randint(0, x.size(-1))), dims=-1)

def spec_augment(x):
    B, _, F, T = x.shape
    fa = torch.arange(F, device=x.device).view(1, 1, F, 1)
    ta = torch.arange(T, device=x.device).view(1, 1, 1, T)
    fill = x.mean(dim=(1, 2, 3), keepdim=True)
    for _ in range(SPEC_MASKS):
        fw = torch.randint(0, FREQ_MASK + 1, (B, 1, 1, 1), device=x.device)
        f0 = (torch.rand(B, 1, 1, 1, device=x.device) * (F - fw)).long()
        tw = torch.randint(0, TIME_MASK + 1, (B, 1, 1, 1), device=x.device)
        t0 = (torch.rand(B, 1, 1, 1, device=x.device) * (T - tw)).long()
        m = ((fa >= f0) & (fa < f0 + fw)) | ((ta >= t0) & (ta < t0 + tw))
        x = torch.where(m, fill.expand_as(x), x)
    return x

def mixup(x, y):
    lam = float(np.random.uniform(*MIXUP_LAM))
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], torch.maximum(y, y[perm])

def make_loader(ds, train, weights=None):
    if train:
        smp = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True) if weights is not None else None
        return DataLoader(ds, batch_size=BATCH, sampler=smp, shuffle=smp is None,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                          persistent_workers=True)
    return DataLoader(ds, batch_size=BATCH * 2, shuffle=False, num_workers=NUM_WORKERS,
                      pin_memory=True, persistent_workers=True)

@torch.no_grad()
def predict(model, loader):
    model.eval(); out = []
    for x, _, _ in loader:
        x = prep(x.to(DEV, non_blocking=True))
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(model(x).float()).cpu())
    return torch.cat(out).numpy()

def fit(model, train_ds, val_ds, Y, weights, epochs=EPOCHS, lr=LR,
        ckpt_dir=None, tag="run", resume=True):
    tl, vl = make_loader(train_ds, True, weights), make_loader(val_ds, False)
    Yv = Y[val_ds.rows]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(tl), pct_start=PCT_START)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.BCEWithLogitsLoss()
    hist, best, start = [], -1.0, 0

    last = f"{ckpt_dir}/{tag}_last.pt" if ckpt_dir else None
    if last and resume and os.path.exists(last):
        s = torch.load(last, map_location=DEV, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"]); scaler.load_state_dict(s["scaler"])
        hist, best, start = s["hist"], s["best"], s["epoch"] + 1
        print(f"resumed from epoch {start}")

    for ep in range(start, epochs):
        model.train(); t0 = time.time(); tot, n = 0.0, 0
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
            tot += loss.item() * len(x); n += len(x)
        auc, k = macro_auc(Yv, predict(model, vl))
        hist.append(dict(epoch=ep, loss=tot / n, val_auc=auc, classes=k, sec=time.time() - t0))
        print(f"ep {ep+1}/{epochs} | loss {tot/n:.4f} | val macro-AUC {auc:.4f} ({k} cls) | {time.time()-t0:.0f}s")
        if ckpt_dir:
            if auc > best:
                best = auc; torch.save(model.state_dict(), f"{ckpt_dir}/{tag}_best.pt")
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            scaler=scaler.state_dict(), hist=hist, best=best, epoch=ep), last)
    return hist
