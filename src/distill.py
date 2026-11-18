import os, gc, time, numpy as np, torch, torch.nn as nn
import torch.nn.functional as Fn
from engine import DEV, prep, time_shift, spec_augment, mixup, make_loader, predict
from train_config import *
from metrics import macro_auc

def kd_loss(zs, zt, y, T, alpha):
    hard = Fn.binary_cross_entropy_with_logits(zs, y)
    if alpha >= 1.0:
        return hard
    soft = Fn.binary_cross_entropy_with_logits(zs / T, torch.sigmoid(zt / T))
    return alpha * hard + (1 - alpha) * (T * T) * soft

def fit_kd(student, teacher, train_ds, val_ds, Y, weights, T=2.0, alpha=0.5,
           epochs=EPOCHS, lr=LR, ckpt_dir=None, tag="kd", resume=True):
    use_kd = alpha < 1.0
    if use_kd: teacher.eval()
    tl, vl = make_loader(train_ds, True, weights), make_loader(val_ds, False)
    Yv = Y[val_ds.rows]
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(tl), pct_start=PCT_START)
    scaler = torch.amp.GradScaler("cuda")
    hist, best, start = [], -1.0, 0
    last = f"{ckpt_dir}/{tag}_last.pt" if ckpt_dir else None
    if last and resume and os.path.exists(last):
        s = torch.load(last, map_location=DEV, weights_only=False)
        student.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"]); scaler.load_state_dict(s["scaler"])
        hist, best, start = s["hist"], s["best"], s["epoch"] + 1
        print(f"resumed from epoch {start}")
    for ep in range(start, epochs):
        student.train(); t0 = time.time(); tot, n = 0.0, 0
        for x, y, _ in tl:
            x = prep(x.to(DEV, non_blocking=True)); y = y.to(DEV, non_blocking=True)
            x = spec_augment(time_shift(x))
            if np.random.rand() < MIXUP_P:
                x, y = mixup(x, y)
            zt = None
            if use_kd:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    zt = teacher(x).float()
            with torch.autocast("cuda", dtype=torch.float16):
                zs = student(x)
            loss = kd_loss(zs.float(), zt, y, T, alpha)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(x); n += len(x)
        auc, k = macro_auc(Yv, predict(student, vl))
        hist.append(dict(epoch=ep, loss=tot / n, val_auc=auc, classes=k, sec=time.time() - t0))
        print(f"[{tag}] ep {ep+1}/{epochs} | loss {tot/n:.4f} | val macro-AUC {auc:.4f} | {time.time()-t0:.0f}s")
        if ckpt_dir:
            if auc > best:
                best = auc; torch.save(student.state_dict(), f"{ckpt_dir}/{tag}_best.pt")
            torch.save(dict(model=student.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            scaler=scaler.state_dict(), hist=hist, best=best, epoch=ep), last)
    del tl, vl; gc.collect()
    return hist
