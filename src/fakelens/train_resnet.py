"""Model B: fine-tune ResNet-18 on the same manifests, preprocessing and evaluation as the CLIP heads.

    python -m fakelens.train_resnet --experiment p1                        # main baseline, all seeds
    python -m fakelens.train_resnet --experiment p1 --raw --seeds 0         # bias ablation (no debias re-encode)
    python -m fakelens.train_resnet --experiment p1 --no-aug --seeds 0      # augmentation ablation
    python -m fakelens.train_resnet --experiment logo --seeds 0 --set resnet.epochs=8
    python -m fakelens.train_resnet --experiment data_eff --seeds 0
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fakelens.data.dataset import ManifestDataset
from fakelens.data.manifest import load_manifest
from fakelens.data.transforms import IMAGENET_MEAN, IMAGENET_STD, Preprocess
from fakelens.evaluate import finalize_run
from fakelens.metrics import group_report, sigmoid
from fakelens.models.resnet import ResNet18Detector
from fakelens.utils import ensure_dir, get_device, load_config, save_json, seed_everything


def _pre(cfg, train: bool, raw: bool, aug: bool) -> Preprocess:
    p = Preprocess.from_config(cfg, IMAGENET_MEAN, IMAGENET_STD, train=train, debias_q=None if raw else "cfg")
    if train and not aug:
        p.aug = {"hflip": 0.5}  # keep random crop + flip only
    return p


@torch.no_grad()
def predict(model, df: pd.DataFrame, cfg: dict, device, raw: bool = False, perturb=None,
            num_workers: int | None = None) -> np.ndarray:
    rc = cfg["resnet"]
    ds = ManifestDataset(df, cfg["roots"], _pre(cfg, False, raw, False), perturb=perturb, seed=cfg["seed"])
    dl = DataLoader(ds, batch_size=rc["batch_size"] * 2, shuffle=False,
                    num_workers=rc["num_workers"] if num_workers is None else num_workers,
                    pin_memory=device.type == "cuda")
    model.eval()
    out = np.zeros(len(ds), dtype=np.float32)
    for x, _, idx in tqdm(dl, leave=False, desc="predict"):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out[idx.numpy()] = model.logits(x.to(device, non_blocking=True)).float().cpu().numpy()
    return out


def train_resnet(cfg: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, device, seed: int,
                 raw: bool = False, aug: bool = True, num_workers: int | None = None):
    seed_everything(seed)
    rc = cfg["resnet"]
    nw = rc["num_workers"] if num_workers is None else num_workers
    ds = ManifestDataset(train_df, cfg["roots"], _pre(cfg, True, raw, aug), seed=seed)
    gen = torch.Generator().manual_seed(seed)
    model = ResNet18Detector(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=rc["lr"], weight_decay=rc["weight_decay"])
    steps = math.ceil(len(ds) / rc["batch_size"])
    total, warm = rc["epochs"] * steps, rc.get("warmup_epochs", 1) * steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best, best_state, bad, hist = -1.0, None, 0, []
    yva = val_df.label.to_numpy()
    for ep in range(rc["epochs"]):
        ds.set_epoch(ep)
        dl = DataLoader(ds, batch_size=rc["batch_size"], shuffle=True, num_workers=nw, generator=gen,
                        pin_memory=device.type == "cuda", drop_last=len(ds) > rc["batch_size"])
        model.train()
        tot, n = 0.0, 0
        for x, y, _ in tqdm(dl, leave=False, desc=f"epoch {ep}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = F.binary_cross_entropy_with_logits(model.logits(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, n = tot + loss.item() * len(y), n + len(y)
        va = predict(model, val_df, cfg, device, raw, num_workers=nw)
        auc = roc_auc_score(yva, va) if len(np.unique(yva)) == 2 else 0.0
        hist.append({"epoch": ep, "train_loss": tot / max(n, 1), "val_auroc": float(auc)})
        print(f"[resnet] epoch {ep}: loss={tot / max(n, 1):.4f} val_auroc={auc:.4f}")
        if auc > best + 1e-5:
            best, best_state, bad = auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= rc["patience"]:
                break
    model.load_state_dict(best_state)
    return model.eval(), hist


def _frame(df, logits):
    return pd.DataFrame({"uid": df.uid.to_numpy(), "generator": df.generator.to_numpy(),
                         "label": df.label.to_numpy(), "logit": logits})


def save_checkpoint(model, path, cfg, raw, T, thr, extra=None):
    pre = _pre(cfg, False, raw, False)
    torch.save({"kind": "resnet", "model": "B", "preprocess": pre.to_dict(), "state_dict": model.state_dict(),
                "temperature": T, "threshold": thr, **(extra or {})}, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--experiment", default="p1", choices=["p1", "logo", "data_eff"])
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--heldout", nargs="*", default=None, help="LOGO: generators to hold out (default all)")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    device = get_device()
    man = load_manifest(cfg["paths"]["manifest"])
    groups = cfg["data"]["groups"]
    tag = "B" + ("_raw" if args.raw else "") + ("_noaug" if args.no_aug else "")
    runs, results = Path(cfg["paths"]["runs"]), Path(cfg["paths"]["results"])
    tr, va, te = (man[man.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    kw = dict(raw=args.raw, num_workers=args.num_workers)

    if args.experiment == "p1":
        for seed in seeds:
            model, hist = train_resnet(cfg, tr, va, device, seed, aug=not args.no_aug, **kw)
            run = ensure_dir(runs / "p1" / tag / f"seed{seed}")
            val, test = _frame(va, predict(model, va, cfg, device, **kw)), _frame(te, predict(model, te, cfg, device, **kw))
            res = finalize_run(run, tag, val, test, groups, cfg["experiments"]["bootstrap"],
                               extra={"seed": seed, "experiment": "p1"})
            model.temperature.fill_(res["temperature"])
            save_checkpoint(model, run / "model.pt", cfg, args.raw, res["temperature"], res["threshold_val"])
            save_json(hist, run / "history.json")

    elif args.experiment == "logo":
        pool = man[man.split == "logo"].reset_index(drop=True)
        held_list = args.heldout or [g for g in cfg["data"]["generators"] if g in set(pool.generator)]
        rows = []
        for seed in seeds[:1]:
            for held in held_list:
                rest = pool[pool.generator != held].sample(frac=1.0, random_state=seed).reset_index(drop=True)
                n_va = max(2, int(0.1 * len(rest)))
                model, _ = train_resnet(cfg, rest.iloc[n_va:], rest.iloc[:n_va], device, seed,
                                        aug=not args.no_aug, **kw)
                te_h = te[te.generator == held].reset_index(drop=True)
                val = _frame(rest.iloc[:n_va], predict(model, rest.iloc[:n_va], cfg, device, **kw))
                test = _frame(te_h, predict(model, te_h, cfg, device, **kw))
                res = finalize_run(runs / "logo" / tag / f"heldout_{held}" / f"seed{seed}", tag, val, test,
                                   {"heldout": [held]}, n_boot=200,
                                   extra={"seed": seed, "heldout": held, "experiment": "logo"})
                for thr_key in ("test@0.5", "test@val_thr"):
                    m = res[thr_key]["per_generator"][held]
                    rows.append({"model": tag, "seed": seed, "heldout": held, "thr": thr_key,
                                 **{k: m[k] for k in ("acc", "auroc", "ap", "real_acc", "fake_acc", "f1")}})
                pd.DataFrame(rows).to_csv(ensure_dir(results / "logo") / f"logo_{tag}.csv", index=False)

    elif args.experiment == "data_eff":
        rows = []
        for seed in seeds:
            for size in cfg["experiments"]["data_efficiency_sizes"]:
                half = min(size // 2, (tr.label == 0).sum(), (tr.label == 1).sum())
                sub = pd.concat([tr[tr.label == c].sample(half, random_state=seed) for c in (0, 1)])
                model, _ = train_resnet(cfg, sub, va, device, seed, aug=not args.no_aug, **kw)
                test = _frame(te, predict(model, te, cfg, device, **kw))
                test["prob"] = sigmoid(test.logit)
                for gname, m in group_report(test, groups, 0.5)["groups"].items():
                    rows.append({"model": tag, "seed": seed, "train_size": 2 * half, "group": gname,
                                 "acc": m["acc"], "auroc": m["auroc"], "mean_acc": m["mean_acc_over_generators"]})
                pd.DataFrame(rows).to_csv(ensure_dir(results / "data_eff") / f"data_eff_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
