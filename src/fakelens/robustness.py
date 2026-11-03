"""Robustness evaluation: run every checkpoint end-to-end on perturbed test images.

All models see exactly the same perturbed image (the same seed for each image). Models that share
preprocessing share the tensor, and heads that share a backbone share one backbone forward pass.

    python -m fakelens.robustness --ckpts runs/p1/A1/seed0/head.pt runs/p1/A2/seed0/head.pt runs/p1/B/seed0/model.pt
    python -m fakelens.robustness --ckpts ... --split wild --conditions clean      # in-the-wild set
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from fakelens.data.manifest import load_manifest, resolve_path
from fakelens.data.transforms import make_perturbation
from fakelens.detector import Detector
from fakelens.metrics import flat_table, group_report, sigmoid
from fakelens.utils import ensure_dir, get_device, load_config, save_json


class MultiPreDataset(Dataset):
    def __init__(self, df, roots, preprocesses: list, perturb=None, seed: int = 0):
        self.df, self.roots, self.pres, self.perturb, self.seed = df.reset_index(drop=True), roots, preprocesses, perturb, seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        from PIL import Image

        with Image.open(resolve_path(self.df.iloc[i], self.roots)) as im:
            im.load()
            img = im.copy()
        if self.perturb is not None:
            img = self.perturb(img, np.random.default_rng([self.seed, i, 777]))
        return tuple(p(img, np.random.default_rng([self.seed, i])) for p in self.pres), i


def robustness_subset(man: pd.DataFrame, split: str, n_per_class: int | None, seed: int) -> pd.DataFrame:
    df = man[man.split == split]
    if n_per_class and split == "test":
        df = pd.concat([g.sample(min(len(g), n_per_class), random_state=seed)
                        for _, g in df.sort_values("uid").groupby(["generator", "label"], sort=True)])
    return df.reset_index(drop=True)


@torch.no_grad()
def evaluate_condition(dets: list[Detector], df, roots, perturb, batch_size, num_workers, seed) -> dict:
    keys = list(dict.fromkeys(d.preprocess_key for d in dets))
    pres = [next(d.preprocess for d in dets if d.preprocess_key == k) for k in keys]
    ds = MultiPreDataset(df, roots, pres, perturb, seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logits = {d.name: np.zeros(len(ds), dtype=np.float32) for d in dets}
    for xs, idx in tqdm(dl, leave=False):
        idx = idx.numpy()
        feat_cache = {}
        for d in dets:
            x = xs[keys.index(d.preprocess_key)]
            if d.kind == "head":
                ck = (d.backbone_name, d.preprocess_key)
                if ck not in feat_cache:
                    with torch.autocast(device_type=d.device.type, dtype=torch.float16, enabled=d.device.type == "cuda"):
                        feat_cache[ck] = d.backbone(x.to(d.device))
                logits[d.name][idx] = d.logits_from_feats(feat_cache[ck]).cpu().numpy()
            else:
                logits[d.name][idx] = d.logits(x).cpu().numpy()
    return logits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--names", nargs="*", default=None, help="display names (default: from checkpoint)")
    ap.add_argument("--split", default="test", choices=["test", "wild", "val"])
    ap.add_argument("--conditions", nargs="*", default=None, help="subset of config robustness.conditions")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    device = get_device()
    names = args.names or [None] * len(args.ckpts)
    dets = [Detector(p, device, n) for p, n in zip(args.ckpts, names)]
    if len({d.name for d in dets}) != len(dets):  # disambiguate duplicate names
        for i, d in enumerate(dets):
            d.name = f"{d.name}#{i}"
    man = load_manifest(cfg["paths"]["manifest"])
    rcfg = cfg["robustness"]
    df = robustness_subset(man, args.split, rcfg["n_per_class"] if args.split == "test" else None, cfg["seed"])
    groups = cfg["data"]["groups"] if args.split == "test" else {}
    conds = {k: v for k, v in rcfg["conditions"].items() if args.conditions is None or k in args.conditions}
    out_dir = ensure_dir(args.out or Path(cfg["paths"]["results"]) / ("robustness" if args.split == "test" else args.split))
    print(f"[robustness] {len(df)} images, {sum(len(v) for v in conds.values())} conditions, models={[d.name for d in dets]}")

    tables, log = [], {"split": args.split, "n_images": len(df), "conditions": {}, "models": {}}
    for d in dets:
        log["models"][d.name] = {"ckpt": d.ckpt_path, "threshold_val": d.threshold, "preprocess": d.preprocess.to_dict()}
    for cond, sevs in conds.items():
        for sev in sevs:
            perturb, desc = make_perturbation(cond, sev)
            log["conditions"][f"{cond}:{sev}"] = desc
            logits = evaluate_condition(dets, df, cfg["roots"], perturb, args.batch_size, args.num_workers, cfg["seed"])
            summary = []
            for d in dets:
                pred = pd.DataFrame({"uid": df.uid, "generator": df.generator, "label": df.label,
                                     "prob": sigmoid(logits[d.name])})
                for thr_name, thr in (("0.5", 0.5), ("val", d.threshold)):
                    rep = group_report(pred, groups, thr)
                    tables.append(flat_table(rep, d.name, {"condition": cond, "severity": sev, "thr": thr_name}))
                    if thr_name == "0.5":
                        a = rep["groups"]["all"]
                        summary.append(f"{d.name} acc={a['acc']:.3f} auroc={a['auroc']:.3f}")
            print(f"  {cond}:{sev} -> " + " | ".join(summary))
    res = pd.concat(tables, ignore_index=True)
    res.to_csv(out_dir / "robustness.csv", index=False)
    save_json(log, out_dir / "robustness_config.json")
    print(f"[robustness] wrote {out_dir / 'robustness.csv'}")


if __name__ == "__main__":
    main()
