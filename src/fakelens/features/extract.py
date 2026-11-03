"""Extract and cache frozen-backbone features for each manifest split.

The backbone runs only ONCE per image. After that every head experiment (seeds, LOGO, data
efficiency, few-shot, layer probes) trains in seconds from these cached features.

For the train split, view 0 is the clean (eval-style) crop. Views 1..k are augmented
(random crop / flip / blur / JPEG). This gives the CLIP heads the same augmentation the ResNet sees.

Usage (Kaggle GPU):
    python -m fakelens.features.extract --backbone clip_vitl14 --splits train val test logo
    python -m fakelens.features.extract --backbone clip_vitl14 --splits train val test --raw   # bias ablation
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fakelens.data.dataset import ManifestDataset
from fakelens.data.manifest import load_manifest
from fakelens.data.transforms import Preprocess
from fakelens.features.backbones import BACKBONES, Backbone, build_backbone
from fakelens.utils import ensure_dir, get_device, load_config, load_json, save_json


def feature_dir(cfg: dict, backbone: str, raw: bool = False) -> Path:
    return Path(cfg["paths"]["features"]) / (backbone + ("_raw" if raw else ""))


@torch.no_grad()
def run_backbone(backbone: Backbone, ds: ManifestDataset, batch_size: int, num_workers: int,
                 device, desc: str = "") -> dict:
    n = len(ds)
    cls = np.zeros((n, backbone.n_layers, backbone.dim), dtype=np.float16)
    emb = np.zeros((n, backbone.embed_dim), dtype=np.float16)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=device.type == "cuda", persistent_workers=False)
    use_amp = device.type == "cuda"
    for x, _, idx in tqdm(dl, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = backbone(x)
        idx = idx.numpy()
        cls[idx] = out["cls"].float().cpu().numpy().astype(np.float16)
        emb[idx] = out["embed"].float().cpu().numpy().astype(np.float16)
    return {"cls": cls, "embed": emb}


def extract(cfg: dict, backbone_name: str, splits: list[str], raw: bool = False,
            device=None, num_workers: int | None = None, overwrite: bool = False) -> Path:
    device = device or get_device()
    df_all = load_manifest(cfg["paths"]["manifest"])
    out_dir = ensure_dir(feature_dir(cfg, backbone_name, raw))
    fcfg = cfg["features"]
    nw = fcfg["num_workers"] if num_workers is None else num_workers
    backbone = build_backbone(backbone_name).to(device).eval()
    debias = None if raw else "cfg"
    pre_eval = Preprocess.from_config(cfg, backbone.mean, backbone.std, train=False, debias_q=debias)
    pre_train = Preprocess.from_config(cfg, backbone.mean, backbone.std, train=True, debias_q=debias)

    for split in splits:
        out_path = out_dir / f"{split}.npz"
        if out_path.exists() and not overwrite:
            print(f"[extract] {out_path} exists, skipping (use --overwrite)")
            continue
        df = df_all[df_all.split == split].reset_index(drop=True)
        if df.empty:
            print(f"[extract] split {split!r} empty, skipping")
            continue
        n_views = 1 + (fcfg.get("train_aug_views", 0) if split == "train" else 0)
        chunks = {"cls": [], "embed": [], "uid": [], "view": []}
        t0 = time.time()
        for v in range(n_views):
            ds = ManifestDataset(df, cfg["roots"], pre_eval if v == 0 else pre_train, seed=cfg["seed"], view=v)
            out = run_backbone(backbone, ds, fcfg["batch_size"], nw, device, desc=f"{split} view{v}")
            chunks["cls"].append(out["cls"])
            chunks["embed"].append(out["embed"])
            chunks["uid"].append(df.uid.to_numpy().astype(str))
            chunks["view"].append(np.full(len(df), v, dtype=np.int16))
        np.savez(out_path, **{k: np.concatenate(v) for k, v in chunks.items()})
        print(f"[extract] {split}: {len(df)} images x {n_views} view(s) -> {out_path} ({time.time() - t0:.0f}s)")

    save_json({"backbone": backbone_name, "n_layers": backbone.n_layers, "dim": backbone.dim,
               "embed_dim": backbone.embed_dim, "preprocess": pre_eval.to_dict(), "raw": raw,
               "train_aug_views": fcfg.get("train_aug_views", 0), "augment": cfg.get("augment", {})},
              out_dir / "meta.json")
    return out_dir


def load_split(feat_dir: str | Path, split: str, manifest: pd.DataFrame, views: str = "all") -> dict:
    """Load cached features for a split, joined with manifest columns (label, generator, ...).

    views: "all" | "clean" (view 0 only).
    """
    z = np.load(Path(feat_dir) / f"{split}.npz")
    keep = np.ones(len(z["uid"]), dtype=bool) if views == "all" else (z["view"] == 0)
    uid = z["uid"][keep]
    meta = manifest.set_index("uid").loc[uid].reset_index()
    return {"cls": z["cls"][keep], "embed": z["embed"][keep], "view": z["view"][keep], "df": meta,
            "y": meta["label"].to_numpy().astype(np.float32)}


def load_meta(feat_dir: str | Path) -> dict:
    return load_json(Path(feat_dir) / "meta.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--backbone", required=True, choices=BACKBONES)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test", "logo"])
    ap.add_argument("--raw", action="store_true", help="skip the debias JPEG re-encode (bias ablation)")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    extract(cfg, args.backbone, args.splits, raw=args.raw, num_workers=args.num_workers, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
