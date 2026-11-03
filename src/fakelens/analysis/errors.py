"""Error analysis: the most confident false positives and false negatives, plus explanation maps.

    python -m fakelens.analysis.errors --run /kaggle/working/runs/p1/A2/seed0 --ckpt .../head.pt --k 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from fakelens.analysis import style
from fakelens.analysis.explain import gradcam_resnet, occlusion_map, overlay
from fakelens.analysis.style import plt
from fakelens.data.manifest import load_manifest, resolve_path
from fakelens.data.transforms import denormalize
from fakelens.utils import ensure_dir, load_config


def confident_errors(pred: pd.DataFrame, man: pd.DataFrame, k: int, thr: float = 0.5) -> pd.DataFrame:
    df = pred.merge(man[["uid", "split", "root_key", "relpath", "format", "width", "height",
                         "jpeg_quality", "imagenet_class"]], on="uid")
    fp = df[(df.label == 0) & (df.prob >= thr)].sort_values("prob", ascending=False)
    fn = df[(df.label == 1) & (df.prob < thr)].sort_values("prob")
    parts = []
    for gen, g in fp.groupby("generator"):
        parts.append(g.head(max(1, k // 4)).assign(error="false_positive"))
    for gen, g in fn.groupby("generator"):
        parts.append(g.head(max(1, k // 4)).assign(error="false_negative"))
    return pd.concat(parts) if parts else df.iloc[:0]


def error_rates_by(pred: pd.DataFrame, man: pd.DataFrame, col: str, thr: float = 0.5) -> pd.DataFrame:
    df = pred if col in pred.columns else pred.merge(man[["uid", col]], on="uid")
    df = df.copy()
    df["wrong"] = (df.prob >= thr).astype(int) != df.label
    return (df.groupby(["label", col]).wrong.agg(["mean", "size"]).rename(columns={"mean": "error_rate"})
              .reset_index().sort_values("error_rate", ascending=False))


def gallery(errs: pd.DataFrame, roots: dict, out: Path, detector=None, k: int = 8, title: str = "") -> None:
    style.apply()
    for kind in ("false_positive", "false_negative"):
        sub = errs[errs.error == kind].sort_values("prob", ascending=(kind == "false_negative")).head(k)
        if sub.empty:
            continue
        rows = 2 if detector is not None else 1
        fig, axes = plt.subplots(rows, len(sub), figsize=(2.3 * len(sub), 2.6 * rows), squeeze=False)
        for j, (_, r) in enumerate(sub.iterrows()):
            with Image.open(resolve_path(r, roots)) as im:
                im.load()
                img = im.copy()
            if detector is not None:
                x = detector.preprocess(img).unsqueeze(0)
                rgb = denormalize(x[0], detector.preprocess.mean, detector.preprocess.std)
                if detector.kind == "resnet":
                    heat, sym = gradcam_resnet(detector.net, x.to(detector.device)), False
                else:
                    heat, sym = occlusion_map(detector.prob_fn(), x.to(detector.device)), True
                axes[1][j].imshow(overlay(rgb, heat, symmetric=sym))
                axes[1][j].axis("off")
            else:
                rgb = np.asarray(detector.preprocess.pil(img) if detector else img.convert("RGB"))
            axes[0][j].imshow(rgb)
            axes[0][j].set_title(f"{r.generator} p(fake)={r.prob:.2f}\n{r.format} {r.width}×{r.height}",
                                 fontsize=8, color=style.TEXT)
            axes[0][j].axis("off")
        nice = "Real images predicted FAKE" if kind == "false_positive" else "Fake images predicted REAL"
        fig.suptitle(f"{title}: {nice} (most confident)", color=style.TEXT, fontweight="bold")
        style.save(fig, out / f"{kind}.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", required=True, help="run dir containing test_pred.csv")
    ap.add_argument("--ckpt", default=None, help="checkpoint for heatmaps (optional)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. roots.train_root=/path")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    man = load_manifest(cfg["paths"]["manifest"])
    pred = pd.read_csv(Path(args.run) / "test_pred.csv", dtype={"uid": str})
    model = Path(args.run).parent.name
    out = ensure_dir(args.out or Path(cfg["paths"]["results"]) / "errors" / model)
    errs = confident_errors(pred, man, k=args.k * 4)
    errs.to_csv(out / "confident_errors.csv", index=False)
    error_rates_by(pred, man, "imagenet_class").head(30).to_csv(out / "error_rate_by_class.csv", index=False)
    error_rates_by(pred, man, "generator").to_csv(out / "error_rate_by_generator.csv", index=False)
    det = None
    if args.ckpt:
        from fakelens.detector import Detector

        det = Detector(args.ckpt)
    gallery(errs, cfg["roots"], out, det, args.k, title=style.label(model))
    print(f"[errors] wrote {out}")


if __name__ == "__main__":
    main()
