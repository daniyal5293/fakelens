"""t-SNE of frozen features on the test split: do real and fake separate, and per generator?

Small multiples: one panel per generator shows that generator's real and fake points on top of
all points in grey. This avoids putting 8 hues in one scatter plot, where they cannot all be told apart.

    python -m fakelens.analysis.embeddings --model A2 [--ckpt runs/p1/A2/seed0/head.pt]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import TSNE

from fakelens.analysis import style
from fakelens.analysis.style import plt
from fakelens.data.manifest import load_manifest
from fakelens.features.extract import feature_dir, load_split
from fakelens.train_head import HEAD_MODELS
from fakelens.utils import load_config

GEN_ORDER = ["sdv14", "sdv15", "midjourney", "adm", "glide", "wukong", "vqdm", "biggan"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="A2", choices=list(HEAD_MODELS))
    ap.add_argument("--ckpt", default=None, help="if given, embed the head's pooled multi-layer feature")
    ap.add_argument("--n-per-gen", type=int, default=300)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. roots.train_root=/path")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    style.apply()
    spec = HEAD_MODELS[args.model]
    man = load_manifest(cfg["paths"]["manifest"])
    data = load_split(feature_dir(cfg, spec["backbone"], args.raw), "test", man, "clean")
    df = data["df"]
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.permutation(np.where(df.generator == g)[0])[:args.n_per_gen]
                          for g in df.generator.unique()])
    if args.ckpt:
        from fakelens.detector import Detector

        det = Detector(args.ckpt, device=torch.device("cpu"))
        with torch.no_grad():
            X = det.head.pooled(torch.from_numpy(data["cls"][idx]).float()).numpy() if hasattr(det.head, "pooled") \
                else data["embed"][idx].astype(np.float32)
        what = "learned multi-layer feature"
    else:
        X = data["embed"][idx].astype(np.float32)
        what = "final embedding"
    Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X)
    sub = df.iloc[idx].reset_index(drop=True)
    gens = [g for g in GEN_ORDER if g in set(sub.generator)]
    cols = 4
    rows = int(np.ceil(len(gens) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)
    for ax, g in zip(axes.flat, gens):
        ax.scatter(Z[:, 0], Z[:, 1], s=4, color="#d6d5d0", lw=0)
        for lab, name in ((0, "real"), (1, "fake")):
            m = (sub.generator == g) & (sub.label == lab)
            ax.scatter(Z[m, 0], Z[m, 1], s=9, color=style.CLASS_COLORS[name], label=name, lw=0)
        ax.set_title(g, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    for ax in list(axes.flat)[len(gens):]:
        ax.axis("off")
    axes[0][0].legend(loc="lower left", markerscale=2)
    fig.suptitle(f"t-SNE of {spec['backbone']} {what} (test split)", color=style.TEXT, fontweight="bold")
    out = Path(cfg["paths"]["results"]) / "figures" / f"tsne_{args.model}{'_raw' if args.raw else ''}.png"
    style.save(fig, out)
    print(f"[tsne] {out}")


if __name__ == "__main__":
    main()
