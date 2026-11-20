"""Figures for the written report and the slides, drawn from the committed result tables.

    python scripts/make_report_figures.py        # writes report/figures/*.png

Uses the project's visual system (fakelens.analysis.style), so each model keeps its colour.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fakelens.analysis import style

R = Path("results")
OUT = Path("report/figures")
GEN_LABELS = {"sdv14": "SD v1.4\n(train)", "sdv15": "SD v1.5", "midjourney": "Midjourney", "adm": "ADM",
              "glide": "GLIDE", "wukong": "Wukong", "vqdm": "VQDM", "biggan": "BigGAN"}
UNSEEN = ["wukong", "midjourney", "vqdm", "glide", "adm", "biggan"]


def setup():
    style.apply()
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "legend.fontsize": 11, "axes.titlesize": 13})


def fig_unseen_models():
    s = pd.read_csv(R / "p1/summary.csv")
    s = s[(s.scope == "groups") & (s.name == "unseen") & (s.thr.astype(str) == "0.5")].set_index("model")
    models = ["A2", "A4", "A3", "A1", "B"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, (col, title) in zip(axes, [("auroc", "AUROC (ranking, threshold-free)"),
                                       ("acc", "Accuracy at threshold 0.5")]):
        y = np.arange(len(models))[::-1]
        vals = [s.loc[m, f"{col}_mean"] for m in models]
        errs = [s.loc[m, f"{col}_std"] for m in models]
        ax.barh(y, vals, xerr=errs, height=0.62, color=[style.color(m) for m in models],
                error_kw={"ecolor": style.TEXT_2, "lw": 1})
        for yi, v, e in zip(y, vals, errs):
            ax.text(v + e + 0.012, yi, f"{v:.3f}", va="center", fontsize=11, color=style.TEXT)
        ax.set_xlim(0.5, 1.04)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(models))[::-1], [style.label(m) for m in models])
    fig.suptitle("Unseen generators (6), trained on SD v1.4 only, mean of 3 seeds", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=style.TEXT)
    fig.tight_layout()
    style.save(fig, OUT / "fig_unseen_models.png")


def fig_threshold_gap():
    s = pd.read_csv(R / "p1/summary.csv")
    s = s[(s.scope == "per_generator") & (s.model == "A2") & (s.thr.astype(str) == "0.5")].set_index("name")
    gens = ["sdv14", "sdv15"] + UNSEEN
    x = np.arange(len(gens))
    auc, acc = s.loc[gens, "auroc_mean"].to_numpy(), s.loc[gens, "acc_mean"].to_numpy()
    c = style.color("A2")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.vlines(x, acc, auc, color=style.GRID, lw=6, zorder=1)
    ax.scatter(x, auc, s=110, color=c, zorder=3, label="AUROC: ranks fakes above reals")
    ax.scatter(x, acc, s=110, facecolor=style.SURFACE, edgecolor=c, lw=2.2, zorder=3,
               label="Accuracy at threshold 0.5")
    for xi, a, b in zip(x, auc, acc):
        if a - b < 0.03:  # marks overlap: one label for both
            ax.text(xi + 0.14, a, f"{a:.2f} / {b:.2f}", va="center", fontsize=10, color=style.TEXT)
            continue
        ax.text(xi + 0.14, a, f"{a:.2f}", va="center", fontsize=10, color=style.TEXT)
        ax.text(xi + 0.14, b, f"{b:.2f}", va="center", fontsize=10, color=style.TEXT_2)
    ax.axvline(1.5, color=style.TEXT_2, lw=0.8, ls=":")
    ax.text(0.5, 0.45, "trained on / near", ha="center", fontsize=10, color=style.TEXT_2)
    ax.text(4.5, 0.45, "unseen generators", ha="center", fontsize=10, color=style.TEXT_2)
    ax.set_xticks(x, [GEN_LABELS[g] for g in gens])
    ax.set_ylim(0.4, 1.04)
    ax.set_ylabel("score")
    ax.set_title("A2 per generator: the features separate, the threshold does not transfer", loc="left")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    style.save(fig, OUT / "fig_threshold_gap.png")


def fig_jpeg():
    r = pd.read_csv(R / "robustness/robustness.csv")
    g = r[(r.scope == "groups") & (r.name == "unseen") & (r.thr == "val")]
    qs = [None, 95, 85, 75, 65, 50, 30]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for m in ["A2", "A4", "A1", "B", "B_noaug"]:
        h = g[g.model == m]
        ys = [h[h.condition == "clean"].auroc.iloc[0]] + \
             [h[(h.condition == "jpeg") & (h.severity == q)].auroc.iloc[0] for q in qs[1:]]
        ax.plot(range(len(qs)), ys, marker="o", color=style.color(m), label=style.label(m),
                ls="--" if m == "B_noaug" else "-")
    ax.set_xticks(range(len(qs)), ["clean"] + [str(q) for q in qs[1:]])
    ax.set_xlabel("JPEG quality applied to test images")
    ax.set_ylabel("unseen-generator AUROC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Robustness to JPEG compression (6 unseen generators)", loc="left")
    ax.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    style.save(fig, OUT / "fig_jpeg.png")


def fig_layers():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    names = {"clip_vitl14": ("CLIP ViT-L/14 (24 blocks)", "A2"), "clip_vitb16": ("CLIP ViT-B/16 (12 blocks)", "A3"),
             "dinov2_vitl14": ("DINOv2 ViT-L/14 (24 blocks)", "A4")}
    for b, (lab, m) in names.items():
        d = pd.read_csv(R / f"layer_probe/layer_probe_{b}.csv")
        d = d[d.group == "unseen"].sort_values("layer")
        depth = (d.layer - 1) / (d.layer.max() - 1)
        i = d.auroc.idxmax()
        ax.plot(depth, d.auroc, marker="o", ms=4, color=style.color(m),
                label=f"{lab}: peak at block {int(d.layer[i])} ({d.auroc[i]:.3f})")
        ax.scatter([depth[i]], [d.auroc[i]], s=160, marker="*", color=style.color(m), edgecolor=style.SURFACE,
                   zorder=4)
    ax.set_xlabel("relative depth (0 = first block, 1 = last block)")
    ax.set_ylabel("unseen-generator AUROC")
    ax.set_ylim(0.55, 1.0)
    ax.set_title("Linear probe on each block: the signal peaks in the middle", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    style.save(fig, OUT / "fig_layers.png")


def fig_fewshot():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ks = [0, 16, 64, 256]
    for m, ls in (("A2", "-"), ("A1", "-")):
        d = pd.read_csv(R / f"fewshot/fewshot_{m}.csv")
        y = d.groupby("k").acc.mean().reindex(ks)
        ax.plot(range(4), y, marker="o", color=style.color(m), ls=ls, label=f"{style.label(m)}, mean of 6 generators")
        off = -0.035 if m == "A2" else 0.02  # A2 labels below its line, A1 above
        for xi, v in enumerate(y):
            ax.text(xi + 0.06, v + off, f"{v:.2f}", ha="left", fontsize=10, color=style.TEXT)
    d = pd.read_csv(R / "fewshot/fewshot_A2.csv")
    y = d[d.generator == "biggan"].groupby("k").acc.mean().reindex(ks)
    ax.plot(range(4), y, marker="s", color=style.color("A2"), ls="--", label="A2 on BigGAN (hardest)")
    ax.set_xticks(range(4), [f"k = {k}" for k in ks])
    ax.set_xlabel("images of the unseen generator added to training")
    ax.set_ylabel("accuracy on that generator")
    ax.set_ylim(0.45, 1.03)
    ax.set_title("Few-shot adaptation: 64 examples fix most of the gap", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    style.save(fig, OUT / "fig_fewshot.png")


def fig_logo():
    rows = []
    for m in ["A2", "A1", "B"]:
        d = pd.read_csv(R / f"logo/logo_{m}.csv")
        d = d[d.thr == "test@0.5"].groupby("heldout").acc.mean()
        rows.append(d.rename(m))
    t = pd.concat(rows, axis=1)
    gens = ["sdv14", "sdv15", "midjourney", "adm", "glide", "wukong", "vqdm", "biggan"]
    x = np.arange(len(gens))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for k, m in enumerate(["A2", "A1", "B"]):
        ax.bar(x + (k - 1) * w, t.loc[gens, m], width=w * 0.92, color=style.color(m),
               label=f"{style.label(m)} (mean {t[m].mean():.3f})")
    ax.set_xticks(x, [GEN_LABELS[g].replace("\n(train)", "") for g in gens])
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("accuracy on the held-out generator")
    ax.set_title("Leave one generator out: train on 7, test on the 8th", loc="left")
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, -0.32))
    fig.tight_layout()
    style.save(fig, OUT / "fig_logo.png")


def fig_compression_test():
    d = pd.read_csv(R / "wild/camera_compression_test.csv")
    labels = ["original\n(JPEG ≈96, ≈4000 px)", "only re-compressed\n(JPEG 69)", "only shrunk\n(1600 px)",
              "both\n(WhatsApp-like)"]
    x = np.arange(len(d))
    w = 0.27
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for k, m in enumerate(["A2-all8", "A2", "B"]):
        bars = ax.bar(x + (k - 1) * w, d[m], width=w * 0.92, color=style.color(m),
                      label={"A2-all8": "A2-all8 (demo model)", "A2": "A2 (SD v1.4 only)", "B": "B ResNet-18"}[m])
        for b, v in zip(bars, d[m]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.0%}", ha="center", fontsize=10, color=style.TEXT)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel('share of REAL photos called "fake"')
    ax.set_title("Same 35 real camera photos, one property changed at a time", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    style.save(fig, OUT / "fig_compression_test.png")


def main():
    setup()
    for f in (fig_unseen_models, fig_threshold_gap, fig_jpeg, fig_layers, fig_fewshot, fig_logo,
              fig_compression_test):
        f()
        print("wrote", f.__name__)


if __name__ == "__main__":
    main()
