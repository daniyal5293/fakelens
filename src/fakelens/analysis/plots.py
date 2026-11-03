"""Make every report figure from the results/ and runs/ outputs. Missing inputs are skipped.

    python -m fakelens.analysis.plots --results results --runs /kaggle/working/runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fakelens.analysis import style
from fakelens.analysis.style import plt

GEN_ORDER = ["sdv14", "sdv15", "midjourney", "adm", "glide", "wukong", "vqdm", "biggan"]
GEN_LABEL = {"sdv14": "SD v1.4\n(train)", "sdv15": "SD v1.5", "midjourney": "Midjourney", "adm": "ADM",
             "glide": "GLIDE", "wukong": "Wukong", "vqdm": "VQDM", "biggan": "BigGAN"}


def _models_in(df, col="model"):
    order = list(style.MODEL_COLORS)
    found = list(dict.fromkeys(df[col]))
    return sorted(found, key=lambda m: order.index(m) if m in order else 99)


def _grouped_bars(ax, cats, models, value, err=None, width=0.8):
    n = len(models)
    w = width / max(n, 1)
    x = np.arange(len(cats))
    for i, m in enumerate(models):
        vals = [value(m, c) for c in cats]
        errs = [err(m, c) for c in cats] if err else None
        ax.bar(x - width / 2 + w * (i + 0.5), vals, w * 0.88, yerr=errs, color=style.color(m),
               label=style.label(m), error_kw={"elinewidth": 1, "ecolor": style.TEXT_2, "capsize": 0})
    ax.set_xticks(x)
    return x


# ---------------------------------------------------------------- P1 per-generator accuracy
def plot_p1(summary_csv: Path, out: Path, metric: str = "acc") -> None:
    df = pd.read_csv(summary_csv)
    df = df[(df.scope == "per_generator") & (df.thr.astype(str) == "0.5")]
    if df.empty:
        return
    gens = [g for g in GEN_ORDER if g in set(df.name)]
    models = _models_in(df)
    look = df.set_index(["model", "name"])
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(gens) + 2), 3.8))
    _grouped_bars(ax, gens, models,
                  lambda m, g: look.loc[(m, g), f"{metric}_mean"] if (m, g) in look.index else np.nan,
                  lambda m, g: np.nan_to_num(look.loc[(m, g), f"{metric}_std"]) if (m, g) in look.index else 0)
    ax.set_xticklabels([GEN_LABEL.get(g, g) for g in gens])
    ax.axhline(0.5, color=style.TEXT_2, lw=1, ls=(0, (4, 3)))
    ax.text(len(gens) - 0.5, 0.505, "chance", ha="right", va="bottom", fontsize=8, color=style.TEXT_2)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Accuracy" if metric == "acc" else metric.upper())
    ax.set_title(f"Train on SD v1.4 → test on 8 generators ({'accuracy' if metric == 'acc' else metric.upper()}, mean ± std over seeds)")
    ax.legend(ncol=min(len(models), 3), loc="upper center", bbox_to_anchor=(0.5, -0.18))
    style.save(fig, out)


# ---------------------------------------------------------------- robustness curves
def plot_robustness(csv: Path, out_dir: Path, group: str = "all", thr: str = "0.5") -> None:
    df = pd.read_csv(csv)
    df = df[(df.scope == "groups") & (df.name == group) & (df.thr.astype(str) == thr)]
    if df.empty:
        return
    models = _models_in(df)
    clean = df[df.condition == "clean"].set_index("model")["acc"]
    conds = [c for c in ["jpeg", "resize", "lowres", "blur", "noise"] if c in set(df.condition)]
    xlabels = {"jpeg": "JPEG quality", "resize": "Resize scale", "lowres": "Shorter side (px)",
               "blur": "Gaussian blur σ (px)", "noise": "Gaussian noise std"}
    for metric in ("acc", "auroc"):
        fig, axes = plt.subplots(1, len(conds), figsize=(3.2 * len(conds), 3.2), sharey=True, squeeze=False)
        for ax, c in zip(axes[0], conds):
            sub = df[df.condition == c].copy()
            sub["severity"] = sub.severity.astype(float)
            for m in models:
                s = sub[sub.model == m].sort_values("severity", ascending=(c not in ("jpeg", "lowres")))
                ax.plot(s.severity, s[metric], marker="o", color=style.color(m), label=style.label(m))
            if c in ("jpeg", "lowres"):
                ax.invert_xaxis()
            ax.set_xlabel(xlabels.get(c, c))
            ax.set_ylim(0.4, 1.0)
        axes[0][0].set_ylabel("Accuracy" if metric == "acc" else "AUROC")
        h, l_ = axes[0][0].get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0.08, 1, 0.95))
        fig.legend(h, l_, ncol=min(len(models), 4), loc="lower center", bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"Robustness to image degradations ({group}, {metric}; severity increases to the right)",
                     color=style.TEXT, fontweight="bold")
        style.save(fig, out_dir / f"robustness_{group}_{metric}.png")
    other = df[df.condition.isin(["social"])]
    if len(other) and len(clean):
        rows = [{"model": m, "clean": clean.get(m, np.nan),
                 "social": other[other.model == m].acc.mean()} for m in models]
        pd.DataFrame(rows).to_csv(out_dir / f"robustness_social_{group}.csv", index=False)


# ---------------------------------------------------------------- data efficiency / few-shot
def _mean_std_lines(ax, df, x, y, models):
    for m in models:
        s = df[df.model == m].groupby(x)[y].agg(["mean", "std"]).reset_index()
        ax.plot(s[x], s["mean"], marker="o", color=style.color(m), label=style.label(m))
        if s["std"].notna().any():
            ax.fill_between(s[x], s["mean"] - s["std"].fillna(0), s["mean"] + s["std"].fillna(0),
                            color=style.color(m), alpha=0.15, lw=0)


def plot_data_eff(dir_: Path, out: Path, group: str = "unseen") -> None:
    files = sorted(dir_.glob("data_eff_*.csv"))
    if not files:
        return
    df = pd.concat(pd.read_csv(f) for f in files)
    df = df[df.group == group]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    _mean_std_lines(ax, df, "train_size", "acc", _models_in(df))
    ax.set_xscale("log")
    ax.set_xlabel("Training images (balanced, log scale)")
    ax.set_ylabel(f"Accuracy on {group} generators")
    ax.set_title("Data efficiency")
    ax.legend()
    style.save(fig, out)


def plot_fewshot(dir_: Path, out: Path) -> None:
    files = sorted(dir_.glob("fewshot_*.csv"))
    if not files:
        return
    df = pd.concat(pd.read_csv(f) for f in files)
    df = df.groupby(["model", "seed", "k"], as_index=False)[["acc", "auroc"]].mean()  # mean over unseen gens
    fig, ax = plt.subplots(figsize=(6, 3.8))
    _mean_std_lines(ax, df, "k", "acc", _models_in(df))
    ax.set_xscale("symlog", linthresh=16)
    ax.set_xticks(sorted(df.k.unique()))
    ax.set_xticklabels([str(k) for k in sorted(df.k.unique())])
    ax.set_xlabel("Images of the unseen generator added to training (k)")
    ax.set_ylabel("Accuracy on that generator (mean of 6)")
    ax.set_title("Few-shot adaptation to an unseen generator")
    ax.legend()
    style.save(fig, out)


# ---------------------------------------------------------------- layer analysis
def plot_layers(probe_csv: Path, weights_json: Path | None, out: Path) -> None:
    df = pd.read_csv(probe_csv)
    have_w = weights_json is not None and weights_json.exists()
    fig, axes = plt.subplots(2 if have_w else 1, 1, figsize=(7, 5.6 if have_w else 3.2), sharex=True, squeeze=False)
    ax = axes[0][0]
    for gname, col in (("in_dist", style.PALETTE[0]), ("unseen", style.PALETTE[1])):
        s = df[df.group == gname].sort_values("layer")
        if len(s):
            ax.plot(s.layer, s.acc, marker="o", color=col, label={"in_dist": "SD v1.4 (seen)",
                                                                   "unseen": "6 unseen generators"}[gname])
    ax.set_ylabel("Linear-probe accuracy")
    ax.set_title(f"Which {df.backbone.iloc[0]} block carries the forensic signal?")
    ax.legend()
    if have_w:
        w = json.loads(weights_json.read_text())["layer_weights"]
        ax2 = axes[1][0]
        ax2.bar(np.arange(1, len(w) + 1), w, 0.8, color=style.PALETTE[0])
        ax2.set_ylabel("Learned layer weight (A2)")
        ax2.set_title("Learned softmax layer importance in the multi-layer head")
    axes[-1][0].set_xlabel("Transformer block")
    axes[-1][0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    style.save(fig, out)


# ---------------------------------------------------------------- calibration + confusion
def plot_reliability(run_dirs: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(run_dirs), figsize=(3.6 * len(run_dirs), 3.6), squeeze=False)
    for ax, (m, rd) in zip(axes[0], run_dirs.items()):
        for fname, lab, ls in (("calibration_test_raw.csv", "before T", (0, (4, 3))), ("calibration_test.csv", "after T", "-")):
            p = Path(rd) / fname
            if p.exists():
                b = pd.read_csv(p).dropna()
                ax.plot(b.mean_prob, b.frac_fake, marker="o", ls=ls, color=style.color(m), label=lab)
        ax.plot([0, 1], [0, 1], color=style.TEXT_2, lw=1)
        met = json.loads((Path(rd) / "metrics.json").read_text())["calibration"]
        ax.set_title(f"{style.label(m)}\nECE {met['ece_test_before_T']:.3f} → {met['ece_test_after_T']:.3f}", fontsize=10)
        ax.set_xlabel("Predicted P(fake)")
        ax.legend(loc="upper left")
    axes[0][0].set_ylabel("Observed fraction fake")
    style.save(fig, out)


def plot_confusions(run_dirs: dict, out: Path, scopes=("in_dist", "unseen")) -> None:
    fig, axes = plt.subplots(len(scopes), len(run_dirs), figsize=(2.8 * len(run_dirs), 2.7 * len(scopes)), squeeze=False)
    for j, (m, rd) in enumerate(run_dirs.items()):
        met = json.loads((Path(rd) / "metrics.json").read_text())["test@0.5"]["groups"]
        for i, sc in enumerate(scopes):
            ax = axes[i][j]
            ax.grid(False)
            if sc not in met:
                ax.axis("off")
                continue
            cm = np.array(met[sc]["confusion"])
            norm = cm / cm.sum(1, keepdims=True)
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for (r, c), v in np.ndenumerate(cm):
                ax.text(c, r, f"{v}\n{norm[r, c]:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if norm[r, c] > 0.6 else style.TEXT)
            ax.set_xticks([0, 1], ["real", "fake"])
            ax.set_yticks([0, 1], ["real", "fake"])
            ax.set_title(f"{style.label(m)}\n{sc}", fontsize=9)
            if j == 0:
                ax.set_ylabel("True")
            if i == len(scopes) - 1:
                ax.set_xlabel("Predicted")
    style.save(fig, out)


def plot_logo(dir_: Path, out: Path) -> None:
    files = sorted(dir_.glob("logo_*.csv"))
    if not files:
        return
    df = pd.concat(pd.read_csv(f) for f in files)
    df = df[df.thr == "test@0.5"].groupby(["model", "heldout"], as_index=False)[["acc", "auroc"]].mean()
    gens = [g for g in GEN_ORDER if g in set(df.heldout)]
    models = _models_in(df)
    look = df.set_index(["model", "heldout"])
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(gens) + 2), 3.8))
    _grouped_bars(ax, gens, models, lambda m, g: look.loc[(m, g), "acc"] if (m, g) in look.index else np.nan)
    ax.set_xticklabels([GEN_LABEL.get(g, g).replace("\n(train)", "") for g in gens])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Accuracy on held-out generator")
    ax.set_title("Leave-one-generator-out (train on the other 7)")
    ax.legend(ncol=min(len(models), 3), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    style.save(fig, out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", default="/kaggle/working/runs")
    ap.add_argument("--seed", default="seed0", help="which seed's run dir to use for calibration/confusion plots")
    args = ap.parse_args(argv)
    style.apply()
    R, runs = Path(args.results), Path(args.runs)
    fig = R / "figures"
    if (R / "p1" / "summary.csv").exists():
        plot_p1(R / "p1" / "summary.csv", fig / "p1_per_generator_acc.png", "acc")
        plot_p1(R / "p1" / "summary.csv", fig / "p1_per_generator_auroc.png", "auroc")
    if (R / "robustness" / "robustness.csv").exists():
        for g in ("all", "unseen", "in_dist"):
            plot_robustness(R / "robustness" / "robustness.csv", fig, g)
    plot_data_eff(R / "data_eff", fig / "data_efficiency.png")
    plot_fewshot(R / "fewshot", fig / "fewshot.png")
    plot_logo(R / "logo", fig / "logo.png")
    for probe in sorted((R / "layer_probe").glob("layer_probe_*.csv")):
        bb = probe.stem.replace("layer_probe_", "")
        model = {"clip_vitl14": "A2", "clip_vitb16": "A3", "dinov2_vitl14": "A4"}.get(bb)
        wj = runs / "p1" / model / args.seed / "layer_weights.json" if model else None
        if wj is not None and not wj.exists():
            wj = R / "layer_probe" / f"layer_weights_{model}.json"
        plot_layers(probe, wj, fig / f"layers_{bb}.png")
    run_dirs = {d.parent.name: d for d in sorted((runs / "p1").glob(f"*/{args.seed}")) if (d / "metrics.json").exists()}
    if run_dirs:
        plot_reliability(run_dirs, fig / "reliability.png")
        plot_confusions(run_dirs, fig / "confusion_matrices.png")
    print(f"[plots] figures in {fig}")


if __name__ == "__main__":
    main()
