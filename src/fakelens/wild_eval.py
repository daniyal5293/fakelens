"""In-the-wild evaluation on a local folder (runs on CPU in minutes; no manifest or Kaggle needed).

Folder layout (the same one ``manifest.scan_wild`` reads):
    <wild>/real/<subset>/*.jpg        e.g. real/camera, real/whatsapp
    <wild>/fake/<generator>/*.png     e.g. fake/midjourney-v7

    python -m fakelens.wild_eval --wild data/wild \
        --ckpts checkpoints/A2_all8_head.pt checkpoints/A2_head.pt checkpoints/B_model.pt \
        --names A2-all8 A2 B --conditions clean social

Each image goes through every frozen backbone once; all heads that share that backbone and preprocessing
reuse the features. Fakes are scored at threshold 0.5 on the calibrated probability (as in the demo), and
the AUROC per generator uses that generator's fakes against *all* reals of the same condition.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from fakelens.data.manifest import scan_wild
from fakelens.data.transforms import make_perturbation
from fakelens.detector import Detector
from fakelens.utils import ensure_dir, get_device

THR = 0.5
SUBSET_LABELS = {  # short two-line tick labels for the figure
    "camera": "camera\noriginal", "whatsapp": "WhatsApp\ncopy", "flux2-klein": "FLUX.2\nklein",
    "gpt-image-1.5": "GPT-image\n1.5", "illustrious-sdxl": "Illustrious\n(SDXL)", "midjourney-v7": "Midjourney\nv7",
    "seedream-v5": "Seedream\n5", "z-image-turbo": "Z-Image\nTurbo",
}


def load_wild(root: str | Path) -> pd.DataFrame:
    df = scan_wild(root)
    if df.empty:
        raise SystemExit(f"No images under {root}/real or {root}/fake")
    parts = df.relpath.str.split("/")
    df["cls"] = np.where(df.label == 1, "fake", "real")
    # subset = second path level: generator for fakes, capture route (camera / whatsapp) for reals
    df["subset"] = [p[1] if len(p) > 2 else c for p, c in zip(parts, df.cls)]
    return df.reset_index(drop=True)


@torch.no_grad()
def score(df: pd.DataFrame, root: Path, dets: dict[str, Detector], condition: str,
          batch_size: int = 16) -> pd.DataFrame:
    """P(fake) for every image under one condition, sharing backbone passes between heads."""
    pert = None if condition == "clean" else make_perturbation(condition)[0]
    groups: dict[tuple, list[str]] = {}
    for name, d in dets.items():
        key = (d.kind, d.backbone_name, d.preprocess_key) if d.kind == "head" else ("resnet", name)
        groups.setdefault(key, []).append(name)
    probs = {name: np.zeros(len(df)) for name in dets}
    for i in range(0, len(df), batch_size):
        imgs = []
        for rel in df.relpath.iloc[i:i + batch_size]:
            im = Image.open(root / rel).convert("RGB")
            imgs.append(pert(im, None) if pert else im)
        for names in groups.values():
            d0 = dets[names[0]]
            x = torch.stack([d0.preprocess(im) for im in imgs]).to(d0.device)
            if d0.kind == "head":
                feats = d0.backbone(x)
                for n in names:
                    probs[n][i:i + len(imgs)] = torch.sigmoid(dets[n].logits_from_feats(feats)).cpu().numpy()
            else:
                probs[names[0]][i:i + len(imgs)] = torch.sigmoid(d0.logits(x)).cpu().numpy()
        print(f"[wild] {condition}: {min(i + batch_size, len(df))}/{len(df)}", flush=True)
    out = df[["relpath", "cls", "subset", "label"]].copy()
    out["condition"] = condition
    for n, p in probs.items():
        out[f"p_{n}"] = p
    return out


def summarize(pred: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    rows = []
    for (cond,), g in pred.groupby(["condition"]):
        reals = g[g.label == 0]
        for n in names:
            p = f"p_{n}"
            y, s = g.label.to_numpy(), g[p].to_numpy()
            rows.append({"condition": cond, "model": n, "subset": "ALL", "cls": "all", "n": len(g),
                         "acc": float(((s >= THR) == y).mean()),
                         "auroc": float(roc_auc_score(y, s)) if len(set(y)) == 2 else np.nan})
            for (cls, sub), h in g.groupby(["cls", "subset"]):
                correct = (h[p] >= THR) if cls == "fake" else (h[p] < THR)
                auc = np.nan
                if cls == "fake" and len(reals):
                    yy = np.r_[np.zeros(len(reals)), np.ones(len(h))]
                    auc = float(roc_auc_score(yy, np.r_[reals[p], h[p]]))
                rows.append({"condition": cond, "model": n, "subset": sub, "cls": cls, "n": len(h),
                             "acc": float(correct.mean()), "auroc": auc,
                             "median_p_fake": float(h[p].median())})
    return pd.DataFrame(rows)


def to_markdown(summ: pd.DataFrame, names: list[str]) -> str:
    lines = []
    for cond, g in summ.groupby("condition", sort=False):
        lines += [f"### Condition: {cond}", "",
                  "Accuracy at threshold 0.5 (reals: share called *real*; fakes: share called *AI-generated*). "
                  "AUROC per generator = its fakes vs all reals.", "",
                  "| Subset | n | " + " | ".join(f"{n} acc" for n in names) + " | "
                  + " | ".join(f"{n} AUROC" for n in names) + " |",
                  "|---|---|" + "---|" * (2 * len(names))]
        for (cls, sub), h in g.groupby(["cls", "subset"], sort=False):
            h = h.set_index("model")
            label = "**All images**" if cls == "all" else f"{cls}: {sub}"
            accs = " | ".join(f"{h.loc[n, 'acc']:.2f}" for n in names)
            aucs = " | ".join("–" if pd.isna(h.loc[n, "auroc"]) else f"{h.loc[n, 'auroc']:.2f}" for n in names)
            lines.append(f"| {label} | {int(h['n'].iloc[0])} | {accs} | {aucs} |")
        lines.append("")
    return "\n".join(lines)


def plot(summ: pd.DataFrame, names: list[str], out: Path) -> None:
    import matplotlib.pyplot as plt

    from fakelens.analysis import style

    style.apply()
    conds = list(dict.fromkeys(summ.condition))
    fig, axes = plt.subplots(len(conds), 1, figsize=(10, 3.4 * len(conds)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, cond in zip(axes, conds):
        g = summ[(summ.condition == cond) & (summ.cls != "all")]
        subsets = list(dict.fromkeys(g.sort_values(["cls", "subset"], ascending=[False, True]).subset))
        xs = np.arange(len(subsets))
        w = 0.8 / len(names)
        for k, n in enumerate(names):
            h = g[g.model == n].set_index("subset").reindex(subsets)
            ax.bar(xs + (k - (len(names) - 1) / 2) * w, h.acc, width=w * 0.92, color=style.color(n),
                   label=style.label(n))
        ax.set_xticks(xs, [SUBSET_LABELS.get(s, s) for s in subsets], rotation=0, fontsize=9)
        ax.set_ylabel("accuracy at threshold 0.5")
        ax.set_ylim(0, 1.12)
        ax.set_title({"clean": "Original files", "social": "After a social-media upload (long side 1080 px, JPEG 80)"}
                     .get(cond, cond) + ": share classified correctly", loc="left")
        n_real = (g.cls == "real").groupby(g.subset).any().sum()
        ax.axvline(n_real - 0.5, color=style.TEXT_2, lw=0.8, ls=":")
        for x, txt in (((n_real - 1) / 2, "real photos"), ((n_real + len(subsets) - 1) / 2, "AI-generated")):
            ax.text(x, 1.07, txt, ha="center", va="center", fontsize=9, color=style.TEXT_2)
    axes[-1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=len(names))
    fig.tight_layout()
    style.save(fig, out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wild", default="data/wild")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+", default=["clean"], help="clean and/or social (long side 1080 + JPEG 80)")
    ap.add_argument("--out", default="results/wild")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args(argv)
    if len(args.ckpts) != len(args.names):
        raise SystemExit("--ckpts and --names must have the same length")
    root, out = Path(args.wild), ensure_dir(args.out)
    device = get_device()
    dets = {n: Detector(c, device, name=n) for n, c in zip(args.names, args.ckpts)}
    df = load_wild(root)
    print(df.groupby(["cls", "subset"]).size().to_string())
    pred = pd.concat([score(df, root, dets, c, args.batch_size) for c in args.conditions], ignore_index=True)
    pred.to_csv(out / "wild_predictions.csv", index=False)
    summ = summarize(pred, args.names)
    summ.to_csv(out / "wild_summary.csv", index=False)
    md = to_markdown(summ, args.names)
    (out / "wild_summary.md").write_text(md, encoding="utf-8")
    plot(summ, args.names, out / "wild_accuracy.png")
    print(md)


if __name__ == "__main__":
    main()
