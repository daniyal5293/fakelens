"""Dataset bias audit: are JPEG format, JPEG quality and image size enough to tell real from fake?

1. Distribution tables and plots of format, JPEG quality and size per generator and class.
2. Shortcut probe: a classifier that sees ONLY file metadata (is_jpeg, quality, width, height),
   with no pixels. It is trained on the train split and tested on every generator. A high score
   proves the dataset has a non-semantic shortcut that a CNN could learn.
3. Verification that the debias preprocessing removes it: after re-encoding and cropping,
   every model input has the same format, quality and size.

    python -m fakelens.analysis.audit --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier

from fakelens.analysis import style
from fakelens.analysis.style import plt
from fakelens.data.manifest import estimate_jpeg_quality, load_manifest, resolve_path
from fakelens.data.transforms import Preprocess, jpeg_reencode, to_rgb
from fakelens.metrics import compute_metrics
from fakelens.utils import ensure_dir, load_config, save_json

GEN_ORDER = ["sdv14", "sdv15", "midjourney", "adm", "glide", "wukong", "vqdm", "biggan"]


def meta_features(df: pd.DataFrame) -> np.ndarray:
    is_jpeg = (df.format == "JPEG").astype(float)
    q = df.jpeg_quality.clip(lower=0).astype(float)
    w, h = df.width.astype(float), df.height.astype(float)
    return np.c_[is_jpeg, q, w, h, w / h, np.log(w * h)]


def distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.split.isin(["train", "test"])].copy()
    d["cls"] = d.label.map({0: "real", 1: "fake"})
    g = d.groupby(["generator", "cls"])
    return pd.DataFrame({
        "n": g.size(),
        "pct_jpeg": g.apply(lambda x: 100 * (x.format == "JPEG").mean()),
        "median_jpeg_q": g.apply(lambda x: x.jpeg_quality[x.jpeg_quality > 0].median()),
        "pct_png": g.apply(lambda x: 100 * (x.format == "PNG").mean()),
        "median_w": g.width.median(), "median_h": g.height.median(),
        "pct_min_side_lt_224": g.apply(lambda x: 100 * (np.minimum(x.width, x.height) < 224).mean()),
        "n_distinct_sizes": g.apply(lambda x: x[["width", "height"]].drop_duplicates().shape[0]),
    }).round(1).reset_index()


def plot_distributions(tab: pd.DataFrame, df: pd.DataFrame, out: Path) -> None:
    style.apply()
    gens = [g for g in GEN_ORDER if g in set(tab.generator)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    x = np.arange(len(gens))
    for i, cls in enumerate(["real", "fake"]):
        vals = [tab[(tab.generator == g) & (tab.cls == cls)].pct_jpeg.sum() for g in gens]
        ax.bar(x + (i - 0.5) * 0.4, vals, 0.36, color=style.CLASS_COLORS[cls], label=cls)
    ax.set_xticks(x, gens, rotation=30)
    ax.set_ylabel("% of images stored as JPEG")
    ax.set_title("File format differs by class")
    ax.legend()
    ax = axes[1]
    d = df[df.split.isin(["train", "test"])]
    for cls, lab in ((0, "real"), (1, "fake")):
        s = np.minimum(d[d.label == cls].width, d[d.label == cls].height)
        ax.hist(s, bins=np.arange(0, 1100, 25), color=style.CLASS_COLORS[lab], alpha=0.6, label=lab)
    ax.axvline(224, color=style.TEXT_2, lw=1, ls=(0, (4, 3)))
    ax.text(230, ax.get_ylim()[1] * 0.92, "crop 224", fontsize=8, color=style.TEXT_2)
    ax.set_xlabel("Shorter image side (px)")
    ax.set_ylabel("Images")
    ax.set_title("Image size differs by class")
    ax.legend()
    style.save(fig, out)


def shortcut_probe(df: pd.DataFrame) -> dict:
    tr, te = df[df.split == "train"], df[df.split == "test"]
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0,
                                         min_samples_leaf=max(1, min(20, len(tr) // 20))).fit(meta_features(tr), tr.label)
    p = clf.predict_proba(meta_features(te))[:, 1]
    res = {"features": ["is_jpeg", "jpeg_quality", "width", "height", "aspect", "log_area"],
           "per_generator": {}, "all": compute_metrics(te.label, p, 0.5)}
    for g, idx in te.groupby("generator").groups.items():
        res["per_generator"][g] = compute_metrics(te.loc[idx].label, p[te.index.get_indexer(idx)], 0.5)
    # single-feature probe: is_jpeg alone
    pj = (tr.format != "JPEG").astype(int)
    res["is_not_jpeg_rule_train_acc"] = float((pj == tr.label).mean())
    res["is_not_jpeg_rule_test_acc"] = float(((te.format != "JPEG").astype(int) == te.label).mean())
    return res


def verify_debias(df: pd.DataFrame, cfg: dict, n: int = 64) -> dict:
    """Run the real preprocessing on a sample and record what the model actually receives."""
    pre = Preprocess.from_config(cfg, (0, 0, 0), (1, 1, 1), train=False)
    pool = df[df.split.isin(["train", "test"])]
    sample = pd.concat([g.sample(min(len(g), n // 2), random_state=0) for _, g in pool.groupby("label")])
    sizes, quals = set(), set()
    for _, r in sample.iterrows():
        with Image.open(resolve_path(r, cfg["roots"])) as im:
            im.load()
            raw = im.copy()
        sizes.add(pre.pil(raw).size)                        # what the network sees
        if pre.debias_q:                                    # quality after the debias re-encode
            quals.add(estimate_jpeg_quality(jpeg_reencode(to_rgb(raw), pre.debias_q)))
    return {"n_checked": len(sample), "model_input_sizes": sorted(map(list, sizes)),
            "jpeg_quality_after_debias": sorted(quals), "debias_q": pre.debias_q}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--skip-verify", action="store_true", help="skip opening images (metadata-only audit)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    df = load_manifest(cfg["paths"]["manifest"])
    out = ensure_dir(Path(cfg["paths"]["results"]) / "audit")
    tab = distribution_table(df)
    tab.to_csv(out / "distribution.csv", index=False)
    print(tab.to_string(index=False))
    plot_distributions(tab, df, Path(cfg["paths"]["results"]) / "figures" / "audit_format_size.png")
    probe = shortcut_probe(df)
    print(f"\nMetadata-only shortcut probe (no pixels!): test acc={probe['all']['acc']:.3f} "
          f"auroc={probe['all']['auroc']:.3f}; 'not JPEG => fake' rule test acc={probe['is_not_jpeg_rule_test_acc']:.3f}")
    for g, m in probe["per_generator"].items():
        print(f"  {g:>11}: acc={m['acc']:.3f}")
    res = {"shortcut_probe": probe}
    if not args.skip_verify:
        res["debias_verification"] = verify_debias(df, cfg)
        print("debias check:", res["debias_verification"])
    save_json(res, out / "audit.json")


if __name__ == "__main__":
    main()
