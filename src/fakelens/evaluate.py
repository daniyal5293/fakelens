"""Shared evaluation and reporting.

* ``finalize_run``: the one function every trainer calls. It fits temperature and threshold on
  VAL only, scores TEST at 0.5 and at the val threshold, adds ECE, bootstrap CIs and per-generator
  and per-group tables, and writes everything to the run directory.
* CLI ``aggregate``: mean +- std over seeds for every model -> results/<exp>/summary.csv + .md
* CLI ``compare``: McNemar test + paired bootstrap between two models' test predictions.

    python -m fakelens.evaluate aggregate --runs /kaggle/working/runs/p1 --out results/p1
    python -m fakelens.evaluate compare --a runs/p1/A2/seed0 --b runs/p1/B/seed0 --out results/p1/A2_vs_B.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fakelens.metrics import (best_threshold, bootstrap_ci, calibration_bins, compute_metrics,
                              expected_calibration_error, fit_temperature, flat_table, group_report,
                              mcnemar_test, sigmoid)
from fakelens.utils import ensure_dir, load_json, save_json


def finalize_run(run_dir: str | Path, model_name: str, val: pd.DataFrame, test: pd.DataFrame,
                 groups: dict, n_boot: int = 1000, extra: dict | None = None) -> dict:
    """``val``/``test`` need columns uid, generator, label, logit (raw, before temperature)."""
    run_dir = ensure_dir(run_dir)
    T = fit_temperature(val.logit.to_numpy(), val.label.to_numpy())
    for df in (val, test):
        df["prob_raw"] = sigmoid(df.logit)
        df["prob"] = sigmoid(df.logit / T)
    thr_val = best_threshold(val.label, val.prob)

    res = {
        "model": model_name, **(extra or {}),
        "temperature": T,
        "threshold_val": thr_val,
        "val": compute_metrics(val.label, val.prob, 0.5),
        "test@0.5": group_report(test, groups, 0.5),
        "test@val_thr": group_report(test, groups, thr_val),
        "calibration": {
            "ece_test_before_T": expected_calibration_error(test.label, test.prob_raw),
            "ece_test_after_T": expected_calibration_error(test.label, test.prob),
            "ece_val_before_T": expected_calibration_error(val.label, val.prob_raw),
            "ece_val_after_T": expected_calibration_error(val.label, val.prob),
        },
        "bootstrap95@0.5": {},
    }
    for name, gens in {**groups, "all": None}.items():
        sub = test if gens is None else test[test.generator.isin(gens)]
        if len(sub) and sub.label.nunique() == 2:
            res["bootstrap95@0.5"][name] = bootstrap_ci(sub.label, sub.prob, 0.5, n_boot=n_boot)

    val.to_csv(run_dir / "val_pred.csv", index=False)
    test.to_csv(run_dir / "test_pred.csv", index=False)
    calibration_bins(test.label, test.prob).to_csv(run_dir / "calibration_test.csv", index=False)
    calibration_bins(test.label, test.prob_raw).to_csv(run_dir / "calibration_test_raw.csv", index=False)
    pd.concat([flat_table(res["test@0.5"], model_name, {"thr": "0.5", **(extra or {})}),
               flat_table(res["test@val_thr"], model_name, {"thr": "val", **(extra or {})})]
              ).to_csv(run_dir / "table.csv", index=False)
    save_json(res, run_dir / "metrics.json")

    g = res["test@0.5"]["groups"]
    msg = " | ".join(f"{k}: acc={v['acc']:.3f} auroc={v['auroc']:.3f}" for k, v in g.items())
    print(f"[{model_name}] {msg}")
    return res


# ---------------------------------------------------------------- aggregation across seeds
def aggregate(runs_dir: str | Path, out_dir: str | Path) -> pd.DataFrame:
    runs_dir, out_dir = Path(runs_dir), ensure_dir(out_dir)
    tables = [pd.read_csv(p) for p in sorted(runs_dir.rglob("table.csv"))]
    if not tables:
        raise FileNotFoundError(f"no table.csv under {runs_dir}")
    df = pd.concat(tables, ignore_index=True)
    keys = [c for c in ("model", "thr", "scope", "name", "heldout", "train_size", "k") if c in df.columns]
    metrics = ["acc", "balanced_acc", "auroc", "ap", "f1", "precision", "recall", "real_acc", "fake_acc"]
    agg = df.groupby(keys, dropna=False)[metrics].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out_dir / "summary.csv", index=False)

    # Compact markdown table: groups at threshold 0.5, mean +- std
    g = agg[(agg.scope == "groups") & (agg.thr.astype(str) == "0.5")]
    lines = ["| model | group | acc | AUROC | F1 | real acc | fake acc | seeds |", "|---|---|---|---|---|---|---|---|"]
    fmt = lambda r, m: f"{r[m + '_mean']:.3f} ± {0 if np.isnan(r[m + '_std']) else r[m + '_std']:.3f}"  # noqa: E731
    for _, r in g.iterrows():
        lines.append(f"| {r['model']} | {r['name']} | {fmt(r, 'acc')} | {fmt(r, 'auroc')} | {fmt(r, 'f1')} | "
                     f"{fmt(r, 'real_acc')} | {fmt(r, 'fake_acc')} | {int(r['acc_count'])} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return agg


# ---------------------------------------------------------------- paired comparison
def compare(run_a: str | Path, run_b: str | Path, groups: dict | None = None, n_boot: int = 1000,
            seed: int = 0) -> dict:
    a = pd.read_csv(Path(run_a) / "test_pred.csv", dtype={"uid": str})
    b = pd.read_csv(Path(run_b) / "test_pred.csv", dtype={"uid": str})
    m = a.merge(b[["uid", "prob"]], on="uid", suffixes=("_a", "_b"))
    groups = groups or {}
    out = {"run_a": str(run_a), "run_b": str(run_b), "n_common": len(m), "scopes": {}}
    rng = np.random.default_rng(seed)
    for name, gens in {**groups, "all": None}.items():
        sub = m if gens is None else m[m.generator.isin(gens)]
        if sub.empty:
            continue
        y = sub.label.to_numpy()
        pa, pb = (sub.prob_a.to_numpy() >= 0.5), (sub.prob_b.to_numpy() >= 0.5)
        diffs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            diffs.append((pa[i] == y[i]).mean() - (pb[i] == y[i]).mean())
        out["scopes"][name] = {
            "acc_a": float((pa == y).mean()), "acc_b": float((pb == y).mean()),
            "acc_diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "mcnemar": mcnemar_test(y, pa, pb),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("aggregate")
    a.add_argument("--runs", required=True)
    a.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args(argv)
    if args.cmd == "aggregate":
        aggregate(args.runs, args.out)
    else:
        from fakelens.utils import load_config

        cfg = load_config(args.config)
        res = compare(args.a, args.b, cfg["data"]["groups"], cfg["experiments"]["bootstrap"])
        save_json(res, args.out)
        for k, v in res["scopes"].items():
            print(f"{k}: acc A={v['acc_a']:.3f} B={v['acc_b']:.3f} diff CI={v['acc_diff_ci95']} "
                  f"McNemar p={v['mcnemar']['exact_p']:.2e}")


if __name__ == "__main__":
    main()
