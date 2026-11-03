"""Metrics, thresholds, calibration and statistical tests. Fake = positive class (label 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def compute_metrics(y, prob, thr: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    pred = (prob >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    real_acc, fake_acc = div(tn, tn + fp), div(tp, tp + fn)
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "threshold": float(thr),
        "acc": div(tp + tn, len(y)),
        "balanced_acc": np.nanmean([real_acc, fake_acc]),
        "precision": precision,
        "recall": recall,
        "f1": div(2 * precision * recall, precision + recall) if tp else 0.0,
        "real_acc": real_acc,
        "fake_acc": fake_acc,
        "auroc": float(roc_auc_score(y, prob)) if both else float("nan"),
        "ap": float(average_precision_score(y, prob)) if both else float("nan"),
        "confusion": [[tn, fp], [fn, tp]],  # rows = true (real, fake), cols = predicted
    }


def best_threshold(y, prob) -> float:
    """Threshold maximising balanced accuracy (= Youden's J). Chosen on validation data only."""
    fpr, tpr, thr = roc_curve(np.asarray(y).astype(int), prob)
    j = tpr - fpr
    t = thr[int(np.argmax(j))]
    return float(min(t, 1.0)) if np.isfinite(t) else 0.5


def bootstrap_ci(y, prob, thr: float = 0.5, keys=("acc", "auroc", "f1", "balanced_acc"),
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    y, prob = np.asarray(y), np.asarray(prob)
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        m = compute_metrics(y[idx], prob[idx], thr)
        for k in keys:
            vals[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2)))]
            for k, v in vals.items()}


def mcnemar_test(y, pred_a, pred_b) -> dict:
    """Paired test: do models A and B make *different* errors on the same images?"""
    y, a, b = (np.asarray(v).astype(int) for v in (y, pred_a, pred_b))
    ca, cb = a == y, b == y
    n01 = int((ca & ~cb).sum())  # A right, B wrong
    n10 = int((~ca & cb).sum())  # A wrong, B right
    n = n01 + n10
    exact_p = float(binomtest(min(n01, n10), n, 0.5).pvalue) if n else 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / n if n else 0.0
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "exact_p": exact_p,
            "chi2_cc": float(stat), "chi2_p": float(chi2.sf(stat, 1)) if n else 1.0}


def calibration_bins(y, prob, n_bins: int = 15) -> pd.DataFrame:
    y, prob = np.asarray(y, dtype=float), np.asarray(prob, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "count": int(m.sum()),
                     "mean_prob": float(prob[m].mean()) if m.any() else np.nan,
                     "frac_fake": float(y[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def expected_calibration_error(y, prob, n_bins: int = 15) -> float:
    b = calibration_bins(y, prob, n_bins)
    b = b[b["count"] > 0]
    return float((b["count"] * (b["mean_prob"] - b["frac_fake"]).abs()).sum() / b["count"].sum())


def fit_temperature(logits, y, max_iter: int = 200) -> float:
    """Temperature scaling (Guo et al., 2017) on validation logits; returns T > 0."""
    import torch

    z = torch.tensor(np.array(logits, dtype=np.float64))
    t = torch.tensor(np.array(y, dtype=np.float64))
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z / log_t.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(np.clip(log_t.exp().item(), 0.05, 100.0))


def group_report(pred: pd.DataFrame, groups: dict, thr: float) -> dict:
    """Metrics per generator, per group (pooled) and the GenImage-style mean of per-generator accuracy.

    ``pred`` needs columns: generator, label, prob.
    """
    out = {"per_generator": {}, "groups": {}}
    for gen, g in pred.groupby("generator", sort=False):
        out["per_generator"][gen] = compute_metrics(g.label, g.prob, thr)
    present = set(out["per_generator"])
    for name, gens in {**groups, "all": sorted(present)}.items():
        gens = [g for g in gens if g in present]
        if not gens:
            continue
        sub = pred[pred.generator.isin(gens)]
        m = compute_metrics(sub.label, sub.prob, thr)
        m["mean_acc_over_generators"] = float(np.mean([out["per_generator"][g]["acc"] for g in gens]))
        aucs = [out["per_generator"][g]["auroc"] for g in gens]
        m["mean_auroc_over_generators"] = float(np.nanmean(aucs)) if not np.all(np.isnan(aucs)) else float("nan")
        m["generators"] = gens
        out["groups"][name] = m
    return out


def flat_table(report: dict, model: str, extra: dict | None = None) -> pd.DataFrame:
    """Flatten a group_report into rows (one per generator / group) for CSV tables."""
    rows = []
    for kind in ("per_generator", "groups"):
        for name, m in report[kind].items():
            r = {"model": model, "scope": kind, "name": name, **(extra or {})}
            r.update({k: v for k, v in m.items() if k not in ("confusion", "generators")})
            tn, fp = m["confusion"][0]
            fn, tp = m["confusion"][1]
            r.update(tn=tn, fp=fp, fn=fn, tp=tp)
            rows.append(r)
    return pd.DataFrame(rows)
