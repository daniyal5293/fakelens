import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from fakelens.metrics import (best_threshold, bootstrap_ci, compute_metrics, expected_calibration_error,
                              fit_temperature, group_report, mcnemar_test, sigmoid)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.3 * y + rng.uniform(0, 0.7, 500), 0, 1)
    return y, p


def test_matches_sklearn(data):
    y, p = data
    m = compute_metrics(y, p, 0.5)
    pred = (p >= 0.5).astype(int)
    assert m["acc"] == pytest.approx(accuracy_score(y, pred))
    assert m["precision"] == pytest.approx(precision_score(y, pred))
    assert m["recall"] == pytest.approx(recall_score(y, pred))
    assert m["f1"] == pytest.approx(f1_score(y, pred))
    assert m["auroc"] == pytest.approx(roc_auc_score(y, p))
    (tn, fp), (fn, tp) = m["confusion"]
    assert tn + fp + fn + tp == len(y)


def test_single_class_auroc_nan():
    m = compute_metrics(np.ones(10), np.linspace(0, 1, 10))
    assert np.isnan(m["auroc"]) and m["fake_acc"] == 0.5


def test_best_threshold_improves_balanced_acc(data):
    y, p = data
    t = best_threshold(y, p)
    assert compute_metrics(y, p, t)["balanced_acc"] >= compute_metrics(y, p, 0.5)["balanced_acc"]


def test_bootstrap_contains_point(data):
    y, p = data
    ci = bootstrap_ci(y, p, n_boot=200)
    acc = compute_metrics(y, p)["acc"]
    assert ci["acc"][0] <= acc <= ci["acc"][1]


def test_mcnemar_identical_models():
    y = np.array([0, 1, 1, 0])
    r = mcnemar_test(y, y, y)
    assert r["exact_p"] == 1.0


def test_temperature_reduces_ece():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    logits = (2 * y - 1) * 1.0 + rng.normal(0, 1, 2000)
    over = logits * 5  # over-confident
    T = fit_temperature(over, y)
    assert T > 1
    assert expected_calibration_error(y, sigmoid(over / T)) < expected_calibration_error(y, sigmoid(over))


def test_group_report():
    df = pd.DataFrame({"generator": ["a"] * 4 + ["b"] * 4, "label": [0, 1] * 4, "prob": [0.1, 0.9] * 4})
    r = group_report(df, {"g": ["a"], "missing": ["zzz"]}, 0.5)
    assert r["groups"]["g"]["acc"] == 1.0 and "missing" not in r["groups"]
    assert r["groups"]["all"]["n"] == 8
