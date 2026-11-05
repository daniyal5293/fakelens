"""End-to-end smoke test on the toy dataset with the dummy backbone (CPU, no downloads)."""
from pathlib import Path

import pandas as pd
import pytest
import torch

from fakelens.data.manifest import build_manifest


@pytest.fixture(scope="module")
def ready(toy):
    cfg = toy["cfg"]
    build_manifest(cfg, workers=4, verbose=False).to_csv(cfg["paths"]["manifest"], index=False)
    from fakelens.features.extract import extract

    extract(cfg, "dummy", ["train", "val", "test", "logo"], num_workers=0)
    return cfg


def test_feature_cache(ready):
    import numpy as np

    from fakelens.features.extract import feature_dir

    z = np.load(feature_dir(ready, "dummy") / "train.npz")
    n_train = 2 * ready["data"]["n_train_per_class"]
    assert z["cls"].shape == (2 * n_train, 4, 32)  # clean + 1 augmented view
    assert set(z["view"]) == {0, 1}


def test_head_experiments(ready):
    from fakelens.train_head import HeadExperiment

    exp = HeadExperiment(ready, "D0")
    exp.p1([0])
    run = Path(ready["paths"]["runs"]) / "p1" / "D0" / "seed0"
    for f in ("metrics.json", "test_pred.csv", "val_pred.csv", "table.csv", "head.pt", "layer_weights.json"):
        assert (run / f).exists(), f
    t = pd.read_csv(run / "table.csv")
    assert {"in_dist", "unseen", "all"} <= set(t[t.scope == "groups"].name)
    assert len(exp.logo([0])) == 2 * 8
    assert not exp.data_eff([8, 28], [0]).empty
    assert not exp.fewshot([0, 4], [0]).empty
    assert len(exp.layer_probe(0).layer.unique()) == 4
    a8 = exp.all8([0])
    assert set(a8[a8.scope == "per_generator"].name) == set(ready["data"]["generators"])
    ck = torch.load(Path(ready["paths"]["runs"]) / "all8" / "D0" / "seed0" / "head.pt", weights_only=False)
    assert ck["trained_on"] == "all8" and ck["backbone_version"] == "dummy"


def test_resnet_robustness_aggregate_plots(ready, tmp_path):
    import json

    import yaml

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(ready))
    from fakelens import evaluate, robustness, train_resnet
    from fakelens.analysis import plots

    train_resnet.main(["--config", str(cfg_path), "--experiment", "p1", "--seeds", "0", "--num-workers", "0"])
    runs = Path(ready["paths"]["runs"])
    head_ckpt, res_ckpt = runs / "p1" / "D0" / "seed0" / "head.pt", runs / "p1" / "B" / "seed0" / "model.pt"
    assert res_ckpt.exists()

    robustness.main(["--config", str(cfg_path), "--ckpts", str(head_ckpt), str(res_ckpt), "--num-workers", "0"])
    r = pd.read_csv(Path(ready["paths"]["results"]) / "robustness" / "robustness.csv")
    assert set(r.condition) == {"clean", "jpeg", "noise", "lowres"}
    robustness.main(["--config", str(cfg_path), "--ckpts", str(head_ckpt), "--split", "wild", "--conditions", "clean",
                     "--num-workers", "0"])

    agg = evaluate.aggregate(runs / "p1", Path(ready["paths"]["results"]) / "p1")
    assert {"D0", "B"} <= set(agg.model)
    cmp = evaluate.compare(runs / "p1" / "D0" / "seed0", runs / "p1" / "B" / "seed0", ready["data"]["groups"], 50)
    assert cmp["n_common"] == 8 * 2 * ready["data"]["n_test_per_class"]
    json.dumps(cmp)

    plots.main(["--results", ready["paths"]["results"], "--runs", str(runs)])
    figs = {p.name for p in (Path(ready["paths"]["results"]) / "figures").glob("*.png")}
    assert {"p1_per_generator_acc.png", "robustness_all_acc.png", "reliability.png", "confusion_matrices.png",
            "logo.png", "data_efficiency.png", "fewshot.png", "layers_dummy.png"} <= figs


def test_audit_and_errors(ready, tmp_path):
    import yaml

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(ready))
    from fakelens.analysis import audit, errors

    audit.main(["--config", str(cfg_path)])
    res = __import__("json").loads((Path(ready["paths"]["results"]) / "audit" / "audit.json").read_text())
    assert res["shortcut_probe"]["is_not_jpeg_rule_test_acc"] == 1.0   # toy data has the bias on purpose
    assert res["debias_verification"]["jpeg_quality_after_debias"] == [96]
    assert res["debias_verification"]["model_input_sizes"] == [[224, 224]]
    run = Path(ready["paths"]["runs"]) / "p1" / "D0" / "seed0"
    errors.main(["--config", str(cfg_path), "--run", str(run), "--ckpt", str(run / "head.pt"), "--k", "2"])
