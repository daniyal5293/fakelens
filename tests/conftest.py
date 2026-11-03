import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def toy(tmp_path_factory):
    """Toy GenImage-layout dataset + a config pointing at it (small sizes, CPU friendly)."""
    from make_toy_dataset import make

    from fakelens.utils import load_config

    base = tmp_path_factory.mktemp("toy")
    roots = make(base / "data", n=40)
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg["roots"] = roots
    cfg["paths"] = {"manifest": str(base / "manifest.csv"), "features": str(base / "features"),
                    "runs": str(base / "runs"), "results": str(base / "results")}
    cfg["data"].update(n_train_per_class=14, n_val_per_class=4, n_test_per_class=6, n_logo_per_class=8,
                       oversample=1.2)
    cfg["features"].update(batch_size=8, num_workers=0, train_aug_views=1)
    cfg["heads"]["linear"]["epochs"] = 5
    cfg["heads"]["multilayer"]["epochs"] = 5
    cfg["heads"]["multilayer"]["hidden"] = 16
    cfg["heads"].update(batch_size=16, patience=3)
    cfg["resnet"].update(epochs=1, batch_size=8, num_workers=0)
    cfg["experiments"].update(data_efficiency_sizes=[8, 28], fewshot_k=[0, 4], bootstrap=20)
    cfg["robustness"]["n_per_class"] = 3
    cfg["robustness"]["conditions"] = {"clean": [None], "jpeg": [50], "noise": [0.03], "lowres": [64]}
    cfg["seeds"] = [0]
    return {"cfg": cfg, "base": base}
