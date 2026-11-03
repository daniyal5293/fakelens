import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fakelens.data.manifest import build_manifest, estimate_jpeg_quality, infer_generator, load_manifest


@pytest.mark.parametrize("path,gen", [
    ("stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/ai", "sdv14"),
    ("Validation/imagenet_ai_0424_sdv5/val/nature", "sdv15"),
    ("Validation/imagenet_ai_0508_adm/val/ai", "adm"),
    ("Validation/imagenet_glide/val/ai", "glide"),
    ("Validation/imagenet_midjourney/val/ai", "midjourney"),
    ("Validation/imagenet_ai_0419_biggan/val/ai", "biggan"),
    ("Validation/imagenet_ai_0424_wukong/val/ai", "wukong"),
    ("Validation/imagenet_ai_0419_vqdm/val/ai", "vqdm"),
    ("some/random/folder", None),
])
def test_infer_generator(path, gen):
    assert infer_generator(path) == gen


@pytest.mark.parametrize("q", [30, 50, 75, 90, 96, 100])
def test_jpeg_quality_estimate(q):
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(buf, "JPEG", quality=q)
    buf.seek(0)
    est = estimate_jpeg_quality(Image.open(buf))
    assert abs(est - q) <= 1


@pytest.fixture(scope="module")
def manifest(toy):
    df = build_manifest(toy["cfg"], workers=4, verbose=False)
    df.to_csv(toy["cfg"]["paths"]["manifest"], index=False)
    return df


def test_splits_disjoint_by_content(manifest):
    assert manifest.sha1.is_unique, "duplicate image content across rows"
    assert manifest.uid.is_unique
    by_split = {s: set(g.sha1) for s, g in manifest.groupby("split")}
    splits = list(by_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            assert not (by_split[a] & by_split[b]), f"leak between {a} and {b}"


def test_split_sizes_and_balance(manifest, toy):
    d = toy["cfg"]["data"]
    tr = manifest[manifest.split == "train"]
    assert (tr.generator == "sdv14").all()
    assert tr.label.value_counts().to_dict() == {0: d["n_train_per_class"], 1: d["n_train_per_class"]}
    te = manifest[manifest.split == "test"]
    assert set(te.generator) == set(d["generators"])
    counts = te.groupby(["generator", "label"]).size()
    assert (counts == d["n_test_per_class"]).all()


def test_train_from_train_folder_test_from_val_folder(manifest):
    assert set(manifest[manifest.split.isin(["train", "val"])].source_split) == {"train"}
    assert set(manifest[manifest.split.isin(["test", "logo"])].source_split) == {"val"}


def test_wild_split_and_roundtrip(manifest, toy):
    w = manifest[manifest.split == "wild"]
    assert len(w) == 8 and set(w.label) == {0, 1}
    re = load_manifest(toy["cfg"]["paths"]["manifest"])
    pd.testing.assert_series_equal(re.uid, manifest.uid, check_names=False)


def test_metadata_reflects_bias(manifest):
    te = manifest[manifest.split == "test"]
    assert (te[te.label == 0].format == "JPEG").all()
    assert (te[te.label == 1].format == "PNG").all()
