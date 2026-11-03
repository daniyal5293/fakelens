import numpy as np
import pytest
import torch
from PIL import Image

from fakelens.data.manifest import estimate_jpeg_quality
from fakelens.data.transforms import (PERTURBATIONS, Preprocess, jpeg_reencode, make_perturbation, to_rgb)


def _img(w=300, h=260, seed=0, mode="RGB"):
    arr = np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr).convert(mode)


def test_eval_preprocess_shape_and_determinism():
    p = Preprocess(debias_q=96)
    a, b = p(_img()), p(_img())
    assert a.shape == (3, 224, 224)
    assert torch.equal(a, b)


def test_small_images_are_upscaled():
    p = Preprocess(debias_q=None)
    assert p.pil(_img(128, 128)).size == (224, 224)


def test_debias_makes_png_and_jpeg_inputs_same_quality():
    png_like = _img(seed=1)
    jpg = jpeg_reencode(_img(seed=2), 75)
    q1 = estimate_jpeg_quality(jpeg_reencode(to_rgb(png_like), 96))
    q2 = estimate_jpeg_quality(jpeg_reencode(to_rgb(jpg), 96))
    assert q1 == q2 == 96


def test_train_aug_seeded():
    p = Preprocess(train=True, aug={"hflip": 0.5, "blur_prob": 0.5, "blur_sigma": [0.1, 3], "jpeg_prob": 0.5,
                                    "jpeg_quality": [30, 100]})
    a = p(_img(), np.random.default_rng(5))
    b = p(_img(), np.random.default_rng(5))
    c = p(_img(), np.random.default_rng(6))
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_rgba_handled():
    assert Preprocess()(_img(mode="RGBA")).shape == (3, 224, 224)


@pytest.mark.parametrize("name,sev", [("clean", None), ("jpeg", 50), ("resize", 0.5), ("lowres", 64),
                                      ("blur", 3.0), ("noise", 0.05), ("social", None)])
def test_perturbations_run_and_are_deterministic(name, sev):
    f, desc = make_perturbation(name, sev)
    a = np.asarray(f(_img(), np.random.default_rng(1)))
    b = np.asarray(f(_img(), np.random.default_rng(1)))
    assert np.array_equal(a, b) and isinstance(desc, str)
    if name == "lowres":
        assert min(a.shape[:2]) == 64
    assert name in PERTURBATIONS
