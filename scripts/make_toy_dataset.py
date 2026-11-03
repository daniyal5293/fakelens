"""Create a tiny synthetic dataset with the same folder layout as GenImage, for CPU smoke tests.

Fakes carry a faint periodic pattern, a stand-in for generator artifacts that varies by generator.
They are PNG at fixed sizes. Reals are smooth noise saved as JPEG at varied sizes, so the
JPEG/size shortcut exists on purpose. Some real images are copied across generator folders to
exercise SHA-1 dedupe.

    python scripts/make_toy_dataset.py --out toy_data --n 40
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

FOLDERS = {
    "sdv14": "imagenet_ai_0419_sdv4", "sdv15": "imagenet_ai_0424_sdv5", "midjourney": "imagenet_midjourney",
    "adm": "imagenet_ai_0508_adm", "glide": "imagenet_glide", "wukong": "imagenet_ai_0424_wukong",
    "vqdm": "imagenet_ai_0419_vqdm", "biggan": "imagenet_ai_0419_biggan",
}
FAKE_SIZE = {"sdv14": 512, "sdv15": 512, "midjourney": 640, "adm": 256, "glide": 256, "wukong": 512,
             "vqdm": 256, "biggan": 128}


def _real(rng, w, h):
    base = rng.normal(0.5, 0.15, (h // 8 + 1, w // 8 + 1, 3)).clip(0, 1)
    img = Image.fromarray((base * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return img


def _fake(rng, size, period):
    img = np.asarray(_real(rng, size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    img += 6 * np.sin(2 * np.pi * xx / period)[..., None] * np.cos(2 * np.pi * yy / period)[..., None]
    return Image.fromarray(img.clip(0, 255).astype(np.uint8))


def make(out: Path, n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    train_root, test_root = out / "train_root", out / "test_root"
    roots = {"train_root": str(train_root), "test_root": str(test_root), "wild_root": str(out / "wild_root")}

    def write(d: Path, gen: str, k: int, src_split: str):
        (d / "ai").mkdir(parents=True, exist_ok=True)
        (d / "nature").mkdir(parents=True, exist_ok=True)
        period = 3 + list(FOLDERS).index(gen)
        for i in range(k):
            _fake(rng, FAKE_SIZE[gen], period).save(d / "ai" / f"{i % 1000:03d}_{gen}_{i:05d}.png")
            w, h = int(rng.integers(200, 520)), int(rng.integers(200, 520))
            name = f"n0144{i % 10:04d}_{i}.JPEG" if src_split == "train" else f"ILSVRC2012_val_{gen}_{i:08d}.JPEG"
            _real(rng, w, h).save(d / "nature" / name, quality=int(rng.choice([75, 90, 96])))

    write(train_root / "stable_diffusion_v_1_4" / FOLDERS["sdv14"] / "train", "sdv14", n, "train")
    for gen, folder in FOLDERS.items():
        write(test_root / "Validation" / folder / "val", gen, n // 2, "val")
    # cross-folder duplicate real images (must be removed by dedupe)
    src = sorted((test_root / "Validation" / FOLDERS["adm"] / "val" / "nature").iterdir())[:3]
    for p in src:
        shutil.copy(p, test_root / "Validation" / FOLDERS["glide"] / "val" / "nature" / ("dup_" + p.name))
    # wild set
    for cls in ("real", "fake/flux"):
        (out / "wild_root" / cls).mkdir(parents=True, exist_ok=True)
    for i in range(4):
        _real(rng, 300, 300).save(out / "wild_root" / "real" / f"r{i}.jpg", quality=90)
        _fake(rng, 300, 9).save(out / "wild_root" / "fake" / "flux" / f"f{i}.png")
    return roots


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="toy_data")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    print(make(Path(a.out), a.n))
