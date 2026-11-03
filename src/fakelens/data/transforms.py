"""Image preprocessing shared by *every* model, plus test-time perturbations.

Pipeline (train and eval are identical apart from the random parts):

    RGB -> [perturbation (robustness eval only)] -> debias JPEG re-encode (both classes)
        -> [train aug: blur / JPEG] -> upscale if smaller than crop -> crop -> [hflip] -> tensor -> normalize

Why debias? In GenImage the real images are ImageNet JPEGs of varying size, but most fakes are PNG
at fixed sizes. A detector can reach very high in-distribution accuracy just by spotting
"JPEG vs not JPEG" (Grommelt et al., "Fake or JPEG?", ECCV-W 2024). We re-encode *every* image
at the same quality. This can be done at test time without knowing the label. Then we crop
without resizing, which removes most of the size signal while keeping high-frequency generator traces.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageFilter

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- primitives
def to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def jpeg_reencode(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out


def ensure_min_side(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if min(w, h) >= size:
        return img
    s = size / min(w, h)
    return img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)


def center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def random_crop(img: Image.Image, size: int, rng: np.random.Generator) -> Image.Image:
    w, h = img.size
    left = int(rng.integers(0, w - size + 1))
    top = int(rng.integers(0, h - size + 1))
    return img.crop((left, top, left + size, top + size))


def to_tensor(img: Image.Image, mean, std) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    m = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (t - m) / s


def denormalize(t: torch.Tensor, mean, std) -> np.ndarray:
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return ((t.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------- main preprocessing
@dataclass
class Preprocess:
    mean: tuple = CLIP_MEAN
    std: tuple = CLIP_STD
    crop: int = 224
    debias_q: int | None = 96
    min_side_resize: bool = True
    train: bool = False
    aug: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict, mean, std, train: bool = False, debias_q="cfg") -> "Preprocess":
        p = cfg["preprocess"]
        return cls(mean=tuple(mean), std=tuple(std), crop=p["crop"],
                   debias_q=p["debias_jpeg_quality"] if debias_q == "cfg" else debias_q,
                   min_side_resize=p.get("min_side_resize", True), train=train,
                   aug=cfg.get("augment", {}) if train else {})

    def to_dict(self) -> dict:
        return {"mean": list(self.mean), "std": list(self.std), "crop": self.crop,
                "debias_q": self.debias_q, "min_side_resize": self.min_side_resize}

    def pil(self, img: Image.Image, rng: np.random.Generator | None = None) -> Image.Image:
        """All PIL-level steps; returns the crop the network will see (useful for visualisation)."""
        rng = rng if rng is not None else np.random.default_rng()
        img = to_rgb(img)
        if self.debias_q:
            img = jpeg_reencode(img, self.debias_q)
        if self.train and self.aug:
            a = self.aug
            if rng.random() < a.get("blur_prob", 0.0):
                lo, hi = a.get("blur_sigma", [0.1, 3.0])
                img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(lo, hi))))
            if rng.random() < a.get("jpeg_prob", 0.0):
                lo, hi = a.get("jpeg_quality", [30, 100])
                img = jpeg_reencode(img, int(rng.integers(lo, hi + 1)))
        if self.min_side_resize:
            img = ensure_min_side(img, self.crop)
        img = random_crop(img, self.crop, rng) if self.train else center_crop(img, self.crop)
        if self.train and rng.random() < self.aug.get("hflip", 0.0):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def __call__(self, img: Image.Image, rng: np.random.Generator | None = None) -> torch.Tensor:
        return to_tensor(self.pil(img, rng), self.mean, self.std)


# ---------------------------------------------------------------- robustness perturbations
Perturb = Callable[[Image.Image, np.random.Generator], Image.Image]


def _p_jpeg(q):
    return lambda img, rng: jpeg_reencode(to_rgb(img), int(q))


def _p_resize(scale):
    def f(img, rng):
        w, h = img.size
        return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    return f


def _p_lowres(size):
    def f(img, rng):
        w, h = img.size
        s = size / min(w, h)
        return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    return f


def _p_blur(sigma):
    return lambda img, rng: to_rgb(img).filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def _p_noise(std):
    def f(img, rng):
        arr = np.asarray(to_rgb(img), dtype=np.float32) / 255.0
        arr = np.clip(arr + rng.normal(0.0, std, arr.shape), 0, 1)
        return Image.fromarray((arr * 255 + 0.5).astype(np.uint8))
    return f


def _p_social(_=None, long_side=1080, quality=80):
    def f(img, rng):
        img = to_rgb(img)
        w, h = img.size
        if max(w, h) > long_side:
            s = long_side / max(w, h)
            img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
        return jpeg_reencode(img, quality)
    return f


PERTURBATIONS: dict[str, tuple[Callable, str]] = {
    "clean": (lambda _=None: (lambda img, rng: img), "no change"),
    "jpeg": (_p_jpeg, "JPEG re-encode at quality={s} (PIL, 4:2:0 subsampling)"),
    "resize": (_p_resize, "bicubic resize by scale={s} (both sides)"),
    "lowres": (_p_lowres, "bicubic resize so the shorter side = {s}px (GenImage Table 5)"),
    "blur": (_p_blur, "Gaussian blur, sigma={s}px (PIL GaussianBlur radius)"),
    "noise": (_p_noise, "additive Gaussian noise, std={s} on [0,1] pixels, seeded per image"),
    "social": (_p_social, "downscale long side to 1080px (if larger) then JPEG quality 80"),
}


def make_perturbation(name: str, severity=None) -> tuple[Perturb, str]:
    if name not in PERTURBATIONS:
        raise KeyError(f"unknown perturbation {name!r}; choose from {list(PERTURBATIONS)}")
    factory, desc = PERTURBATIONS[name]
    return factory(severity), desc.format(s=severity)
