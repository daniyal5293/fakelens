"""Explainability: occlusion sensitivity (works for any detector) and Grad-CAM (ResNet-18).

Occlusion: slide a patch filled with the dataset mean over the input and record how much P(fake)
*drops*. Positive regions are evidence the model uses for "fake". It is model-agnostic, so the CLIP
head and the ResNet are explained the same way.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps


@torch.no_grad()
def occlusion_map(prob_fn, x: torch.Tensor, patch: int = 32, stride: int = 16, batch: int = 64) -> np.ndarray:
    """x: [1,3,H,W] preprocessed. Returns [H,W] map of P(fake) drop (positive = supports 'fake')."""
    _, _, H, W = x.shape
    base = float(prob_fn(x)[0])
    ys = list(range(0, H - patch + 1, stride))
    xs = list(range(0, W - patch + 1, stride))
    coords = [(i, j) for i in ys for j in xs]
    drops = np.zeros(len(coords), dtype=np.float32)
    for k in range(0, len(coords), batch):
        chunk = coords[k:k + batch]
        xb = x.repeat(len(chunk), 1, 1, 1).clone()
        for n, (i, j) in enumerate(chunk):
            xb[n, :, i:i + patch, j:j + patch] = 0.0  # 0 in normalised space = dataset mean colour
        drops[k:k + len(chunk)] = base - prob_fn(xb).float().cpu().numpy()
    heat = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    for d, (i, j) in zip(drops, coords):
        heat[i:i + patch, j:j + patch] += d
        cnt[i:i + patch, j:j + patch] += 1
    return heat / np.maximum(cnt, 1)


def gradcam_resnet(model, x: torch.Tensor) -> np.ndarray:
    """Grad-CAM on layer4 of a ResNet18Detector for the 'fake' logit. Returns [H,W] in [0,1]."""
    net = model.net
    acts, grads = {}, {}
    h1 = net.layer4.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = net.layer4.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    try:
        model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            logit = model.logits(x)
            logit.sum().backward()
    finally:
        h1.remove()
        h2.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return cam / (cam.max() + 1e-8)


def overlay(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45, symmetric: bool = True) -> np.ndarray:
    """Blend a heatmap over an RGB uint8 image. Symmetric maps use a diverging colormap centred at 0."""
    if symmetric:
        m = np.abs(heat).max() + 1e-8
        norm = (heat / m + 1) / 2
        cmap = colormaps["RdBu_r"]
    else:
        norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        cmap = colormaps["magma"]
    colored = (cmap(norm)[..., :3] * 255).astype(np.float32)
    return (rgb.astype(np.float32) * (1 - alpha) + colored * alpha).clip(0, 255).astype(np.uint8)
