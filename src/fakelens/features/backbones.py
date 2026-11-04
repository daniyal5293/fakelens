"""Frozen backbones that return the CLS token of *every* transformer block plus the final embedding.

    feats = backbone(x)        # x: [B,3,224,224], normalised with backbone.mean/std
    feats["cls"]   -> [B, L, D]  CLS token after each of the L blocks (RINE-style multi-layer features)
    feats["embed"] -> [B, E]     final image embedding (CLIP: after ln_post + projection)

Registry names:
    clip_vitl14   OpenAI CLIP ViT-L/14   (L=24, D=1024, E=768)   <- main model
    clip_vitb16   OpenAI CLIP ViT-B/16   (L=12, D=768,  E=512)   <- backbone-size ablation
    dinov2_vitl14 DINOv2 ViT-L/14        (L=24, D=1024, E=1024)  <- self-supervised (no text) comparison
    dummy         tiny random CNN, for CPU smoke tests only (no download)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fakelens.data.transforms import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD


class Backbone(nn.Module):
    name: str
    mean: tuple
    std: tuple
    n_layers: int
    dim: int
    embed_dim: int

    def train(self, mode: bool = True):  # always frozen / eval
        return super().train(False)


class OpenCLIPBackbone(Backbone):
    def __init__(self, arch: str, pretrained: str, name: str):
        super().__init__()
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.visual = model.visual
        del model
        self.visual.requires_grad_(False)
        self.name, self.mean, self.std = name, CLIP_MEAN, CLIP_STD
        blocks = self.visual.transformer.resblocks
        self.n_layers = len(blocks)
        self.dim = self.visual.transformer.width
        self.embed_dim = self.visual.output_dim
        self._batch_first = getattr(self.visual.transformer, "batch_first", False)
        self._buf: list[torch.Tensor] = []
        for blk in blocks:
            blk.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        out = out[0] if isinstance(out, tuple) else out
        self._buf.append(out[:, 0] if self._batch_first else out[0])  # CLS token

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict:
        self._buf = []
        embed = self.visual(x)
        cls = torch.stack(self._buf, dim=1)
        self._buf = []
        return {"cls": cls, "embed": embed}


class DINOv2Backbone(Backbone):
    def __init__(self, hub_name: str = "dinov2_vitl14", name: str = "dinov2_vitl14"):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", hub_name)
        self.model.requires_grad_(False)
        self.name, self.mean, self.std = name, IMAGENET_MEAN, IMAGENET_STD
        self.n_layers = len(self.model.blocks)
        self.dim = self.model.embed_dim
        self.embed_dim = self.model.embed_dim

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict:
        outs = self.model.get_intermediate_layers(x, n=self.n_layers, return_class_token=True, norm=False)
        cls = torch.stack([c for _, c in outs], dim=1)
        embed = self.model.norm(cls[:, -1])
        return {"cls": cls, "embed": embed}


class DummyBackbone(Backbone):
    """Deterministic random CNN with 4 'layers' so the pipeline can be smoke-tested on CPU."""

    def __init__(self, name: str = "dummy"):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.name, self.mean, self.std = name, IMAGENET_MEAN, IMAGENET_STD
        self.n_layers, self.dim, self.embed_dim = 4, 32, 16
        self.stem = nn.Conv2d(3, 32, 8, stride=8)
        self.blocks = nn.ModuleList(nn.Conv2d(32, 32, 3, padding=1) for _ in range(4))
        self.proj = nn.Linear(32, 16)
        for p in self.parameters():
            p.data = torch.randn(p.shape, generator=g) * 0.1
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        h = self.stem(x)
        cls = []
        for b in self.blocks:
            h = torch.relu(b(h)) + h
            cls.append(h.mean(dim=(2, 3)))
        cls = torch.stack(cls, dim=1)
        return {"cls": cls, "embed": self.proj(cls[:, -1])}


# Bumped whenever a registry name starts meaning a different network. Head checkpoints store it, so a
# head trained on old features is never silently paired with a new backbone (see detector.py).
BACKBONE_VERSION = {
    "clip_vitl14": "ViT-L-14-quickgelu/openai",
    "clip_vitb16": "ViT-B-16-quickgelu/openai",
    "dinov2_vitl14": "dinov2_vitl14/torchhub",
    "dummy": "dummy",
}


def build_backbone(name: str) -> Backbone:
    # The OpenAI CLIP weights were trained with QuickGELU; the plain "ViT-L-14" config uses nn.GELU,
    # which silently changes every block's activation. The "-quickgelu" configs match the original model.
    if name == "clip_vitl14":
        return OpenCLIPBackbone("ViT-L-14-quickgelu", "openai", name)
    if name == "clip_vitb16":
        return OpenCLIPBackbone("ViT-B-16-quickgelu", "openai", name)
    if name == "dinov2_vitl14":
        return DINOv2Backbone("dinov2_vitl14", name)
    if name == "dummy":
        return DummyBackbone()
    raise KeyError(f"unknown backbone {name!r}")


BACKBONES = ["clip_vitl14", "clip_vitb16", "dinov2_vitl14", "dummy"]
