"""End-to-end detector loaded from a checkpoint (head.pt or model.pt). Used by robustness eval,
explainability and the Gradio demo, so preprocessing always matches what the model was trained with.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fakelens.data.transforms import Preprocess
from fakelens.features.backbones import BACKBONE_VERSION, Backbone, build_backbone
from fakelens.models.heads import build_head
from fakelens.models.resnet import ResNet18Detector

_BACKBONE_CACHE: dict[tuple[str, str], Backbone] = {}


def get_backbone(name: str, device) -> Backbone:
    key = (name, str(device))
    if key not in _BACKBONE_CACHE:
        _BACKBONE_CACHE[key] = build_backbone(name).to(device).eval()
    return _BACKBONE_CACHE[key]


class Detector:
    def __init__(self, ckpt_path: str | Path, device=None, name: str | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.ckpt_path = str(ckpt_path)
        self.kind = ck["kind"]
        self.name = name or ck.get("model", Path(ckpt_path).parent.parent.name)
        p = ck["preprocess"]
        self.preprocess = Preprocess(mean=tuple(p["mean"]), std=tuple(p["std"]), crop=p["crop"],
                                     debias_q=p["debias_q"], min_side_resize=p["min_side_resize"], train=False)
        self.threshold = float(ck.get("threshold", 0.5))
        self.temperature = float(ck.get("temperature", 1.0))
        self.layer_weights = ck.get("layer_weights")
        if self.kind == "head":
            self.backbone_name = ck["backbone"]
            want, got = BACKBONE_VERSION.get(self.backbone_name), ck.get("backbone_version")
            if want != got:
                warnings.warn(f"{ckpt_path}: head was trained on backbone '{got or 'unknown (old checkpoint)'}' "
                              f"but '{want}' is loaded now - predictions will be unreliable. Retrain the head.")
            self.backbone = get_backbone(self.backbone_name, self.device)
            self.head = build_head(ck["head"], ck["n_layers"], ck["dim"], ck["embed_dim"], **ck["head_kwargs"])
            self.head.load_state_dict(ck["state_dict"])
            self.head.to(self.device).eval()
            self.net = None
        elif self.kind == "resnet":
            self.backbone_name = None
            self.net = ResNet18Detector(pretrained=False)
            self.net.load_state_dict(ck["state_dict"])
            self.net.to(self.device).eval()
        else:
            raise ValueError(self.kind)

    @property
    def preprocess_key(self) -> tuple:
        p = self.preprocess
        return (p.mean, p.std, p.crop, p.debias_q, p.min_side_resize)

    @torch.no_grad()
    def logits_from_feats(self, feats: dict) -> torch.Tensor:
        return self.head(embed=feats["embed"], cls=feats["cls"]).float()

    @torch.no_grad()
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Calibrated (temperature-scaled) logits for a preprocessed batch."""
        x = x.to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            if self.kind == "head":
                return self.logits_from_feats(self.backbone(x))
            return self.net(x).float()

    def prob_fn(self):
        """Batch tensor -> P(fake) tensor; for occlusion maps."""
        return lambda x: torch.sigmoid(self.logits(x))

    def predict_pil(self, img: Image.Image) -> float:
        x = self.preprocess(img).unsqueeze(0)
        return float(torch.sigmoid(self.logits(x))[0])

    def predict_pils(self, imgs: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(imgs), batch_size):
            x = torch.stack([self.preprocess(im) for im in imgs[i:i + batch_size]])
            out.append(torch.sigmoid(self.logits(x)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)
