"""Shared helpers: config loading, seeding, device, JSON I/O, environment capture."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | os.PathLike = "configs/default.yaml", overrides: list[str] | None = None) -> dict:
    """Load YAML config. `overrides` are 'a.b.c=value' strings (value parsed as YAML)."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(prefer: str | None = None):
    import torch

    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj: Any, path: str | os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_NumpyEncoder)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def capture_environment(path: str | os.PathLike) -> dict:
    """Record Python/torch/CUDA/GPU + pip freeze for the reproducibility section of the report."""
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120)
        info["pip_freeze"] = freeze.stdout.splitlines()
    except Exception as e:  # pragma: no cover - best effort
        info["pip_freeze_error"] = str(e)
    save_json(info, path)
    return info
