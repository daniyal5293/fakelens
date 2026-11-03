"""Torch dataset reading images listed in the manifest."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from fakelens.data.manifest import resolve_path
from fakelens.data.transforms import Perturb, Preprocess


class ManifestDataset(Dataset):
    """Yields ``(tensor, label, row_index)``.

    Randomness (train aug, noise perturbation) uses ``seed + index (+ epoch * N)``. Runs are
    therefore reproducible no matter how many workers there are or what order they run in.
    """

    def __init__(self, df: pd.DataFrame, roots: dict, preprocess: Preprocess,
                 perturb: Perturb | None = None, seed: int = 0, view: int = 0):
        self.df = df.reset_index(drop=True)
        self.roots = roots
        self.preprocess = preprocess
        self.perturb = perturb
        self.seed = seed
        self.view = view      # augmented-view id when caching several views of the same image
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.df)

    def load_image(self, i: int) -> Image.Image:
        path = resolve_path(self.df.iloc[i], self.roots)
        with Image.open(path) as im:
            im.load()
            return im.copy()

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        rng = np.random.default_rng([self.seed, int(i), self.epoch, self.view])
        img = self.load_image(i)
        if self.perturb is not None:
            img = self.perturb(img, rng)
        x = self.preprocess(img, rng)
        return x, torch.tensor(float(row["label"])), i
