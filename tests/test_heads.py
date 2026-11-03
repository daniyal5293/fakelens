import numpy as np
import torch

from fakelens.models.heads import LinearHead, MultiLayerHead
from fakelens.train_head import fit_head, predict_head


def test_multilayer_shapes_and_weights():
    h = MultiLayerHead(n_layers=6, dim=32, hidden=16)
    out = h(cls=torch.randn(5, 6, 32))
    assert out.shape == (5,)
    w = h.layer_weights()
    assert torch.allclose(w.sum(), torch.tensor(1.0))


def test_temperature_scales_logits():
    h = LinearHead(8)
    e = torch.randn(3, 8)
    base = h(embed=e)
    h.temperature.fill_(2.0)
    assert torch.allclose(h(embed=e), base / 2)


def test_head_learns_separable_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400).astype(np.float32)
    X = rng.normal(0, 1, (400, 4, 16)).astype(np.float16)
    X[:, 2, :4] += (y * 4 - 2).astype(np.float16)[:, None]  # signal only in layer 2
    hp = {"lr": 3e-3, "weight_decay": 1e-4, "epochs": 30}
    h, hist = fit_head(MultiLayerHead(4, 16, hidden=16, dropout=0.1), X[:300], y[:300], X[300:], y[300:],
                       hp, 64, 10, torch.device("cpu"), 0)
    acc = ((predict_head(h, X[300:], torch.device("cpu")) > 0) == y[300:]).mean()
    assert acc > 0.9
    assert int(h.layer_weights().argmax()) == 2  # learned layer importance finds the signal
