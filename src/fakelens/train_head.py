"""Train lightweight heads on cached frozen features (Models A1-A4) and run every head experiment.

Models
    A1  clip_vitl14   + linear head on final embedding  (UnivFD-style replication)
    A2  clip_vitl14   + multi-layer head                (main model)
    A3  clip_vitb16   + multi-layer head                (backbone-size ablation)
    A4  dinov2_vitl14 + multi-layer head                (self-supervised, no-text comparison)

Experiments
    p1           train SD v1.4 -> test all 8 generators (GenImage protocol, Table 3)   [all seeds]
    logo         leave-one-generator-out on the LOGO pools (GenImage Table 4)
    data_eff     accuracy vs number of training images
    fewshot      add k images of an unseen generator to training
    layer_probe  linear probe on each single layer's CLS token (which depth carries forensic signal?)
    shuffle      sanity check: shuffled training labels -> ~50%

    python -m fakelens.train_head --model A2 --experiment p1
    python -m fakelens.train_head --model A2 --experiment p1 --raw      # trained on non-debiased features
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from fakelens.data.manifest import load_manifest
from fakelens.evaluate import finalize_run
from fakelens.features.backbones import BACKBONE_VERSION
from fakelens.features.extract import feature_dir, load_meta, load_split
from fakelens.metrics import compute_metrics, group_report, sigmoid
from fakelens.models.heads import build_head
from fakelens.utils import ensure_dir, get_device, load_config, save_json, seed_everything

HEAD_MODELS = {
    "A1": {"backbone": "clip_vitl14", "head": "linear"},
    "A2": {"backbone": "clip_vitl14", "head": "multilayer"},
    "A3": {"backbone": "clip_vitb16", "head": "multilayer"},
    "A4": {"backbone": "dinov2_vitl14", "head": "multilayer"},
    "D0": {"backbone": "dummy", "head": "multilayer"},  # smoke tests only
}


# ---------------------------------------------------------------- core fit / predict
def _arr(data: dict, key: str, idx=None) -> np.ndarray:
    a = data[key]
    return a if idx is None else a[idx]


def fit_head(head: torch.nn.Module, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray,
             hp: dict, batch_size: int, patience: int, device, seed: int) -> tuple[torch.nn.Module, list]:
    """Minibatch AdamW + cosine LR, early stopping on val AUROC. Inputs stay fp16 on device."""
    seed_everything(seed)
    key = head.input_key
    head = head.to(device)
    Xtr_t = torch.from_numpy(np.ascontiguousarray(Xtr)).to(device)
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(device)
    n = len(Xtr_t)
    steps_per_epoch = max(1, int(np.ceil(n / batch_size)))
    epochs = hp["epochs"]
    opt = torch.optim.AdamW(head.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hp["lr"], total_steps=epochs * steps_per_epoch,
                                                pct_start=0.1, anneal_strategy="cos")
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n, generator=g).to(device)
        tot = 0.0
        for s in range(steps_per_epoch):
            b = perm[s * batch_size:(s + 1) * batch_size]
            logits = head.logits(**{key: Xtr_t[b]})
            loss = F.binary_cross_entropy_with_logits(logits, ytr_t[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        va_logits = predict_head(head, Xva, device)
        auc = roc_auc_score(yva, va_logits) if len(np.unique(yva)) == 2 else 0.0
        hist.append({"epoch": ep, "train_loss": tot / n, "val_auroc": float(auc)})
        if auc > best + 1e-5:
            best, best_state, bad = auc, copy.deepcopy(head.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    return head, hist


@torch.no_grad()
def predict_head(head: torch.nn.Module, X: np.ndarray, device, batch_size: int = 2048) -> np.ndarray:
    head.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).to(device)
        out.append(head.logits(**{head.input_key: xb}).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, dtype=np.float32)


# ---------------------------------------------------------------- experiment runner
class HeadExperiment:
    def __init__(self, cfg: dict, model: str, raw: bool = False, device=None):
        self.cfg, self.model = cfg, model
        spec = HEAD_MODELS[model]
        self.backbone, self.head_kind = spec["backbone"], spec["head"]
        self.raw = raw
        self.tag = model + ("_raw" if raw else "")
        self.fdir = feature_dir(cfg, self.backbone, raw)
        self.meta = load_meta(self.fdir)
        self.manifest = load_manifest(cfg["paths"]["manifest"])
        self.device = device or get_device()
        hcfg = cfg["heads"]
        self.hp = dict(hcfg[self.head_kind])
        self.input_key = self.hp.pop("input")
        self.batch_size, self.patience = hcfg["batch_size"], hcfg["patience"]
        self.groups = cfg["data"]["groups"]
        self.runs = Path(cfg["paths"]["runs"])
        self.results = Path(cfg["paths"]["results"])
        self._cache: dict = {}

    def split(self, name: str, views: str = "all") -> dict:
        k = (name, views)
        if k not in self._cache:
            self._cache[k] = load_split(self.fdir, name, self.manifest, views)
        return self._cache[k]

    def new_head(self, **override):
        kw = {k: v for k, v in self.hp.items() if k in ("hidden", "dropout")}
        kw.update(override)
        return build_head(self.head_kind, self.meta["n_layers"], self.meta["dim"], self.meta["embed_dim"], **kw)

    def X(self, data: dict, idx=None, key=None):
        return _arr(data, key or self.input_key, idx)

    def _pred_frame(self, data: dict, logits: np.ndarray, idx=None) -> pd.DataFrame:
        df = data["df"] if idx is None else data["df"].iloc[idx]
        return pd.DataFrame({"uid": df.uid.to_numpy(), "generator": df.generator.to_numpy(),
                             "label": df.label.to_numpy(), "logit": logits})

    def save_checkpoint(self, head, path: Path, T: float, thr: float, extra: dict | None = None):
        torch.save({
            "kind": "head", "model": self.model, "backbone": self.backbone, "head": self.head_kind,
            "backbone_version": BACKBONE_VERSION.get(self.backbone),
            "head_kwargs": {k: v for k, v in self.hp.items() if k in ("hidden", "dropout")},
            "n_layers": self.meta["n_layers"], "dim": self.meta["dim"], "embed_dim": self.meta["embed_dim"],
            "preprocess": self.meta["preprocess"], "state_dict": head.state_dict(),
            "temperature": T, "threshold": thr, **(extra or {}),
        }, path)

    # ---- P1: official GenImage protocol
    def p1(self, seeds: list[int], shuffle_labels: bool = False) -> None:
        tr, va, te = self.split("train"), self.split("val", "clean"), self.split("test", "clean")
        exp = "shuffle" if shuffle_labels else "p1"
        for seed in seeds:
            ytr = tr["y"].copy()
            if shuffle_labels:
                ytr = np.random.default_rng(seed).permutation(ytr)
            head, hist = fit_head(self.new_head(), self.X(tr), ytr, self.X(va), va["y"],
                                  self.hp, self.batch_size, self.patience, self.device, seed)
            run = ensure_dir(self.runs / exp / self.tag / f"seed{seed}")
            val = self._pred_frame(va, predict_head(head, self.X(va), self.device))
            test = self._pred_frame(te, predict_head(head, self.X(te), self.device))
            res = finalize_run(run, self.tag, val, test, self.groups, self.cfg["experiments"]["bootstrap"],
                               extra={"seed": seed, "experiment": exp})
            head.temperature.fill_(res["temperature"])
            extra = {}
            if hasattr(head, "layer_weights"):
                extra["layer_weights"] = head.layer_weights().detach().cpu().numpy().tolist()
                save_json({"layer_weights": extra["layer_weights"]}, run / "layer_weights.json")
            self.save_checkpoint(head, run / "head.pt", res["temperature"], res["threshold_val"], extra)
            save_json(hist, run / "history.json")

    # ---- P2: leave-one-generator-out
    def logo(self, seeds: list[int]) -> pd.DataFrame:
        pool, te = self.split("logo", "clean"), self.split("test", "clean")
        gens = [g for g in self.cfg["data"]["generators"] if g in set(pool["df"].generator)]
        rows = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for held in gens:
                idx = np.where(pool["df"].generator.to_numpy() != held)[0]
                rng.shuffle(idx)
                n_va = max(2, int(0.1 * len(idx)))
                va_idx, tr_idx = idx[:n_va], idx[n_va:]
                head, _ = fit_head(self.new_head(), self.X(pool, tr_idx), pool["y"][tr_idx],
                                   self.X(pool, va_idx), pool["y"][va_idx],
                                   self.hp, self.batch_size, self.patience, self.device, seed)
                te_idx = np.where(te["df"].generator.to_numpy() == held)[0]
                val = self._pred_frame(pool, predict_head(head, self.X(pool, va_idx), self.device), va_idx)
                test = self._pred_frame(te, predict_head(head, self.X(te, te_idx), self.device), te_idx)
                run = self.runs / "logo" / self.tag / f"heldout_{held}" / f"seed{seed}"
                res = finalize_run(run, self.tag, val, test, {"heldout": [held]},
                                   n_boot=200, extra={"seed": seed, "heldout": held, "experiment": "logo"})
                for thr_key in ("test@0.5", "test@val_thr"):
                    m = res[thr_key]["per_generator"][held]
                    rows.append({"model": self.tag, "seed": seed, "heldout": held, "thr": thr_key,
                                 **{k: m[k] for k in ("acc", "auroc", "ap", "real_acc", "fake_acc", "f1")}})
        df = pd.DataFrame(rows)
        out = ensure_dir(self.results / "logo")
        df.to_csv(out / f"logo_{self.tag}.csv", index=False)
        print(df.groupby(["thr", "heldout"])[["acc", "auroc"]].mean().round(3))
        return df

    # ---- data efficiency
    def data_eff(self, sizes: list[int], seeds: list[int]) -> pd.DataFrame:
        tr, va, te = self.split("train"), self.split("val", "clean"), self.split("test", "clean")
        uids = tr["df"].uid.to_numpy()
        labels = tr["df"].label.to_numpy()
        rows = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            real_u = rng.permutation(np.unique(uids[labels == 0]))
            fake_u = rng.permutation(np.unique(uids[labels == 1]))
            for size in sizes:
                half = min(size // 2, len(real_u), len(fake_u))
                chosen = set(real_u[:half]) | set(fake_u[:half])
                idx = np.where(np.isin(uids, list(chosen)))[0]
                head, _ = fit_head(self.new_head(), self.X(tr, idx), tr["y"][idx], self.X(va), va["y"],
                                   self.hp, self.batch_size, self.patience, self.device, seed)
                test = self._pred_frame(te, predict_head(head, self.X(te), self.device))
                test["prob"] = sigmoid(test.logit)
                rep = group_report(test, self.groups, 0.5)
                for gname, m in rep["groups"].items():
                    rows.append({"model": self.tag, "seed": seed, "train_size": 2 * half, "group": gname,
                                 "acc": m["acc"], "auroc": m["auroc"], "mean_acc": m["mean_acc_over_generators"]})
        df = pd.DataFrame(rows)
        df.to_csv(ensure_dir(self.results / "data_eff") / f"data_eff_{self.tag}.csv", index=False)
        print(df[df.group == "unseen"].groupby("train_size")[["acc", "auroc"]].mean().round(3))
        return df

    # ---- few-shot adaptation to an unseen generator
    def fewshot(self, ks: list[int], seeds: list[int]) -> pd.DataFrame:
        tr, va, te = self.split("train"), self.split("val", "clean"), self.split("test", "clean")
        pool = self.split("logo", "clean")
        rows = []
        for seed in seeds:
            for held in self.groups["unseen"]:
                pidx = np.where(pool["df"].generator.to_numpy() == held)[0]
                te_idx = np.where(te["df"].generator.to_numpy() == held)[0]
                if len(pidx) == 0 or len(te_idx) == 0:
                    continue
                rng = np.random.default_rng(seed)
                p_real = rng.permutation(pidx[pool["y"][pidx] == 0])
                p_fake = rng.permutation(pidx[pool["y"][pidx] == 1])
                for k in ks:
                    add = np.concatenate([p_real[:k // 2], p_fake[:k // 2]])
                    Xtr = np.concatenate([self.X(tr), self.X(pool, add)])
                    ytr = np.concatenate([tr["y"], pool["y"][add]])
                    head, _ = fit_head(self.new_head(), Xtr, ytr, self.X(va), va["y"],
                                       self.hp, self.batch_size, self.patience, self.device, seed)
                    logits = predict_head(head, self.X(te, te_idx), self.device)
                    m = compute_metrics(te["y"][te_idx], sigmoid(logits), 0.5)
                    rows.append({"model": self.tag, "seed": seed, "generator": held, "k": k,
                                 **{x: m[x] for x in ("acc", "auroc", "real_acc", "fake_acc")}})
        df = pd.DataFrame(rows)
        df.to_csv(ensure_dir(self.results / "fewshot") / f"fewshot_{self.tag}.csv", index=False)
        print(df.groupby(["k"])[["acc", "auroc"]].mean().round(3))
        return df

    # ---- per-layer linear probes
    def layer_probe(self, seed: int = 0) -> pd.DataFrame:
        tr, va, te = self.split("train"), self.split("val", "clean"), self.split("test", "clean")
        rows = []
        hp = dict(self.cfg["heads"]["multilayer"])
        hp.pop("input")
        for layer in range(self.meta["n_layers"]):
            head = build_head("multilayer", self.meta["n_layers"], self.meta["dim"], self.meta["embed_dim"],
                              layers=[layer], hidden=0)
            Xtr, Xva, Xte = (d["cls"][:, layer:layer + 1] for d in (tr, va, te))
            head.layer_idx.zero_()  # inputs are already sliced to the single layer
            head, _ = fit_head(head, Xtr, tr["y"], Xva, va["y"], hp, self.batch_size, self.patience,
                               self.device, seed)
            test = self._pred_frame(te, predict_head(head, Xte, self.device))
            test["prob"] = sigmoid(test.logit)
            rep = group_report(test, self.groups, 0.5)
            for gname, m in rep["groups"].items():
                rows.append({"backbone": self.backbone, "layer": layer + 1, "group": gname,
                             "acc": m["acc"], "auroc": m["auroc"]})
            print(f"layer {layer + 1:2d}: unseen acc={rep['groups'].get('unseen', {}).get('acc', float('nan')):.3f}")
        df = pd.DataFrame(rows)
        df.to_csv(ensure_dir(self.results / "layer_probe") / f"layer_probe_{self.backbone}{'_raw' if self.raw else ''}.csv",
                  index=False)
        return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", required=True, choices=list(HEAD_MODELS))
    ap.add_argument("--experiment", default="p1",
                    choices=["p1", "logo", "data_eff", "fewshot", "layer_probe", "shuffle"])
    ap.add_argument("--raw", action="store_true", help="use features extracted without debiasing")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    exp = HeadExperiment(cfg, args.model, raw=args.raw)
    e = cfg["experiments"]
    if args.experiment == "p1":
        exp.p1(seeds)
    elif args.experiment == "shuffle":
        exp.p1(seeds[:1], shuffle_labels=True)
    elif args.experiment == "logo":
        exp.logo(seeds[:1])
    elif args.experiment == "data_eff":
        exp.data_eff(e["data_efficiency_sizes"], seeds)
    elif args.experiment == "fewshot":
        exp.fewshot(e["fewshot_k"], seeds)
    elif args.experiment == "layer_probe":
        exp.layer_probe(seeds[0])


if __name__ == "__main__":
    main()
