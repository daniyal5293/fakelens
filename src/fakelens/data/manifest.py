"""Build the single source-of-truth manifest for every experiment.

Design
------
* Scan GenImage-style folders (``<generator>/<train|val>/<ai|nature>/``) under each root.
* Sample candidates deterministically (seed), read metadata (format, size, JPEG quality) and SHA-1.
* Remove exact duplicates *globally* by SHA-1, giving evaluation splits priority
  (test > val > wild > logo > train) so nothing seen in training can leak into evaluation.
* Store paths relative to a named root (``root_key``) so the committed CSV is portable.

Usage
-----
    python -m fakelens.data.manifest --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import io
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from fakelens import LABEL_FAKE, LABEL_REAL
from fakelens.utils import ensure_dir, load_config

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# (canonical name, substrings that identify it in GenImage / Kaggle-mirror folder names)
_GENERATOR_PATTERNS: list[tuple[str, list[str]]] = [
    ("sdv14", ["sdv4", "sdv14", "v_1_4", "v1_4", "v1-4", "v1.4"]),
    ("sdv15", ["sdv5", "sdv15", "v_1_5", "v1_5", "v1-5", "v1.5"]),
    ("midjourney", ["midjourney"]),
    ("wukong", ["wukong"]),
    ("vqdm", ["vqdm"]),
    ("biggan", ["biggan"]),
    ("glide", ["glide"]),
    ("adm", []),  # matched with a word-boundary regex below ("adm" is a common substring)
]
_ADM_RE = re.compile(r"(^|[^a-z])adm([^a-z]|$)")

SPLIT_PRIORITY = {"test": 0, "val": 1, "wild": 2, "logo": 3, "train": 4}

# Standard libjpeg luminance quantisation table (quality 50).
_STD_LUMA = np.array(
    [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
     14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
     18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
     49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99],
    dtype=np.float64,
)


def _scaled_table(quality: int) -> np.ndarray:
    scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    return np.clip(np.floor((_STD_LUMA * scale + 50.0) / 100.0), 1, 255)


_SCALED_SUMS = np.array([_scaled_table(q).sum() for q in range(1, 101)])


def estimate_jpeg_quality(img: Image.Image) -> int:
    """Estimate libjpeg quality (1-100) from the luminance quantisation table.

    Uses the table *sum*, which is independent of coefficient ordering (zig-zag vs natural),
    and returns the quality whose standard scaled table sum is closest. Returns -1 if not a JPEG.
    """
    q = getattr(img, "quantization", None)
    if not q:
        return -1
    luma = np.asarray(q.get(0, next(iter(q.values()))), dtype=np.float64)
    return int(np.argmin(np.abs(_SCALED_SUMS - luma.sum())) + 1)


def infer_generator(path: str | os.PathLike) -> str | None:
    parts = [p.lower() for p in Path(path).parts]
    for part in reversed(parts):  # deepest folder wins
        for name, pats in _GENERATOR_PATTERNS:
            if name == "adm":
                if _ADM_RE.search(part):
                    return "adm"
            elif any(p in part for p in pats):
                return name
    return None


def _imagenet_class(filename: str) -> str:
    m = re.match(r"^(n\d{8})_", filename)  # ImageNet train: n01440764_123.JPEG
    if m:
        return m.group(1)
    m = re.match(r"^(\d{1,4})_", filename)  # GenImage fakes: 016_sdv4_00134.png -> class index
    return m.group(1) if m else ""


def scan_genimage(root: str | os.PathLike, root_key: str, source_split: str,
                  generator: str | None = None) -> pd.DataFrame:
    """Find every image in ``.../<source_split>/<ai|nature>/`` below ``root``."""
    root = Path(root)
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        if d.name.lower() not in ("ai", "nature") or d.parent.name.lower() != source_split:
            continue
        label = LABEL_FAKE if d.name.lower() == "ai" else LABEL_REAL
        rel_dir = d.relative_to(root)
        gen = generator or infer_generator(rel_dir) or infer_generator(root)
        for fn in filenames:
            if Path(fn).suffix.lower() in IMG_EXTS:
                rows.append({
                    "relpath": (rel_dir / fn).as_posix(),
                    "root_key": root_key,
                    "label": label,
                    "generator": gen,
                    "source_split": source_split,
                    "imagenet_class": _imagenet_class(fn),
                })
    df = pd.DataFrame(rows, columns=["relpath", "root_key", "label", "generator", "source_split", "imagenet_class"])
    return df.sort_values("relpath", kind="stable").reset_index(drop=True)


def scan_wild(root: str | os.PathLike, root_key: str = "wild_root") -> pd.DataFrame:
    """In-the-wild set: ``root/real/**`` and ``root/fake/<optional generator name>/**``."""
    root = Path(root)
    rows = []
    for cls_name, label in (("real", LABEL_REAL), ("fake", LABEL_FAKE)):
        base = root / cls_name
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            rel = p.relative_to(root)
            sub = rel.parts[1] if len(rel.parts) > 2 else "wild"
            rows.append({
                "relpath": rel.as_posix(), "root_key": root_key, "label": label,
                "generator": f"wild_{sub}" if label == LABEL_FAKE and sub != "wild" else "wild",
                "source_split": "wild", "imagenet_class": "",
            })
    return pd.DataFrame(rows, columns=["relpath", "root_key", "label", "generator", "source_split", "imagenet_class"])


def image_metadata(abs_path: str | os.PathLike) -> dict:
    try:
        data = Path(abs_path).read_bytes()
        sha1 = hashlib.sha1(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format or ""
            w, h = im.size
            q = estimate_jpeg_quality(im) if fmt == "JPEG" else -1
            im.verify()
        return {"sha1": sha1, "format": fmt, "width": w, "height": h, "jpeg_quality": q, "error": ""}
    except Exception as e:  # corrupted / unreadable file
        return {"sha1": "", "format": "", "width": -1, "height": -1, "jpeg_quality": -1, "error": repr(e)}


def add_metadata(df: pd.DataFrame, roots: dict, workers: int = 16) -> pd.DataFrame:
    paths = [resolve_path(r, roots) for r in df.itertuples()]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        meta = list(ex.map(image_metadata, paths, chunksize=32))
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(meta)], axis=1)


def resolve_path(row, roots: dict) -> str:
    rk = row["root_key"] if isinstance(row, (dict, pd.Series)) else row.root_key
    rp = row["relpath"] if isinstance(row, (dict, pd.Series)) else row.relpath
    return str(Path(roots[rk]) / rp)


def _no_images_help(root, split: str) -> str:
    """Explain an empty scan: show what IS under the root (folders, archives) so the path can be fixed."""
    root = Path(root)
    dirs, archives = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth <= 5:
            dirs.append(str(Path(dirpath).relative_to(root)) or ".")
        archives += [str(Path(dirpath, f).relative_to(root)) for f in filenames
                     if f.lower().endswith((".zip", ".tar", ".gz", ".7z")) or ".part" in f.lower() or "_part_" in f.lower()]
        if len(dirs) > 60:
            break
    msg = (f"No images found under {root} in any '<...>/{split}/ai' or '<...>/{split}/nature' folder.\n"
           f"Folders found (first 60): {dirs}\n")
    if archives:
        msg += f"Archive files found (dataset may need extracting): {archives[:10]}\n"
    return msg


def _take(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(df) <= n:
        return df.iloc[rng.permutation(len(df))]
    return df.iloc[rng.choice(len(df), size=n, replace=False)]


def _assign(cands: pd.DataFrame, plan: list[tuple[str, int]]) -> pd.DataFrame:
    """Assign consecutive chunks of an (already shuffled) candidate frame to splits."""
    out, start = [], 0
    for split, n in plan:
        chunk = cands.iloc[start:start + n].copy()
        chunk["split"] = split
        out.append(chunk)
        start += n
    return pd.concat(out) if out else cands.iloc[:0].assign(split=[])


def build_manifest(cfg: dict, roots: dict | None = None, include_wild: bool = True,
                   workers: int = 16, verbose: bool = True) -> pd.DataFrame:
    roots = roots or cfg["roots"]
    d = cfg["data"]
    rng = np.random.default_rng(cfg["seed"])
    os_ = float(d.get("oversample", 1.15))
    n_tr, n_va = d["n_train_per_class"], d["n_val_per_class"]
    n_te, n_lo = d["n_test_per_class"], d["n_logo_per_class"]
    c = lambda n: int(math.ceil(n * os_))  # noqa: E731

    cands = []
    # --- train / val: SD v1.4 train/ ---
    for key in ("train_root", "test_root"):
        if not Path(roots[key]).exists():
            raise FileNotFoundError(f"roots.{key} = {roots[key]!r} does not exist. Run "
                                    "`!find /kaggle/input -maxdepth 6 -type d` and fix the path (config or --set).")
    pool = scan_genimage(roots["train_root"], "train_root", "train", generator=None)
    pool["generator"] = pool["generator"].fillna("sdv14")
    if verbose:
        print(f"[manifest] train_root: {len(pool)} images found")
    if pool.empty:
        raise RuntimeError(_no_images_help(roots["train_root"], "train"))
    for label in (LABEL_REAL, LABEL_FAKE):
        sub = _take(pool[pool.label == label], c(n_tr + n_va), rng)
        cands.append(_assign(sub, [("val", c(n_va)), ("train", c(n_tr))]))

    # --- test / logo: val/ of every generator ---
    pool = scan_genimage(roots["test_root"], "test_root", "val")
    if pool.empty:
        raise RuntimeError(_no_images_help(roots["test_root"], "val"))
    unknown = pool[pool.generator.isna()]
    if len(unknown):
        dirs = sorted({str(Path(p).parent) for p in unknown.relpath})[:10]
        print(f"[manifest] WARNING: {len(unknown)} images in folders with no recognisable generator name, e.g. {dirs}")
    if verbose:
        print(f"[manifest] test_root: {len(pool)} images found; per generator:\n"
              f"{pool.groupby(['generator', 'label']).size().unstack(fill_value=0)}")
    for gen in d["generators"]:
        for label in (LABEL_REAL, LABEL_FAKE):
            sub = pool[(pool.generator == gen) & (pool.label == label)]
            if sub.empty:
                print(f"[manifest] WARNING: no images for generator={gen} label={label}")
                continue
            sub = _take(sub, c(n_te + n_lo), rng)
            cands.append(_assign(sub, [("test", c(n_te)), ("logo", c(n_lo))]))

    # --- optional in-the-wild set ---
    wild_root = roots.get("wild_root")
    if include_wild and wild_root and Path(wild_root).exists():
        wild = scan_wild(wild_root)
        wild["split"] = "wild"
        cands.append(wild)
        if verbose:
            print(f"[manifest] wild_root: {len(wild)} images")

    df = pd.concat(cands, ignore_index=True)
    df["sample_order"] = np.arange(len(df))
    df = add_metadata(df, roots, workers=workers)

    bad = df[df.error != ""]
    if len(bad) and verbose:
        print(f"[manifest] dropping {len(bad)} unreadable files, e.g. {bad.relpath.iloc[0]}: {bad.error.iloc[0]}")
    df = df[df.error == ""].drop(columns="error")

    # --- global dedupe: evaluation splits win ---
    df["priority"] = df["split"].map(SPLIT_PRIORITY)
    n_before = len(df)
    df = df.sort_values(["priority", "sample_order"], kind="stable").drop_duplicates("sha1", keep="first")
    if verbose:
        print(f"[manifest] removed {n_before - len(df)} exact duplicates (SHA-1)")

    # --- truncate to targets ---
    target = {"train": n_tr, "val": n_va, "test": n_te, "logo": n_lo}
    parts = []
    for (split, gen, label), g in df.groupby(["split", "generator", "label"], sort=False):
        g = g.sort_values("sample_order")
        n = target.get(split)
        if n is not None:
            if len(g) < n:
                print(f"[manifest] WARNING: {split}/{gen}/label={label} has {len(g)} < {n} after dedupe")
            g = g.iloc[:n]
        parts.append(g)
    df = pd.concat(parts).sort_values("sample_order").drop(columns=["priority", "sample_order"])
    df["uid"] = df["sha1"].str[:16]
    cols = ["uid", "split", "generator", "label", "root_key", "relpath", "source_split",
            "imagenet_class", "format", "width", "height", "jpeg_quality", "sha1"]
    df = df[cols].reset_index(drop=True)
    assert df.uid.is_unique, "uid collision"
    return df


def load_manifest(path: str | os.PathLike) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"uid": str, "imagenet_class": str, "sha1": str},
                       keep_default_na=False)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (df.assign(cls=df.label.map({LABEL_REAL: "real", LABEL_FAKE: "fake"}))
              .groupby(["split", "generator", "cls"]).size().unstack(fill_value=0))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None, help="defaults to paths.manifest in the config")
    ap.add_argument("--no-wild", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. data.n_train_per_class=100")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    df = build_manifest(cfg, include_wild=not args.no_wild, workers=args.workers)
    out = Path(args.out or cfg["paths"]["manifest"])
    ensure_dir(out.parent)
    df.to_csv(out, index=False)
    print(summarize(df))
    print(f"[manifest] wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
