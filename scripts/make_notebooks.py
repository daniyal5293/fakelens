"""Generate the thin Kaggle driver notebooks in notebooks/ (all logic lives in src/fakelens).

    python scripts/make_notebooks.py
"""
import json
from pathlib import Path

SETUP = """# --- Kaggle setup (run first in every notebook) ---
# Settings: Accelerator = GPU (T4 x2 or P100), Internet = ON.
# Add Data: vtphatt2/GenImage-stable-diffusion-v1-4 and vtphatt2/GenImage-Validation
#           (+ the output dataset of the previous notebook, if you split stages across sessions).
import os, subprocess
REPO = "https://github.com/daniyal5293/fakelens.git"   # <- change, or upload the repo as a Kaggle dataset
if not os.path.exists("/kaggle/working/fakelens"):
    subprocess.run(["git", "clone", REPO, "/kaggle/working/fakelens"], check=True)
else:
    subprocess.run(["git", "-C", "/kaggle/working/fakelens", "pull"], check=False)
os.chdir("/kaggle/working/fakelens")
!pip install -q open_clip_torch
!pip install -q -e . --no-deps
# Show the dataset folder structure (look for .../train/ai, .../train/nature, .../val/ai, .../val/nature):
!find /kaggle/input -maxdepth 7 -type d | head -80"""

ROOTS = """# Check the dataset folder names printed above and fix them here if they differ.
CFG = "configs/default.yaml"
SET = "--set roots.train_root=/kaggle/input/datasets/vtphatt2/genimage-stable-diffusion-v1-4 roots.test_root=/kaggle/input/datasets/vtphatt2/genimage-validation"
RUNS = "/kaggle/working/runs\""""

NB = {
    "00_data_manifest_audit": [
        ("md", "# 00 · Data: manifest, dedupe, bias audit\nBuilds `data/manifests/manifest.csv` "
               "(commit it), then audits the JPEG/size shortcut in GenImage."),
        ("code", SETUP), ("code", ROOTS),
        ("code", "!python -m fakelens.data.manifest --config $CFG $SET"),
        ("code", "!python -m fakelens.analysis.audit --config $CFG $SET"),
        ("code", "from IPython.display import Image; Image('results/figures/audit_format_size.png')"),
        ("code", "from fakelens.utils import capture_environment; capture_environment('results/env.json')['gpu']"
                 " if __import__('torch').cuda.is_available() else 'no gpu'"),
    ],
    "01_extract_features": [
        ("md", "# 01 · Frozen feature extraction\nEach backbone runs **once** per image. "
               "Save this notebook's output (Save Version) and attach it as input to 02 if you change sessions."),
        ("code", SETUP), ("code", ROOTS),
        ("code", "!python -m fakelens.features.extract --config $CFG $SET --backbone clip_vitl14 --splits train val test logo"),
        ("code", "!python -m fakelens.features.extract --config $CFG $SET --backbone clip_vitl14 --splits train val test --raw"),
        ("code", "!python -m fakelens.features.extract --config $CFG $SET --backbone clip_vitb16 --splits train val test"),
        ("code", "!python -m fakelens.features.extract --config $CFG $SET --backbone dinov2_vitl14 --splits train val test"),
    ],
    "02_clip_heads": [
        ("md", "# 02 · CLIP heads (A1–A4) and all head experiments\nP1 (GenImage protocol), bias ablation, "
               "LOGO, data efficiency, few-shot, layer probes, shuffled-label sanity check, t-SNE."),
        ("code", SETUP), ("code", ROOTS),
        ("code", "for m in ['A1', 'A2', 'A3', 'A4']:\n    !python -m fakelens.train_head --config $CFG $SET --model {m} --experiment p1"),
        ("code", "!python -m fakelens.train_head --config $CFG $SET --model A2 --experiment p1 --raw\n"
                 "!python -m fakelens.train_head --config $CFG $SET --model A2 --experiment shuffle"),
        ("code", "for m in ['A1', 'A2']:\n    for e in ['logo', 'data_eff', 'fewshot']:\n"
                 "        !python -m fakelens.train_head --config $CFG $SET --model {m} --experiment {e}"),
        ("code", "for m in ['A2', 'A3', 'A4']:\n    !python -m fakelens.train_head --config $CFG $SET --model {m} --experiment layer_probe --seeds 0"),
        ("code", "!python -m fakelens.analysis.embeddings --config $CFG $SET --model A2 --ckpt $RUNS/p1/A2/seed0/head.pt\n"
                 "!python -m fakelens.evaluate aggregate --runs $RUNS/p1 --out results/p1\n"
                 "print(open('results/p1/summary.md').read())"),
    ],
    "03_resnet_baseline": [
        ("md", "# 03 · ResNet-18 baseline (Model B)\nSame manifests, preprocessing and evaluation as the CLIP heads."),
        ("code", SETUP), ("code", ROOTS),
        ("code", "!python -m fakelens.train_resnet --config $CFG $SET --experiment p1"),
        ("code", "!python -m fakelens.train_resnet --config $CFG $SET --experiment p1 --raw --seeds 0\n"
                 "!python -m fakelens.train_resnet --config $CFG $SET --experiment p1 --no-aug --seeds 0"),
        ("code", "!python -m fakelens.train_resnet --config $CFG $SET resnet.epochs=8 --experiment logo --seeds 0"),
        ("code", "!python -m fakelens.train_resnet --config $CFG $SET resnet.epochs=8 --experiment data_eff --seeds 0"),
    ],
    "04_robustness_wild": [
        ("md", "# 04 · Robustness (JPEG, resize, low-res, blur, noise, social-media chain) and in-the-wild set"),
        ("code", SETUP), ("code", ROOTS),
        ("code", "names = ['A1', 'A2', 'A3', 'A4', 'B', 'A2_raw', 'B_raw', 'B_noaug']\n"
                 "ckpts = ' '.join(f\"{RUNS}/p1/{n}/seed0/{'model' if n.startswith('B') else 'head'}.pt\" for n in names)\n"
                 "N = ' '.join(names)\n"
                 "!python -m fakelens.robustness --config $CFG $SET --ckpts $ckpts --names $N\n"
                 "!python -m fakelens.robustness --config $CFG $SET --ckpts $ckpts --names $N --split wild --conditions clean"),
    ],
    "05_analysis_figures": [
        ("md", "# 05 · Tables, statistics, figures, error analysis"),
        ("code", SETUP), ("code", ROOTS),
        ("code", "!python -m fakelens.evaluate aggregate --runs $RUNS/p1 --out results/p1\n"
                 "!python -m fakelens.evaluate compare --a $RUNS/p1/A2/seed0 --b $RUNS/p1/B/seed0 --out results/p1/A2_vs_B.json\n"
                 "!python -m fakelens.evaluate compare --a $RUNS/p1/A2/seed0 --b $RUNS/p1/A1/seed0 --out results/p1/A2_vs_A1.json\n"
                 "!python -m fakelens.evaluate compare --a $RUNS/p1/A2/seed0 --b $RUNS/p1/A2_raw/seed0 --out results/p1/A2_vs_A2raw.json"),
        ("code", "!python -m fakelens.analysis.plots --results results --runs $RUNS"),
        ("code", "!python -m fakelens.analysis.errors --config $CFG $SET --run $RUNS/p1/A2/seed0 --ckpt $RUNS/p1/A2/seed0/head.pt\n"
                 "!python -m fakelens.analysis.errors --config $CFG $SET --run $RUNS/p1/B/seed0 --ckpt $RUNS/p1/B/seed0/model.pt"),
        ("code", "import glob\nfrom IPython.display import Image, display\n"
                 "for f in sorted(glob.glob('results/figures/*.png')):\n    print(f); display(Image(f))"),
    ],
    "99_run_everything": [
        ("md", "# 99 · Run the whole pipeline in one go\n"
               "Use this after notebook 00 has worked interactively. Click **Save Version → Save & Run All (Commit)**. "
               "Kaggle runs it in the background (up to 12 h) and keeps everything written to `/kaggle/working` "
               "in the notebook's **Output** tab. Download `fakelens_results.zip` from there when it finishes."),
        ("code", SETUP), ("code", ROOTS),
        ("code", "# Write a Kaggle config with the dataset paths from the cell above\n"
                 "import yaml\n"
                 "cfg = yaml.safe_load(open('configs/default.yaml'))\n"
                 "for kv in SET.replace('--set', '').split():\n"
                 "    k, v = kv.split('=', 1); cfg['roots'][k.split('.')[1]] = v\n"
                 "yaml.safe_dump(cfg, open('configs/kaggle.yaml', 'w'))\n"
                 "print(cfg['roots'])"),
        ("code", "!CFG=configs/kaggle.yaml bash scripts/run_all.sh 2>&1 | tee /kaggle/working/run_all.log"),
        ("code", "!cd /kaggle/working/fakelens && zip -qr /kaggle/working/fakelens_results.zip results data/manifests checkpoints\n"
                 "!ls -lh /kaggle/working/fakelens_results.zip"),
        ("code", "# Mark the Kaggle version as 'Error' (not 'Successful') if the pipeline stopped early.\n"
                 "# The zip above still holds whatever finished, so partial results are never lost.\n"
                 "log = open('/kaggle/working/run_all.log').read()\n"
                 "assert 'Done. Tables' in log, 'Pipeline stopped early - read the last lines of run_all.log above'\n"
                 "print('Pipeline finished OK')"),
    ],
    "07_all8_demo_model": [
        ("md", "# 07 · Demo model trained on all 8 generators\n"
               "Leave-one-generator-out showed that training on more generators generalises best "
               "(A2 detects a never-seen 8th generator with 95% accuracy). This notebook trains A2's head on "
               "**all 8** GenImage generators for the demo. The report keeps the official SD v1.4-only protocol (P1).\n\n"
               "Run with **Save Version → Save & Run All (Commit)**. Download `fakelens_all8.zip` from the Output tab."),
        ("code", SETUP), ("code", ROOTS),
        ("code", "# Use the committed manifest (identical splits to the main run); build it only if it is missing\n"
                 "import os\n"
                 "if not os.path.exists('data/manifests/manifest.csv'):\n"
                 "    !python -m fakelens.data.manifest --config $CFG $SET\n"
                 "!wc -l data/manifests/manifest.csv"),
        ("code", "# CLIP ViT-L/14 features for the 8 generator pools and the test split (no SD v1.4 train split needed)\n"
                 "!python -m fakelens.features.extract --config $CFG $SET --backbone clip_vitl14 --splits logo test"),
        ("code", "!python -m fakelens.train_head --config $CFG $SET --model A2 --experiment all8"),
        ("code", "!mkdir -p checkpoints && cp $RUNS/all8/A2/seed0/head.pt checkpoints/A2_all8_head.pt\n"
                 "!cd /kaggle/working/fakelens && zip -qr /kaggle/working/fakelens_all8.zip results/all8 checkpoints/A2_all8_head.pt\n"
                 "!ls -lh /kaggle/working/fakelens_all8.zip"),
        ("code", "import os\n"
                 "assert os.path.exists('/kaggle/working/fakelens_all8.zip'), 'Something failed - read the output above'\n"
                 "print('Done')"),
    ],
    "06_demo": [
        ("md", "# 06 · Gradio demo\nCopies the A2 and B checkpoints to `checkpoints/` and launches a public share link."),
        ("code", SETUP), ("code", ROOTS),
        ("code", "!pip install -q gradio\n!mkdir -p checkpoints\n"
                 "!cp $RUNS/p1/A2/seed0/head.pt checkpoints/A2_head.pt\n!cp $RUNS/p1/B/seed0/model.pt checkpoints/B_model.pt"),
        ("code", "!python app/app.py --share"),
    ],
}


def cell(kind, src):
    lines = src.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


def main():
    out = Path(__file__).resolve().parents[1] / "notebooks"
    out.mkdir(exist_ok=True)
    for name, cells in NB.items():
        nb = {"cells": [cell(k, s) for k, s in cells],
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                           "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 5}
        (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print("wrote", out / f"{name}.ipynb")


if __name__ == "__main__":
    main()
