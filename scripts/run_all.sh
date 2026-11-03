#!/usr/bin/env bash
# Full FakeLens pipeline, in order. This is the source of truth the notebooks mirror.
# Run from the repo root on a GPU machine (Kaggle). Total time is roughly 6-8 GPU hours,
# so split it across sessions at the stage markers below if needed.
set -euo pipefail
CFG=${CFG:-configs/default.yaml}
RUNS=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['paths']['runs'])")
S0="--seeds 0"

# ---------------- Stage 0: data ----------------
python -m fakelens.data.manifest --config "$CFG"
python -m fakelens.analysis.audit --config "$CFG"
python -c "from fakelens.utils import capture_environment; capture_environment('results/env.json')"

# ---------------- Stage 1: features (frozen backbones, once) ----------------
python -m fakelens.features.extract --config "$CFG" --backbone clip_vitl14 --splits train val test logo
python -m fakelens.features.extract --config "$CFG" --backbone clip_vitl14 --splits train val test --raw
python -m fakelens.features.extract --config "$CFG" --backbone clip_vitb16 --splits train val test
python -m fakelens.features.extract --config "$CFG" --backbone dinov2_vitl14 --splits train val test

# ---------------- Stage 2: CLIP heads (seconds each) ----------------
for M in A1 A2 A3 A4; do python -m fakelens.train_head --config "$CFG" --model $M --experiment p1; done
python -m fakelens.train_head --config "$CFG" --model A2 --experiment p1 --raw            # bias ablation
python -m fakelens.train_head --config "$CFG" --model A2 --experiment shuffle             # sanity: ~50%
for M in A1 A2; do
  python -m fakelens.train_head --config "$CFG" --model $M --experiment logo
  python -m fakelens.train_head --config "$CFG" --model $M --experiment data_eff
  python -m fakelens.train_head --config "$CFG" --model $M --experiment fewshot
done
for M in A2 A3 A4; do python -m fakelens.train_head --config "$CFG" --model $M --experiment layer_probe $S0; done
python -m fakelens.analysis.embeddings --config "$CFG" --model A2 --ckpt "$RUNS/p1/A2/seed0/head.pt"
python -m fakelens.analysis.embeddings --config "$CFG" --model A1

# ---------------- Stage 3: ResNet-18 baseline ----------------
python -m fakelens.train_resnet --config "$CFG" --experiment p1
python -m fakelens.train_resnet --config "$CFG" --experiment p1 --raw $S0
python -m fakelens.train_resnet --config "$CFG" --experiment p1 --no-aug $S0
python -m fakelens.train_resnet --config "$CFG" --experiment logo $S0 --set resnet.epochs=8
python -m fakelens.train_resnet --config "$CFG" --experiment data_eff $S0 --set resnet.epochs=8

# ---------------- Stage 4: robustness + in-the-wild ----------------
CKPTS="$RUNS/p1/A1/seed0/head.pt $RUNS/p1/A2/seed0/head.pt $RUNS/p1/A3/seed0/head.pt $RUNS/p1/A4/seed0/head.pt \
       $RUNS/p1/B/seed0/model.pt $RUNS/p1/A2_raw/seed0/head.pt $RUNS/p1/B_raw/seed0/model.pt $RUNS/p1/B_noaug/seed0/model.pt"
NAMES="A1 A2 A3 A4 B A2_raw B_raw B_noaug"
python -m fakelens.robustness --config "$CFG" --ckpts $CKPTS --names $NAMES
python -m fakelens.robustness --config "$CFG" --ckpts $CKPTS --names $NAMES --split wild --conditions clean || true

# ---------------- Stage 5: tables, statistics, figures, errors ----------------
python -m fakelens.evaluate aggregate --runs "$RUNS/p1" --out results/p1
python -m fakelens.evaluate compare --a "$RUNS/p1/A2/seed0" --b "$RUNS/p1/B/seed0" --out results/p1/A2_vs_B.json
python -m fakelens.evaluate compare --a "$RUNS/p1/A2/seed0" --b "$RUNS/p1/A1/seed0" --out results/p1/A2_vs_A1.json
python -m fakelens.evaluate compare --a "$RUNS/p1/A2/seed0" --b "$RUNS/p1/A2_raw/seed0" --out results/p1/A2_vs_A2raw.json
python -m fakelens.analysis.plots --results results --runs "$RUNS"
python -m fakelens.analysis.errors --config "$CFG" --run "$RUNS/p1/A2/seed0" --ckpt "$RUNS/p1/A2/seed0/head.pt"
python -m fakelens.analysis.errors --config "$CFG" --run "$RUNS/p1/B/seed0" --ckpt "$RUNS/p1/B/seed0/model.pt"

# ---------------- demo checkpoints ----------------
mkdir -p checkpoints
cp "$RUNS/p1/A2/seed0/head.pt" checkpoints/A2_head.pt
cp "$RUNS/p1/B/seed0/model.pt" checkpoints/B_model.pt
echo "Done. Tables: results/p1/summary.md   Figures: results/figures/"
