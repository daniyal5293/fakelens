# FakeLens: Generalizable AI-Generated Image Detection with Frozen CLIP Features

Course project for **Deep Learning for Perception**, by Daniyal Ahmed Khan, Abdul Hadi and Taha Ali Khan.

FakeLens labels an image as **Real** or **AI-generated**. We train only on **Stable Diffusion v1.4** images and test on **8 generators**, 6 of which the model never sees. We also measure robustness to JPEG compression, resizing, blur and noise.

> The output is a model prediction, not forensic proof. See [Limitations](#limitations-and-responsible-use).

---

## What makes this project different

| | What we do | Why it matters |
|---|---|---|
| **Official protocol** | Train on SD v1.4 and test on all 8 GenImage generators (GenImage Table 3), plus leave-one-generator-out (Table 4). | Our numbers can be compared directly with published ones. |
| **Bias control** | We audit GenImage for the JPEG-format and image-size shortcut ([Grommelt et al., ECCV-W 2024](https://arxiv.org/abs/2403.17608)) and remove it. Every image of both classes is re-encoded as JPEG at quality 96, then cropped without resizing. | Without this, a detector can score highly by learning "PNG = fake". We show the shortcut exists with a metadata-only classifier, then show that removing it changes the results. |
| **Multi-layer frozen CLIP** | A2 reads the CLS token from **all 24** ViT-L/14 blocks and learns a weight per layer before an MLP (RINE-style, ECCV 2024). The backbone stays frozen. | Forensic traces sit in intermediate layers, while the final CLIP embedding is semantic. A per-layer probe and the learned weights show *which depth* carries the signal. |
| **Controlled comparisons** | A1 (UnivFD-style linear) vs A2 (multi-layer) vs A3 (ViT-B/16) vs A4 (DINOv2) vs B (ResNet-18), all on identical splits, preprocessing and evaluation code. | Each comparison isolates one factor: depth, backbone size, text supervision, or frozen vs fine-tuned. |
| **Statistics** | 3 seeds (mean ± std), bootstrap 95% CIs, McNemar paired tests, ECE with temperature scaling, and thresholds chosen on validation data only. | Differences are reported with their uncertainty. |
| **Beyond accuracy** | Data-efficiency curves, few-shot adaptation to an unseen generator, t-SNE, occlusion and Grad-CAM maps, a confident-error gallery, and an in-the-wild set from modern generators. | We explain *why* the models behave as they do and where they fail. |

## Models

| ID | Backbone | Trainable part | Role |
|---|---|---|---|
| A1 | CLIP ViT-L/14 (OpenAI), frozen | Linear layer on the final embedding | UnivFD-style replication |
| **A2** | CLIP ViT-L/14, frozen | Per-layer LayerNorm, softmax layer weights, 2-layer MLP | **Main model** |
| A3 | CLIP ViT-B/16, frozen | Same as A2 | Backbone-size ablation |
| A4 | DINOv2 ViT-L/14, frozen | Same as A2 | Self-supervised features with no text supervision |
| B | ResNet-18 (ImageNet), **fully fine-tuned** | All layers | Proposal baseline |

Variants: `A2_raw` and `B_raw` are trained without the debias step (bias ablation). `B_noaug` is trained without blur/JPEG augmentation.

## Data

* **Source:** [GenImage](https://github.com/GenImage-Dataset/GenImage) (Zhu et al., NeurIPS 2023), through the Kaggle mirror by `vtphatt2`:
  * [`GenImage-stable-diffusion-v1-4`](https://www.kaggle.com/datasets/vtphatt2/GenImage-stable-diffusion-v1-4): training source;
  * [`GenImage-Validation`](https://www.kaggle.com/datasets/vtphatt2/GenImage-Validation): the val split of all 8 generators, used for testing.
* **Licence:** CC BY-NC-SA 4.0. We use it for non-commercial academic work only and do not redistribute images. The repo contains only the manifest (relative paths and hashes).

| Split | Source | Real | Fake |
|---|---|---|---|
| train | SD v1.4 `train/` | 5,000 | 5,000 |
| val | SD v1.4 `train/` (different files) | 500 | 500 |
| test | `val/` of each of the 8 generators | 500 × 8 | 500 × 8 |
| logo | rest of `val/`, for leave-one-generator-out | 1,000 × 8 | 1,000 × 8 |
| wild | optional: team photos plus modern generators (see `data/wild_sources.csv`) | ~100 | ~150 |

* **Leakage control:** duplicates are removed globally by **SHA-1 of the file bytes**, so the same image can never appear in two splits. Evaluation splits get priority when a duplicate is found. This is checked by `tests/test_manifest.py`.
* **Reporting groups:**
  * in-distribution: SD v1.4;
  * near-distribution: SD v1.5;
  * **unseen** (the headline number): Midjourney, ADM, GLIDE, Wukong, VQDM and BigGAN.
* The final counts after dedupe are printed by the manifest step and recorded in `results/`.

## Repository layout

```
configs/default.yaml        every hyper-parameter, path and experiment setting
data/manifests/             manifest.csv (commit this), the split definition
data/wild_sources.csv       provenance + licence for the in-the-wild images
src/fakelens/
  data/                     manifest builder, preprocessing + perturbations, dataset
  features/                 frozen backbones (all-layer CLS), feature caching
  models/                   heads (linear, multi-layer), ResNet-18
  train_head.py             A1-A4 + experiments: p1, logo, data_eff, fewshot, layer_probe, shuffle
  train_resnet.py           Model B: p1, logo, data_eff
  evaluate.py               shared metrics, temperature, thresholds, bootstrap, McNemar, aggregation
  robustness.py             perturbation grid, every model on identical perturbed images
  detector.py               end-to-end checkpoint loader (robustness, explanations, demo)
  analysis/                 audit, plots, t-SNE, occlusion/Grad-CAM, error gallery
notebooks/                  thin Kaggle drivers 00-06 (generated by scripts/make_notebooks.py)
scripts/run_all.sh          the whole pipeline in order
app/app.py                  Gradio demo
tests/                      unit tests + an end-to-end smoke test on a toy dataset (CPU)
```

## Reproducing

### 1. On Kaggle (GPU)

1. Create a notebook with **GPU** and **Internet** on. Add the two GenImage datasets above.
2. Run `notebooks/00` → `06` in order, or run `bash scripts/run_all.sh`.
   * Run `!ls /kaggle/input` and fix `roots` in `configs/default.yaml` if the folder names differ. You can also pass an override, e.g. `--set roots.train_root=...`.
   * Feature caches and checkpoints go to `/kaggle/working`. To split stages across sessions, use **Save Version** and attach the output as input to the next notebook.
3. Commit `data/manifests/manifest.csv` and `results/` back to the repo.

Approximate cost on a T4: manifest + audit ≈ 15 min; feature extraction ≈ 1 h for all backbones; all head experiments < 30 min; ResNet experiments ≈ 2–3 h; robustness ≈ 1 h.

### 2. Locally (CPU: tests and demo)

```bash
py -3.12 -m venv .venv && .venv\Scripts\activate          # Windows (Python 3.12 recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,demo]"
pytest                                                     # 46 tests, ~2 min, no downloads except ResNet-18 weights
python app/app.py --clip checkpoints/A2_head.pt --resnet checkpoints/B_model.pt
```

### Key commands

```bash
python -m fakelens.data.manifest                                   # build splits
python -m fakelens.analysis.audit                                  # bias audit + shortcut probe
python -m fakelens.features.extract --backbone clip_vitl14         # cache frozen features
python -m fakelens.train_head --model A2 --experiment p1           # train/eval the main model (3 seeds)
python -m fakelens.train_resnet --experiment p1                    # baseline (3 seeds)
python -m fakelens.robustness --ckpts runs/p1/A2/seed0/head.pt runs/p1/B/seed0/model.pt --names A2 B
python -m fakelens.evaluate aggregate --runs runs/p1 --out results/p1
python -m fakelens.analysis.plots --results results --runs runs
```

## Evaluation protocol

* **Metrics:**
  * accuracy, precision, recall, F1 and the confusion matrix (as in the proposal);
  * AUROC and AP;
  * real and fake accuracy reported separately. On unseen generators the typical failure is a *shifted threshold*: AUROC stays high while fake accuracy collapses.
* **Thresholds:** we report at 0.5 and at the threshold that maximises balanced accuracy on **validation**. The test set is never used for tuning.
* **Calibration:** temperature scaling is fitted on validation. We report ECE before and after, with reliability diagrams.
* **Statistics:**
  * 3 seeds for heads and ResNet, reported as mean ± std;
  * 1,000-sample bootstrap 95% CIs;
  * McNemar exact test for paired model comparisons.
* **Robustness:** exact parameters are written to `results/robustness/robustness_config.json`.

  | Condition | Settings |
  |---|---|
  | JPEG quality | 95 / 85 / 75 / 65 / 50 / 30 |
  | Resize scale | 0.5 / 0.75 / 1.25 / 1.5 |
  | Low resolution (shorter side) | 112 / 64 |
  | Gaussian blur σ | 1 / 3 / 5 |
  | Gaussian noise std | 0.01 / 0.03 / 0.05 |
  | Social-media chain | long side → 1080 px, then JPEG 80 |

* **Sanity checks:**
  * training on shuffled labels should give about 50%;
  * the metadata-only probe should be high on raw data;
  * the final-embedding consistency of the CLIP hooks is verified.

## Results

*Filled in from `results/p1/summary.md`, `results/robustness/`, `results/logo/` and `results/figures/` after the Kaggle runs.*

| Model | SD v1.4 acc | Unseen acc (mean of 6) | Unseen AUROC | ECE (after T) |
|---|---|---|---|---|
| A1 CLIP-L/14 linear | – | – | – | – |
| **A2 CLIP-L/14 multi-layer** | – | – | – | – |
| A3 CLIP-B/16 multi-layer | – | – | – | – |
| A4 DINOv2-L/14 multi-layer | – | – | – | – |
| B ResNet-18 | – | – | – | – |
| A2 without debias | – | – | – | – |
| B without debias | – | – | – | – |

## Limitations and responsible use

* **Not proof.** The detector makes mistakes, especially on generators, image types or edits it has not seen. It is an academic demonstration, not a verification system.
* **Dataset scope.**
  * GenImage fakes are ImageNet-class images from 2021–2023 generators. Newer generators (SDXL, FLUX, DALL·E 3, recent Midjourney) may behave differently; the in-the-wild set gives a first indication.
  * The reals are ImageNet photographs, so other domains (art, screenshots, faces, medical images) are untested.
* **Remaining biases.**
  * Re-encoding both classes at quality 96 removes the format shortcut, but reals end up double-compressed while fakes are compressed once.
  * BigGAN images (128 px) must be upscaled to 224, which adds interpolation traces that real images rarely have.
  * We discuss both effects in the report.
* **Licence.** Trained weights inherit the non-commercial terms of the training data.

## AI-assistance disclosure

Parts of the code scaffolding and documentation were drafted with an AI coding assistant (Claude Code). The team reviewed, tested and is responsible for all code, experiments and conclusions, as required by university policy.

## References

* Zhu et al. *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image.* NeurIPS 2023 Datasets & Benchmarks.
* Ojha, Li, Lee. *Towards Universal Fake Image Detectors that Generalize Across Generative Models* (UnivFD). CVPR 2023.
* Koutlis, Papadopoulos. *Leveraging Representations from Intermediate Encoder-blocks for Synthetic Image Detection* (RINE). ECCV 2024.
* Grommelt, Weiss, Pfreundt, Keuper. *Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets.* ECCV Workshops 2024.
* Wang et al. *CNN-generated images are surprisingly easy to spot… for now* (CNNDetection). CVPR 2020.
* Radford et al. *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML 2021.
* Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024.
* Guo et al. *On Calibration of Modern Neural Networks.* ICML 2017.
