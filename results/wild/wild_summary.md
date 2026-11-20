### Condition: clean

Accuracy at threshold 0.5 (reals: share called *real*; fakes: share called *AI-generated*). AUROC per generator = its fakes vs all reals.

| Subset | n | A2-all8 acc | A2 acc | B acc | A2-all8 AUROC | A2 AUROC | B AUROC |
|---|---|---|---|---|---|---|---|
| **All images** | 250 | 0.58 | 0.56 | 0.52 | 0.63 | 0.60 | 0.49 |
| fake: flux2-klein | 25 | 0.36 | 0.40 | 0.00 | 0.54 | 0.66 | 0.12 |
| fake: gpt-image-1.5 | 25 | 0.36 | 0.24 | 0.20 | 0.50 | 0.47 | 0.42 |
| fake: illustrious-sdxl | 25 | 0.80 | 0.40 | 0.68 | 0.77 | 0.70 | 0.78 |
| fake: midjourney-v7 | 25 | 0.80 | 0.40 | 0.24 | 0.76 | 0.66 | 0.52 |
| fake: seedream-v5 | 25 | 0.12 | 0.12 | 0.12 | 0.31 | 0.29 | 0.45 |
| fake: z-image-turbo | 25 | 0.96 | 0.72 | 0.48 | 0.89 | 0.84 | 0.65 |
| real: camera | 35 | 0.20 | 0.54 | 0.83 | – | – | – |
| real: whatsapp | 65 | 0.80 | 1.00 | 0.88 | – | – | – |

### Condition: social

Accuracy at threshold 0.5 (reals: share called *real*; fakes: share called *AI-generated*). AUROC per generator = its fakes vs all reals.

| Subset | n | A2-all8 acc | A2 acc | B acc | A2-all8 AUROC | A2 AUROC | B AUROC |
|---|---|---|---|---|---|---|---|
| **All images** | 250 | 0.41 | 0.46 | 0.52 | 0.46 | 0.55 | 0.53 |
| fake: flux2-klein | 25 | 0.00 | 0.04 | 0.00 | 0.39 | 0.59 | 0.25 |
| fake: gpt-image-1.5 | 25 | 0.00 | 0.04 | 0.12 | 0.24 | 0.33 | 0.42 |
| fake: illustrious-sdxl | 25 | 0.00 | 0.16 | 0.56 | 0.64 | 0.72 | 0.75 |
| fake: midjourney-v7 | 25 | 0.04 | 0.08 | 0.28 | 0.47 | 0.59 | 0.60 |
| fake: seedream-v5 | 25 | 0.00 | 0.00 | 0.20 | 0.27 | 0.28 | 0.51 |
| fake: z-image-turbo | 25 | 0.08 | 0.36 | 0.36 | 0.75 | 0.78 | 0.66 |
| real: camera | 35 | 0.97 | 0.94 | 0.83 | – | – | – |
| real: whatsapp | 65 | 1.00 | 1.00 | 0.97 | – | – | – |
