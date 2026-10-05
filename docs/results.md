# Reproduction results

Means ± sample standard deviations over seeds 42, 43, 44. ID is selection data.
Accuracy is percent (higher is better); ImageEditing L1 is lower-is-better.
Hard grounding is ON for GridWorld and Arithmetic, and OFF for ImageEditing,
in both checkpoint selection and final evaluation. Paper baselines were not rerun.

[Training and evaluation](reproduction.md) · [Per-seed results](../configs/results.json)

## GridWorld

| Alpha | Model | Source | ID | Comp OOD | Length OOD |
|---|---|---|---:|---:|---:|
| 0.33 | Disc-Mono | Paper | 98.3 | 0 | 0 |
| 0.33 | Cont-Mono | Paper | 0.1 | 0 | 0 |
| 0.33 | Cont-Mono-Opt | Paper | 0 | 0 | 0 |
| 0.33 | NEO | Paper | 91.1 | 93.3 | 84.5 |
| 0.33 | NEO | Reproduction | 96.85 ± 2.90 | 95.48 ± 4.73 | 88.76 ± 9.75 |
| 0.33 | NEO-S | Paper | 97 | 97.6 | 90.7 |
| 0.33 | NEO-S | Reproduction | 97.73 ± 1.88 | 97.05 ± 3.25 | 93.57 ± 4.85 |
| 0.66 | Disc-Mono | Paper | 97.3 | 0 | 0 |
| 0.66 | Cont-Mono | Paper | 0 | 0 | 0 |
| 0.66 | Cont-Mono-Opt | Paper | 0 | 0 | 0.1 |
| 0.66 | NEO | Paper | 96.5 | 96.3 | 92.7 |
| 0.66 | NEO | Reproduction | 94.50 ± 1.70 | 93.77 ± 2.10 | 84.44 ± 3.10 |
| 0.66 | NEO-S | Paper | 98.7 | 98.7 | 94.9 |
| 0.66 | NEO-S | Reproduction | 95.94 ± 1.27 | 96.15 ± 1.77 | 89.15 ± 3.00 |
| 1.00 | Disc-Mono | Paper | 92.1 | — | 0 |
| 1.00 | Cont-Mono | Paper | 0 | — | 0 |
| 1.00 | Cont-Mono-Opt | Paper | 0 | — | 0.1 |
| 1.00 | NEO | Paper | 94.9 | — | 89.8 |
| 1.00 | NEO | Reproduction | 93.66 ± 2.89 | — | 80.94 ± 9.79 |
| 1.00 | NEO-S | Paper | 97.5 | — | 92.6 |
| 1.00 | NEO-S | Reproduction | 95.87 ± 1.74 | — | 88.90 ± 5.69 |

## Arithmetic

| Alpha | Model | Source | ID | Comp OOD | Length OOD |
|---|---|---|---:|---:|---:|
| 0.33 | Disc-Mono | Paper | 66.8 | 0.4 | 0.9 |
| 0.33 | Cont-Mono | Paper | 0.1 | 0 | 0 |
| 0.33 | Cont-Mono-Opt | Paper | 0.1 | 0 | 0 |
| 0.33 | NEO | Paper | 79.2 | 34.5 | 3.8 |
| 0.33 | NEO | Reproduction | 80.01 ± 9.48 | 51.89 ± 12.76 | 29.68 ± 7.07 |
| 0.33 | NEO-S | Paper | 80.9 | 75.9 | 52.4 |
| 0.33 | NEO-S | Reproduction | 90.45 ± 8.17 | 80.52 ± 20.31 | 77.17 ± 14.32 |
| 0.66 | Disc-Mono | Paper | 57.2 | 0.2 | 0.4 |
| 0.66 | Cont-Mono | Paper | 0.2 | 0 | 0.1 |
| 0.66 | Cont-Mono-Opt | Paper | 0.2 | 0 | 0.1 |
| 0.66 | NEO | Paper | 73.1 | 57.3 | 1.9 |
| 0.66 | NEO | Reproduction | 77.49 ± 10.38 | 71.23 ± 7.36 | 29.17 ± 6.36 |
| 0.66 | NEO-S | Paper | 93.9 | 95.9 | 69.6 |
| 0.66 | NEO-S | Reproduction | 93.85 ± 2.80 | 92.95 ± 3.24 | 75.60 ± 13.37 |
| 1.00 | Disc-Mono | Paper | 47.5 | — | 0.4 |
| 1.00 | Cont-Mono | Paper | 0 | — | 0 |
| 1.00 | Cont-Mono-Opt | Paper | 0 | — | 0 |
| 1.00 | NEO | Paper | 67.5 | — | 2.3 |
| 1.00 | NEO | Reproduction | 74.09 ± 2.10 | — | 35.21 ± 0.62 |
| 1.00 | NEO-S | Paper | 95.4 | — | 70.7 |
| 1.00 | NEO-S | Reproduction | 94.49 ± 3.06 | — | 82.27 ± 6.13 |

## ImageEditing

| Alpha | Model | Source | ID | Comp OOD | Length OOD |
|---|---|---|---:|---:|---:|
| 0.33 | Disc-Mono | Paper | 0.06 | 0.17 | 0.19 |
| 0.33 | Cont-Mono | Paper | 0.08 | 0.17 | 0.19 |
| 0.33 | Cont-Mono-Opt | Paper | 0.08 | 0.16 | 0.19 |
| 0.33 | NEO | Paper | 0.07 | 0.12 | 0.13 |
| 0.33 | NEO | Reproduction | 0.0689 ± 0.0021 | 0.1272 ± 0.0056 | 0.1443 ± 0.0087 |
| 0.66 | Disc-Mono | Paper | 0.07 | 0.15 | 0.17 |
| 0.66 | Cont-Mono | Paper | 0.13 | 0.18 | 0.21 |
| 0.66 | Cont-Mono-Opt | Paper | 0.13 | 0.18 | 0.21 |
| 0.66 | NEO | Paper | 0.07 | 0.09 | 0.11 |
| 0.66 | NEO | Reproduction | 0.0683 ± 0.0009 | 0.0957 ± 0.0045 | 0.1134 ± 0.0026 |
| 1.00 | Disc-Mono | Paper | 0.08 | — | 0.17 |
| 1.00 | Cont-Mono | Paper | 0.12 | — | 0.18 |
| 1.00 | Cont-Mono-Opt | Paper | 0.12 | — | 0.18 |
| 1.00 | NEO | Paper | 0.07 | — | 0.1 |
| 1.00 | NEO | Reproduction | 0.0777 ± 0.0018 | — | 0.1111 ± 0.0031 |

NEO-S uses the same selected NEO checkpoints. Each task's three downstream
seeds share one pretrained observation model.

Changes since `reproduction-20260929`: GridWorld and Arithmetic observation
models are now pretrained to reconstruct single observations, with new
checkpoints, and Arithmetic NEO uses the paper's codebook update, grounding
normalization and no orthogonal regularization, with action codes sampled during
training. The pretraining settings and Arithmetic's code sampling were chosen by
mean ID selection score over these seeds. Per-seed scores are recorded in [results.json](../configs/results.json).

All reported runs used PyTorch 2.7.1 / CUDA 12.6.

OOD results were inspected during development; these runs are not an untouched
confirmatory test.
