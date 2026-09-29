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
| 0.33 | NEO | Reproduction | 96.04 ± 4.16 | 95.17 ± 3.82 | 88.08 ± 9.73 |
| 0.33 | NEO-S | Paper | 97 | 97.6 | 90.7 |
| 0.33 | NEO-S | Reproduction | 96.97 ± 3.05 | 96.55 ± 3.02 | 90.09 ± 8.46 |
| 0.66 | Disc-Mono | Paper | 97.3 | 0 | 0 |
| 0.66 | Cont-Mono | Paper | 0 | 0 | 0 |
| 0.66 | Cont-Mono-Opt | Paper | 0 | 0 | 0.1 |
| 0.66 | NEO | Paper | 96.5 | 96.3 | 92.7 |
| 0.66 | NEO | Reproduction | 94.31 ± 5.06 | 90.58 ± 9.16 | 83.72 ± 12.33 |
| 0.66 | NEO-S | Paper | 98.7 | 98.7 | 94.9 |
| 0.66 | NEO-S | Reproduction | 95.79 ± 3.96 | 93.41 ± 7.47 | 87.96 ± 10.10 |
| 1.00 | Disc-Mono | Paper | 92.1 | — | 0 |
| 1.00 | Cont-Mono | Paper | 0 | — | 0 |
| 1.00 | Cont-Mono-Opt | Paper | 0 | — | 0.1 |
| 1.00 | NEO | Paper | 94.9 | — | 89.8 |
| 1.00 | NEO | Reproduction | 96.27 ± 2.29 | — | 87.53 ± 6.18 |
| 1.00 | NEO-S | Paper | 97.5 | — | 92.6 |
| 1.00 | NEO-S | Reproduction | 97.63 ± 1.31 | — | 91.74 ± 4.13 |

## Arithmetic

| Alpha | Model | Source | ID | Comp OOD | Length OOD |
|---|---|---|---:|---:|---:|
| 0.33 | Disc-Mono | Paper | 66.8 | 0.4 | 0.9 |
| 0.33 | Cont-Mono | Paper | 0.1 | 0 | 0 |
| 0.33 | Cont-Mono-Opt | Paper | 0.1 | 0 | 0 |
| 0.33 | NEO | Paper | 79.2 | 34.5 | 3.8 |
| 0.33 | NEO | Reproduction | 66.97 ± 5.93 | 17.23 ± 18.42 | 14.85 ± 4.33 |
| 0.33 | NEO-S | Paper | 80.9 | 75.9 | 52.4 |
| 0.33 | NEO-S | Reproduction | 83.62 ± 11.65 | 55.15 ± 35.92 | 64.05 ± 18.13 |
| 0.66 | Disc-Mono | Paper | 57.2 | 0.2 | 0.4 |
| 0.66 | Cont-Mono | Paper | 0.2 | 0 | 0.1 |
| 0.66 | Cont-Mono-Opt | Paper | 0.2 | 0 | 0.1 |
| 0.66 | NEO | Paper | 73.1 | 57.3 | 1.9 |
| 0.66 | NEO | Reproduction | 67.93 ± 4.71 | 54.08 ± 15.18 | 17.39 ± 5.67 |
| 0.66 | NEO-S | Paper | 93.9 | 95.9 | 69.6 |
| 0.66 | NEO-S | Reproduction | 90.54 ± 8.70 | 86.41 ± 18.64 | 68.75 ± 8.41 |
| 1.00 | Disc-Mono | Paper | 47.5 | — | 0.4 |
| 1.00 | Cont-Mono | Paper | 0 | — | 0 |
| 1.00 | Cont-Mono-Opt | Paper | 0 | — | 0 |
| 1.00 | NEO | Paper | 67.5 | — | 2.3 |
| 1.00 | NEO | Reproduction | 74.81 ± 8.28 | — | 32.17 ± 7.75 |
| 1.00 | NEO-S | Paper | 95.4 | — | 70.7 |
| 1.00 | NEO-S | Reproduction | 96.00 ± 2.13 | — | 81.31 ± 3.39 |

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
seeds share one pretrained observation model. Source revisions and per-seed scores are recorded in [results.json](../configs/results.json).

The reported GridWorld and ImageEditing runs used PyTorch 2.7.1 / CUDA 12.6;
Arithmetic used PyTorch 2.9.1 / CUDA 12.6.

OOD results were inspected during development; these runs are not an untouched
confirmatory test.
