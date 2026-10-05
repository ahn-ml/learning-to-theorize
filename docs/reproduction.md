# Reproduction

Install the [shared environment](../environments/README.md) and run commands
from the repository root.

## Data

The public generator produces task HDF5 files. It does not overwrite existing
datasets. For ImageEditing, first place the CIFAR-10 Python distribution's
`cifar-10-batches-py` directory inside the data root.

```bash
python scripts/generate_data.py --task gridworld --data-root /path/to/gridworld-data
python scripts/generate_data.py --task arithmetic_factorization --data-root /path/to/arithmetic-data
python scripts/generate_data.py --task image_editing --data-root /path/to/image-data
```

## Observation pretraining

Both observation models learn to reconstruct single observations; unlike the
paper's runs, pretraining has no transition or grounding objective. GridWorld
trains the VAE on individually shuffled grids with four GPUs, 2048 grids per rank
per step (the grids of 512 episodes), 50 epochs, learning rate 0.0007 and KL
weight 1e-5. Arithmetic reads the same paired episodes as before but trains the
digit autoencoder with reconstruction only, on four GPUs for 500 epochs.

The GridWorld pretraining grids come from the data generator. The Arithmetic
pretraining episodes (`arith_lv1_10000_samples_train.h5` and
`arith_lv1_1000_samples_test.h5`) are not produced by the generator; download
them with the observation checkpoints (below) and use their directory as the
data root:

```bash
python scripts/pretrain.py --task gridworld --devices 0,1,2,3 \
  --data-root /path/to/gridworld-data --output-root /path/to/new-grid-pretraining
python scripts/pretrain.py --task arithmetic_factorization --devices 0,1,2,3 \
  --data-root checkpoints/arithmetic_factorization/pretraining-data \
  --output-root /path/to/new-arithmetic-pretraining
```

For GridWorld, training must finish all 50 epochs. Pass the checkpoint in
`best_reconstruction/` (named in `best_reconstruction.json`) to NEO; it is the
earliest epoch with the highest globally aggregated reconstruction accuracy.

For Arithmetic, use `checkpoints/checkpoint_final.pth` after 500 epochs. This is
a deterministic digit-embedding model with learned position embeddings; it is not
a VAE. The pretraining loss is digit reconstruction only (no grounding term), and
the minimum LR ratio is 0.005.

All three tasks can start from the provided observation checkpoints:

```bash
python scripts/download_checkpoints.py --task gridworld --output-root checkpoints
python scripts/download_checkpoints.py --task arithmetic_factorization --output-root checkpoints
python scripts/download_checkpoints.py --task image_editing --output-root checkpoints
```

Downloads are verified against `configs/checkpoints.json`. The Arithmetic download
also places the pretraining episodes in
`checkpoints/arithmetic_factorization/pretraining-data/`. ImageEditing starts from
its provided VAE; its observation-pretraining data and loop are not included.
Only observation weights are distributed. Train NEO before evaluating it.

## NEO training

Use the same command for each task; supply its data and observation checkpoint:

```bash
python scripts/train.py --task gridworld --alpha all --seed all \
  --devices 0 --data-root /path/to/gridworld-data --output-root /path/to/new-grid-training \
  --observation-checkpoint /path/to/checkpoints/gridworld/observation/best_reconstruction/checkpoint_2206.pt
python scripts/train.py --task arithmetic_factorization --alpha all --seed all \
  --devices 0 --data-root /path/to/arithmetic-data --output-root /path/to/new-arithmetic-training \
  --observation-checkpoint /path/to/checkpoints/arithmetic_factorization/observation/model.pth
python scripts/train.py --task image_editing --alpha all --seed all \
  --devices 0,1 --data-root /path/to/image-data --output-root /path/to/new-image-training \
  --observation-checkpoint /path/to/checkpoints/image_editing/observation/model.pth
```

Each command runs its nine conditions sequentially. To parallelize, invoke
individual alpha/seed conditions on separate devices and unique output roots.
ImageEditing retains two GPU ranks and batch size 64 per rank; one GPU is not
the same training schedule. GridWorld/Arithmetic NEO each use one rank.

Arithmetic NEO updates its EMA codebook at every quantization, divides each
episode's grounding by its program length and uses no orthogonal regularization,
as in the paper's runs. It also samples action codes during training (temperature
0.3 to 0.05 over the first quarter of steps) and uses the nearest code at
evaluation.

Model and optimizer settings are defined in
`src/tasks/gridworld/theorizer_config.py`,
`src/tasks/arithmetic_factorization/experiment.py` and
`src/tasks/image_editing/experiment_config.py`. Each run saves its resolved
settings. Use `--dry-run` to inspect the commands a script would run.

## Checkpoint selection and evaluation

```bash
python scripts/evaluate.py --task gridworld --seed all --devices 0 \
  --data-root /path/to/gridworld-data --training-root /path/to/new-grid-training \
  --output-root /path/to/new-grid-evaluation
```

Replace the task and paths for Arithmetic or ImageEditing. Selection evaluates
all saved candidates on full ID query transfer. Hard grounding is ON for
GridWorld and Arithmetic, and OFF for ImageEditing. Each task uses the same
setting for checkpoint selection and final evaluation.
Exact score ties choose the latest optimizer step. No OOD data enter that
selection. For a single run, `--checkpoint-directory` with explicit `--alpha`
and `--seed` avoids ambiguity between multiple training directories.

For NEO-S, reuse the selected NEO checkpoint:

```bash
python scripts/evaluate.py --task gridworld \
  --alpha 0.33 --seed 42 --devices 0 --protocol test-time-scaling \
  --data-root data/gridworld \
  --selection-record runs/gridworld/evaluate/selection/gridworld-alpha0.33-seed42/selected.json \
  --output-root runs/gridworld/scaling
```

Do not select a different checkpoint for NEO-S. GridWorld uses budgets
1/4/16/64 at temperature 0.3; Arithmetic uses its fixed budgets through 1024 at
temperature 1.0. Report selected-program transfer, not oracle pass rate.
ImageEditing has only the standard evaluation.

GridWorld ID/Comp contain 5000 episodes each and Length contains 10000.
Arithmetic selection uses all ID episodes; the final reported metrics use the
first 5000 episodes of each split. ImageEditing uses the complete task splits.
ID is selection data. There is no compositional split at alpha 1.00.

## Comparison values

This repository implements NEO and NEO-S. Baseline values in the comparison
tables are quoted from the paper; baseline implementations are not included.
