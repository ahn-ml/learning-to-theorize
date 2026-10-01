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

GridWorld pretraining uses four GPUs, 512 examples per rank, 50 epochs,
learning rate 0.0025 and KL weight 1e-5. Arithmetic uses four GPUs and 500 epochs.

```bash
python scripts/pretrain.py --task gridworld --devices 0,1,2,3 \
  --data-root /path/to/gridworld-data --output-root /path/to/new-grid-pretraining
python scripts/pretrain.py --task arithmetic_factorization --devices 0,1,2,3 \
  --data-root /path/to/arithmetic-data --output-root /path/to/new-arithmetic-pretraining
```

For GridWorld, training must finish all 50 epochs. Pass the checkpoint identified
by `best_reconstruction.json` to NEO. Keep `config.json`, `status.json`,
`metrics.jsonl`, `best_reconstruction.json` and the `best_reconstruction/`
directory together. Selection uses the earliest maximum of globally aggregated
reconstruction accuracy.

For Arithmetic, use `checkpoints/checkpoint_final.pth` after 500 epochs. This is
a digit embedding and linear decoder trained only to reconstruct 10,000
individual numbers (0000–9999). Pretraining has no program network, transition
model, or VQ. The minimum LR ratio is 0.005. Reconstruction accuracy is measured
on the same observation vocabulary; it is not task transfer accuracy.

All three tasks can start from the provided observation checkpoints:

```bash
python scripts/download_checkpoints.py --task gridworld --output-root checkpoints
python scripts/download_checkpoints.py --task arithmetic_factorization --output-root checkpoints
python scripts/download_checkpoints.py --task image_editing --output-root checkpoints
```

Downloads are verified against `configs/checkpoints.json`. ImageEditing starts
from its provided VAE; its observation-pretraining data and loop are not included.
Only observation weights are distributed. Train NEO before evaluating it.

The Arithmetic weights currently provided with the release were trained with
the earlier pretraining procedure. They remain available for the recorded
results; use the checkpoint from the command above for observation-only pretraining.

## NEO training

Use the same command for each task; supply its data and observation checkpoint:

```bash
python scripts/train.py --task gridworld --method neo --alpha all --seed all \
  --devices 0 --data-root /path/to/gridworld-data --output-root /path/to/new-grid-training \
  --observation-checkpoint /path/to/checkpoints/gridworld/observation/best_reconstruction/checkpoint_2941.pt
python scripts/train.py --task arithmetic_factorization --method neo --alpha all --seed all \
  --devices 0 --data-root /path/to/arithmetic-data --output-root /path/to/new-arithmetic-training \
  --observation-checkpoint /path/to/checkpoints/arithmetic_factorization/observation/model.pth
python scripts/train.py --task image_editing --method neo --alpha all --seed all \
  --devices 0,1 --data-root /path/to/image-data --output-root /path/to/new-image-training \
  --observation-checkpoint /path/to/checkpoints/image_editing/observation/model.pth
```

Each command runs its nine conditions sequentially. To parallelize, invoke
individual alpha/seed conditions on separate devices and unique output roots.
ImageEditing retains two GPU ranks and batch size 64 per rank; one GPU is not
the same training schedule. GridWorld/Arithmetic NEO each use one rank.

Execution recipes are in `configs/<task>/reproduction.yaml`. Model and optimizer
settings are defined in `src/tasks/gridworld/theorizer_config.py`,
`src/tasks/arithmetic_factorization/experiment.py` and
`src/tasks/image_editing/experiment_config.py`. Each run saves its resolved settings.
Use `--config` for another execution recipe and `--dry-run` to inspect commands.

## Checkpoint selection and evaluation

```bash
python scripts/evaluate.py --task gridworld --method neo --seed all --devices 0 \
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
python scripts/evaluate.py --task gridworld --method neo \
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
