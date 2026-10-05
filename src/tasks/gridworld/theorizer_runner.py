"""Single-GPU runner for one GridWorld NEO alpha and seed."""

from __future__ import annotations

import argparse
import os
import platform
import socket
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from tasks.gridworld.data.dataset import (
    GridWorldHDF5Dataset,
    collate_gridworld_observation_batch,
    seed_observation_worker,
)
from models.neo import NEOOutput
from tasks.gridworld.observation_checkpoint import (
    inspect_observation_checkpoint,
    load_gridworld_observation_checkpoint,
)
from tasks.gridworld.theorizer_config import (
    GridWorldAlphaExperiment,
    GridWorldTheorizerTrainingConfig,
    available_theorizer_experiments,
    get_theorizer_experiment,
)
from tasks.gridworld.task import build_neo
from training.optimization import (
    OptimizationConfig,
    build_cosine_warmup_scheduler,
    build_program_optimizer,
)
from training.runtime import (
    DistributedContext,
    ExperimentTrackingConfig,
    GitSourceState,
    MetricLog,
    WandbMode,
    bfloat16_autocast,
    capture_git_source,
    create_run_directory,
    initialize_distributed,
    seed_everything,
    sha256_file,
    write_json_atomic,
    write_json_exclusive,
)


@dataclass(frozen=True, slots=True)
class GridWorldTheorizerRunConfig:
    """Executable settings for one alpha and model-seed run."""

    experiment: str
    seed: int
    train_h5: Path
    test_h5: Path
    observation_checkpoint: Path
    output_root: Path
    wandb_project: str = "LearningToTheorize"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_mode: WandbMode = "online"

    def __post_init__(self) -> None:
        get_theorizer_experiment(self.experiment)
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.wandb_project.strip():
            raise ValueError("wandb_project must not be empty")

    @property
    def paper_experiment(
        self,
    ) -> GridWorldAlphaExperiment:
        return get_theorizer_experiment(self.experiment)

    @property
    def run_name(self) -> str:
        return f"gridworld-{self.experiment}-seed-{self.seed}"

    @property
    def tracking_group(self) -> str:
        return self.wandb_group or f"gridworld-neo-{self.experiment}"


@dataclass(frozen=True, slots=True)
class GridWorldTheorizerTrainingState:
    """State at an epoch boundary."""

    completed_epochs: int
    global_step: int
    best_support_grid_accuracy: float


@dataclass(frozen=True, slots=True)
class GridWorldTheorizerRunResult:
    output_directory: Path
    state: GridWorldTheorizerTrainingState
    periodic_checkpoints: tuple[Path, ...]
    best_checkpoint: Path | None


@dataclass(slots=True)
class _TheorizerLoader:
    source_dataset: GridWorldHDF5Dataset
    dataloader: DataLoader[Tensor]
    sampler: DistributedSampler[tuple[Tensor, Tensor]]
    num_episodes: int
    iteration_count: int = 0

    def batches(self) -> Iterator[Tensor]:
        self.sampler.set_epoch(self.iteration_count)
        self.iteration_count += 1
        self.source_dataset.close()
        return iter(self.dataloader)

    def close(self) -> None:
        self.source_dataset.close()


class _ProcessSafeDataLoader(DataLoader[Tensor]):
    def __iter__(self) -> Any:
        source = getattr(self, "_source_dataset", None)
        if isinstance(source, GridWorldHDF5Dataset):
            source.close()
        return super().__iter__()


def build_theorizer_loader(
    path: str | Path,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
    pin_memory: bool,
    episode_limit: int | None = None,
) -> _TheorizerLoader:
    """Build a world-size-one DistributedSampler loader seeded by the run."""

    source = GridWorldHDF5Dataset(path)
    if episode_limit is not None and episode_limit > len(source):
        source.close()
        raise ValueError(
            f"episode_limit {episode_limit} exceeds artifact length {len(source)}"
        )
    num_episodes = len(source) if episode_limit is None else episode_limit
    dataset: Dataset[tuple[Tensor, Tensor]]
    if num_episodes == len(source):
        dataset = source
    else:
        dataset = Subset(source, range(num_episodes))
    # Use the experiment seed for DistributedSampler, including single-GPU runs.
    sampler = DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=shuffle,
        seed=seed,
        drop_last=False,
    )
    generator = torch.Generator().manual_seed(seed)
    dataloader = _ProcessSafeDataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_gridworld_observation_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_observation_worker,
        generator=generator,
        drop_last=False,
        persistent_workers=False,
    )
    dataloader._source_dataset = source  # type: ignore[attr-defined]
    return _TheorizerLoader(source, dataloader, sampler, num_episodes)


def build_theorizer_optimizer(model: nn.Module) -> AdamW:
    """Build programmer/executor/other AdamW parameter groups."""

    config = model.experiment.training
    return build_program_optimizer(
        model,
        learning_rate=config.learning_rate,
        programmer_learning_rate=config.policy_learning_rate,
        executor_learning_rate=config.transition_learning_rate,
        weight_decay=config.weight_decay,
    )


def build_theorizer_scheduler(
    optimizer: Optimizer,
    experiment: GridWorldAlphaExperiment,
) -> LambdaLR:
    training = experiment.training
    optimization = OptimizationConfig(
        total_steps=training.total_steps,
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        max_gradient_norm=training.gradient_clip_norm,
        warmup_ratio=training.warmup_ratio,
        minimum_learning_rate_ratio=training.minimum_learning_rate_ratio,
    )
    return build_cosine_warmup_scheduler(optimizer, optimization)


def run_gridworld_theorizer(
    config: GridWorldTheorizerRunConfig,
) -> GridWorldTheorizerRunResult:
    """Train one alpha/seed model, evaluating before each epoch."""

    context = initialize_distributed()
    training = config.paper_experiment.training
    train_loader: _TheorizerLoader | None = None
    test_loader: _TheorizerLoader | None = None
    metric_log: MetricLog | None = None
    output_directory: Path | None = None
    periodic_checkpoints: list[Path] = []
    best_checkpoint: Path | None = None
    state = GridWorldTheorizerTrainingState(0, 0, 0.0)
    succeeded = False
    try:
        if context.world_size != 1:
            raise ValueError("GridWorld NEO training runs in exactly one process")
        source = capture_git_source(Path(__file__).resolve().parent)
        artifacts = _prepare_artifact_evidence(config)
        output_directory = create_run_directory(
            config.output_root,
            config.run_name,
            source,
            context,
        )
        resolved = _resolved_config(config, context, source, artifacts)
        write_json_exclusive(output_directory / "config.json", resolved)
        write_json_exclusive(output_directory / "source.json", asdict(source))
        metric_log = MetricLog(
            output_directory / "metrics.jsonl",
            ExperimentTrackingConfig(
                project=config.wandb_project,
                entity=config.wandb_entity,
                group=config.tracking_group,
                name=f"{config.run_name}-{output_directory.name}",
                mode=config.wandb_mode,
                tags=(
                    "gridworld",
                    "theorizer",
                    "neo",
                    config.experiment,
                    f"seed-{config.seed}",
                ),
            ),
            resolved,
        )
        _write_status(output_directory, "running", state)

        # Initialize W&B, reset random seeds, construct the model,
        # then load pretrained observation weights.
        seed_everything(config.seed)
        model = build_neo(config.experiment)
        load_gridworld_observation_checkpoint(
            model,
            config.observation_checkpoint,
            expected_sha256=str(artifacts["observation"]["sha256"]),
        )
        model = model.to(context.device)

        train_loader = build_theorizer_loader(
            config.train_h5,
            batch_size=training.batch_size,
            num_workers=training.num_workers,
            seed=config.seed,
            shuffle=True,
            pin_memory=context.device.type == "cuda",
        )
        test_loader = build_theorizer_loader(
            config.test_h5,
            batch_size=training.batch_size,
            num_workers=training.num_workers,
            seed=config.seed,
            shuffle=False,
            pin_memory=context.device.type == "cuda",
        )
        _validate_loader_contract(training, train_loader, test_loader)

        optimizer = build_theorizer_optimizer(model)
        scheduler = build_theorizer_scheduler(optimizer, config.paper_experiment)

        for epoch in range(training.epochs):
            evaluation = _evaluate(model, test_loader, context.device)
            metric_log.log(
                _metric_record(
                    "eval",
                    evaluation,
                    epoch=epoch,
                    global_step=state.global_step,
                    learning_rate=scheduler.get_last_lr()[0],
                )
            )
            if (epoch + 1) % training.checkpoint_interval_epochs == 0:
                checkpoint_path = (
                    output_directory / "checkpoints" / f"checkpoint_{state.global_step + 1}.pth"
                )
                _save_checkpoint(
                    checkpoint_path,
                    model=model,
                    state=state,
                    evaluated_epoch=epoch,
                    resolved_config=resolved,
                )
                periodic_checkpoints.append(checkpoint_path)

            support_grid_accuracy = evaluation["support_grid_accuracy"]
            if support_grid_accuracy >= state.best_support_grid_accuracy:
                state = GridWorldTheorizerTrainingState(
                    completed_epochs=state.completed_epochs,
                    global_step=state.global_step,
                    best_support_grid_accuracy=support_grid_accuracy,
                )
                best_checkpoint = output_directory / "checkpoints" / "best_model.pth"
                _save_checkpoint(
                    best_checkpoint,
                    model=model,
                    state=state,
                    evaluated_epoch=epoch,
                    resolved_config=resolved,
                    overwrite=True,
                )

            model.train()
            for batch in train_loader.batches():
                episode_grids = batch.to(context.device, non_blocking=True).long()
                optimizer.zero_grad(set_to_none=True)
                with bfloat16_autocast(context.device):
                    output = model(episode_grids, is_eval=False)
                output.loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    model.parameters(),
                    training.gradient_clip_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                model.quantizer.step()
                state = GridWorldTheorizerTrainingState(
                    completed_epochs=state.completed_epochs,
                    global_step=state.global_step + 1,
                    best_support_grid_accuracy=state.best_support_grid_accuracy,
                )
                if state.global_step % training.log_interval_steps == 0:
                    record = _output_values(output)
                    record["gradient_norm"] = float(gradient_norm.detach().float().item())
                    metric_log.log(
                        _metric_record(
                            "train",
                            record,
                            epoch=epoch,
                            global_step=state.global_step,
                            learning_rate=scheduler.get_last_lr()[0],
                        )
                    )
                    _write_status(output_directory, "running", state)

            state = GridWorldTheorizerTrainingState(
                completed_epochs=epoch + 1,
                global_step=state.global_step,
                best_support_grid_accuracy=state.best_support_grid_accuracy,
            )

        _write_status(
            output_directory,
            "completed",
            state,
            best_checkpoint=str(best_checkpoint) if best_checkpoint else None,
            periodic_checkpoints=[str(path) for path in periodic_checkpoints],
        )
        succeeded = True
        return GridWorldTheorizerRunResult(
            output_directory=output_directory,
            state=state,
            periodic_checkpoints=tuple(periodic_checkpoints),
            best_checkpoint=best_checkpoint,
        )
    except BaseException as error:
        if output_directory is not None:
            _write_status(
                output_directory,
                "failed",
                state,
                error=f"{type(error).__name__}: {error}",
            )
        raise
    finally:
        if train_loader is not None:
            train_loader.close()
        if test_loader is not None:
            test_loader.close()
        if metric_log is not None:
            metric_log.finish(exit_code=0 if succeeded else 1)


def _prepare_artifact_evidence(
    config: GridWorldTheorizerRunConfig,
) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for name, value in (("train", config.train_h5), ("test", config.test_h5)):
        path = Path(value).expanduser().resolve(strict=True)
        evidence[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    observation = Path(config.observation_checkpoint).expanduser().resolve(strict=True)
    evidence["observation"] = {
        "path": str(observation),
        "bytes": observation.stat().st_size,
        "sha256": inspect_observation_checkpoint(observation),
    }
    return evidence


def _validate_loader_contract(
    training: GridWorldTheorizerTrainingConfig,
    train: _TheorizerLoader,
    test: _TheorizerLoader,
) -> None:
    if train.num_episodes != training.train_episodes:
        raise ValueError(
            f"expected {training.train_episodes} training episodes, got {train.num_episodes}"
        )
    if test.num_episodes != training.test_episodes:
        raise ValueError(
            f"expected {training.test_episodes} test episodes, got {test.num_episodes}"
        )
    if len(train.dataloader) != training.train_batches_per_epoch:
        raise ValueError("training loader batch count does not match the experiment")
    if len(test.dataloader) != training.test_batches_per_epoch:
        raise ValueError("test loader batch count does not match the experiment")


def _evaluate(
    model: nn.Module,
    loader: _TheorizerLoader,
    device: torch.device,
) -> dict[str, float]:
    """Average each metric over evaluation batches."""

    model.eval()
    sums: dict[str, float] = {}
    batch_count = 0
    with torch.no_grad():
        for batch in loader.batches():
            episode_grids = batch.to(device, non_blocking=True).long()
            with bfloat16_autocast(device):
                output = model(episode_grids, is_eval=True)
            for name, value in _output_values(output).items():
                sums[name] = sums.get(name, 0.0) + value
            batch_count += 1
    if not batch_count:
        raise RuntimeError("evaluation loader produced no batches")
    return {name: value / batch_count for name, value in sums.items()}


def _output_values(
    output: NEOOutput,
) -> dict[str, float]:
    values = {
        "loss": float(output.loss.detach().float().item()),
        "reconstruction_loss": float(output.reconstruction_loss.detach().float().item()),
        "support_pixel_accuracy": output.accuracy.pixel,
        "support_grid_accuracy": output.accuracy.grid,
        "support_macro_f1": output.accuracy.macro_f1,
    }
    values.update(
        auto_reconstruction_loss=float(output.auto_reconstruction_loss.detach().float().item()),
        auto_pixel_accuracy=output.auto_reconstruction_accuracy.pixel,
        auto_grid_accuracy=output.auto_reconstruction_accuracy.grid,
        transition_consistency_loss=float(output.transition_consistency_loss.detach().float().item()),
        action_vq_loss=float(output.action_vq_loss.detach().float().item()),
        mean_explanation_length=float(output.mean_explanation_length.detach().float().item()),
    )
    if output.query_reconstruction_loss is not None and output.query_accuracy is not None:
        values.update(
            query_reconstruction_loss=float(output.query_reconstruction_loss.detach().float().item()),
            query_pixel_accuracy=output.query_accuracy.pixel,
            query_grid_accuracy=output.query_accuracy.grid,
            query_macro_f1=output.query_accuracy.macro_f1,
        )
    for metrics in output.per_length_metrics:
        prefix = f"length_{metrics.length}"
        values[f"{prefix}_count"] = float(metrics.count)
        values[f"{prefix}_reconstruction_loss"] = float(
            metrics.reconstruction_loss.detach().float().item()
        )
        values[f"{prefix}_pixel_accuracy"] = metrics.accuracy.pixel
        values[f"{prefix}_grid_accuracy"] = metrics.accuracy.grid
        values[f"{prefix}_macro_f1"] = metrics.accuracy.macro_f1
    for length, quantized in enumerate(output.action_vq_by_length, start=1):
        prefix = f"action_vq_length_{length}"
        values[f"{prefix}_loss"] = float(quantized.loss.detach().float().item())
        values[f"{prefix}_commitment_loss"] = float(
            quantized.commitment_loss.detach().float().item()
        )
        values[f"{prefix}_codebook_loss"] = float(
            quantized.codebook_loss.detach().float().item()
        )
        values[f"{prefix}_temperature"] = float(quantized.temperature.item())
        values[f"{prefix}_utilization"] = (
            quantized.indices.unique().numel()
            / output.action_vq_by_length[0].logits.shape[-1]
        )
    return values


def _metric_record(
    split: str,
    values: Mapping[str, float],
    *,
    epoch: int,
    global_step: int,
    learning_rate: float,
) -> dict[str, int | float]:
    record: dict[str, int | float] = {
        "epoch": epoch + 1,
        "global_step": global_step,
        "learning_rate": learning_rate,
    }
    record.update({f"{split}/{key}": value for key, value in values.items()})
    return record


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    state: GridWorldTheorizerTrainingState,
    evaluated_epoch: int,
    resolved_config: Mapping[str, Any],
    overwrite: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    payload = {
        "format_version": 1,
        "epoch": evaluated_epoch,
        "global_step": state.global_step,
        "model_state_dict": model.state_dict(),
        "args": dict(resolved_config),
    }
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("xb") as file:
        torch.save(payload, file)
    temporary.replace(path)


def _resolved_config(
    config: GridWorldTheorizerRunConfig,
    context: DistributedContext,
    source: GitSourceState,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "domain": "gridworld",
        "stage": "theorizer_training",
        "method": "neo",
        "experiment": config.experiment,
        "seed": config.seed,
        "training": asdict(config.paper_experiment.training),
        "length_control_coefficient": config.paper_experiment.length_control_coefficient,
        "runtime": {
            "world_size": context.world_size,
            "precision": "bf16-mixed" if context.device.type == "cuda" else "float32",
            "sampler": {
                "implementation": "DistributedSampler",
                "num_replicas": 1,
                "rank": 0,
                "seed": config.seed,
                "drop_last": False,
                "train_shuffle": True,
                "test_shuffle": False,
                "epoch_source": "loader_iteration_count",
            },
            "evaluation_aggregation_for_best": "unweighted_batch_mean",
        },
        "tracking": {
            "project": config.wandb_project,
            "entity": config.wandb_entity,
            "group": config.tracking_group,
            "mode": config.wandb_mode,
        },
        "artifacts": dict(artifacts),
        "source": asdict(source),
        "environment": {
            "hostname": socket.gethostname(),
            "python_pid": os.getpid(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "hdf5": h5py.version.hdf5_version,
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "visible_cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
    }


def _write_status(
    output_directory: Path,
    status: str,
    state: GridWorldTheorizerTrainingState,
    **extra: Any,
) -> None:
    write_json_atomic(
        output_directory / "status.json",
        {
            "status": status,
            **asdict(state),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train NEO for one GridWorld alpha and seed")
    parser.add_argument("--experiment", choices=available_theorizer_experiments(), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-h5", type=Path, required=True)
    parser.add_argument("--test-h5", type=Path, required=True)
    parser.add_argument("--observation-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wandb-project", default="LearningToTheorize")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    return parser


def _config_from_arguments(arguments: argparse.Namespace) -> GridWorldTheorizerRunConfig:
    return GridWorldTheorizerRunConfig(
        experiment=arguments.experiment,
        seed=arguments.seed,
        train_h5=arguments.train_h5,
        test_h5=arguments.test_h5,
        observation_checkpoint=arguments.observation_checkpoint,
        output_root=arguments.output_root,
        wandb_project=arguments.wandb_project,
        wandb_entity=arguments.wandb_entity,
        wandb_group=arguments.wandb_group,
        wandb_mode=arguments.wandb_mode,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    try:
        result = run_gridworld_theorizer(_config_from_arguments(parser.parse_args(argv)))
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(result.output_directory)


if __name__ == "__main__":
    main()
