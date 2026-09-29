"""Single-GPU runner for the paper GridWorld theorizer alpha sweep."""

from __future__ import annotations

import argparse
import os
import platform
import socket
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

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
from models.neo import (
    NEO,
    NEOOutput,
)
from tasks.gridworld.observation_checkpoint import (
    OBSERVATION_SELECTION_POLICIES,
    inspect_observation_checkpoint,
    load_gridworld_observation_checkpoint,
)
from tasks.gridworld.theorizer_config import (
    GRIDWORLD_THEORIZER_SEEDS,
    GridWorldAlphaExperiment,
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


PAPER_THEORIZER_CHECKPOINT_POSITION = "after_eval_before_train"
GridWorldTrainingMethod = Literal["neo"]
_TRAINING_METHODS: tuple[GridWorldTrainingMethod, ...] = (
    "neo",
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
    method: GridWorldTrainingMethod = "neo"
    observation_selection: str = "recorded-final"
    run_name: str | None = None
    wandb_project: str = "LearningToTheorize"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_mode: WandbMode = "online"
    canonical: bool = True
    force_cpu: bool = False
    max_train_steps: int | None = None
    development_epochs: int | None = None
    development_train_episodes: int | None = None
    development_test_episodes: int | None = None
    development_batch_size: int | None = None
    development_num_workers: int | None = None
    development_save_interval: int | None = None
    development_log_interval: int | None = None
    resume_checkpoint: Path | None = None

    def __post_init__(self) -> None:
        if self.observation_selection not in OBSERVATION_SELECTION_POLICIES:
            raise ValueError("unknown observation selection policy")
        if self.method == "neo":
            get_theorizer_experiment(self.experiment)
        else:
            raise ValueError(f"unknown GridWorld training method: {self.method}")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.wandb_mode not in ("online", "offline", "disabled"):
            raise ValueError("wandb_mode must be online, offline, or disabled")
        if not self.wandb_project.strip():
            raise ValueError("wandb_project must not be empty")
        for name in (
            "max_train_steps",
            "development_epochs",
            "development_train_episodes",
            "development_test_episodes",
            "development_batch_size",
            "development_save_interval",
            "development_log_interval",
        ):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive when provided")
        if self.development_num_workers is not None and self.development_num_workers < 0:
            raise ValueError("development_num_workers must be non-negative")

    @property
    def paper_experiment(
        self,
    ) -> GridWorldAlphaExperiment:
        return get_theorizer_experiment(self.experiment)

    @property
    def resolved_run_name(self) -> str:
        if self.run_name is not None:
            return self.run_name
        return f"gridworld-{self.experiment}-seed-{self.seed}"

    @property
    def epochs(self) -> int:
        if not self.canonical and self.development_epochs is not None:
            return self.development_epochs
        return self.paper_experiment.training.epochs

    @property
    def batch_size(self) -> int:
        if not self.canonical and self.development_batch_size is not None:
            return self.development_batch_size
        return self.paper_experiment.training.batch_size

    @property
    def num_workers(self) -> int:
        if not self.canonical and self.development_num_workers is not None:
            return self.development_num_workers
        return self.paper_experiment.training.num_workers

    @property
    def save_interval(self) -> int:
        if not self.canonical and self.development_save_interval is not None:
            return self.development_save_interval
        return self.paper_experiment.training.checkpoint_interval_epochs

    @property
    def log_interval(self) -> int:
        if not self.canonical and self.development_log_interval is not None:
            return self.development_log_interval
        return self.paper_experiment.training.log_interval_steps


@dataclass(frozen=True, slots=True)
class GridWorldTheorizerTrainingState:
    """State at an epoch boundary or after a bounded development run."""

    completed_epochs: int
    global_step: int
    best_support_grid_accuracy: float


@dataclass(frozen=True, slots=True)
class GridWorldTheorizerRunResult:
    output_directory: Path
    state: GridWorldTheorizerTrainingState
    stopped_early: bool
    periodic_checkpoints: tuple[Path, ...]
    best_checkpoint: Path | None


@dataclass(slots=True)
class _TheorizerLoader:
    source_dataset: GridWorldHDF5Dataset
    dataloader: DataLoader[Tensor]
    sampler: DistributedSampler[tuple[Tensor, Tensor]]
    generator: torch.Generator
    num_episodes: int
    iteration_count: int = 0

    def batches(self) -> Iterator[Tensor]:
        self.sampler.set_epoch(self.iteration_count)
        self.iteration_count += 1
        self.source_dataset.close()
        return iter(self.dataloader)

    def set_iteration_count(self, count: int) -> None:
        if count < 0:
            raise ValueError("loader iteration count must be non-negative")
        self.iteration_count = count

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
    """Reproduce Fabric's world-size-one DistributedSampler injection."""

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
    return _TheorizerLoader(source, dataloader, sampler, generator, num_episodes)


def build_theorizer_optimizer(model: nn.Module) -> AdamW:
    """Build paper-exact programmer/executor/other AdamW parameter groups."""

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

    context = initialize_distributed(force_cpu=config.force_cpu)
    train_loader: _TheorizerLoader | None = None
    test_loader: _TheorizerLoader | None = None
    metric_log: MetricLog | None = None
    output_directory: Path | None = None
    periodic_checkpoints: list[Path] = []
    best_checkpoint: Path | None = None
    state = GridWorldTheorizerTrainingState(0, 0, 0.0)
    stopped_early = False
    succeeded = False
    try:
        source = capture_git_source(Path(__file__).resolve().parent)
        _validate_run_mode(config, context, source)
        artifacts = _prepare_artifact_evidence(config)
        output_directory = create_run_directory(
            config.output_root,
            config.resolved_run_name,
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
                group=config.wandb_group or f"gridworld-{config.method}-{config.experiment}",
                name=f"{config.resolved_run_name}-{output_directory.name}",
                mode=config.wandb_mode,
                tags=(
                    "gridworld",
                    "theorizer",
                    config.method,
                    config.experiment,
                    f"seed-{config.seed}",
                ),
            ),
            resolved,
        )
        _write_status(output_directory, "running", state, config.canonical)

        # Initialize W&B, reset random seeds, construct the model,
        # then load pretrained observation weights.
        seed_everything(config.seed)
        model = build_neo(config.experiment)
        load_gridworld_observation_checkpoint(
            model,
            config.observation_checkpoint,
            require_canonical=config.canonical,
            selection_policy=config.observation_selection,
            expected_sha256=str(artifacts["observation"]["sha256"]),
        )
        model = model.to(context.device)

        train_loader = build_theorizer_loader(
            config.train_h5,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=config.seed,
            shuffle=True,
            pin_memory=context.device.type == "cuda",
            episode_limit=(None if config.canonical else config.development_train_episodes),
        )
        test_loader = build_theorizer_loader(
            config.test_h5,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=config.seed,
            shuffle=False,
            pin_memory=context.device.type == "cuda",
            episode_limit=(None if config.canonical else config.development_test_episodes),
        )
        _validate_loader_contract(config, train_loader, test_loader)

        optimizer = build_theorizer_optimizer(model)
        scheduler = build_theorizer_scheduler(optimizer, config.paper_experiment)
        start_epoch = 0
        skip_initial_evaluation = False
        if config.resume_checkpoint is not None:
            restored = _load_checkpoint(
                config.resume_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_generator=train_loader.generator,
                test_generator=test_loader.generator,
                device=context.device,
                expected_experiment=config.experiment,
                expected_method=config.method,
                expected_seed=config.seed,
            )
            state = restored
            start_epoch = restored.completed_epochs
            # A periodic checkpoint is written after evaluation and before
            # training. Fabric had therefore opened `completed_epochs` train
            # iterators and `completed_epochs + 1` test iterators at this point.
            train_loader.set_iteration_count(restored.completed_epochs)
            test_loader.set_iteration_count(restored.completed_epochs + 1)
            skip_initial_evaluation = True

        for epoch in range(start_epoch, config.epochs):
            if not (skip_initial_evaluation and epoch == start_epoch):
                legacy_eval, weighted_eval = _evaluate(model, test_loader, context.device)
                if metric_log is not None:
                    metric_log.log(
                        _metric_record(
                            "eval",
                            legacy_eval,
                            weighted_eval,
                            epoch=epoch,
                            global_step=state.global_step,
                            learning_rate=scheduler.get_last_lr()[0],
                        )
                    )
                periodic_number = (
                    state.global_step + 1
                    if (epoch + 1) % config.save_interval == 0
                    else None
                )
                if periodic_number is not None:
                    checkpoint_path = (
                        output_directory / "checkpoints" / f"checkpoint_{periodic_number}.pth"
                    )
                    _save_checkpoint(
                        checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        state=state,
                        evaluated_epoch=epoch,
                        test_accuracy=None,
                        resolved_config=resolved,
                        train_generator=train_loader.generator,
                        test_generator=test_loader.generator,
                    )
                    periodic_checkpoints.append(checkpoint_path)

                support_grid_accuracy = legacy_eval["support_grid_accuracy"]
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
                        optimizer=optimizer,
                        scheduler=scheduler,
                        state=state,
                        evaluated_epoch=epoch,
                        test_accuracy=support_grid_accuracy,
                        resolved_config=resolved,
                        train_generator=train_loader.generator,
                        test_generator=test_loader.generator,
                        overwrite=True,
                    )
            skip_initial_evaluation = False

            model.train()
            for batch in train_loader.batches():
                if config.max_train_steps is not None and state.global_step >= config.max_train_steps:
                    stopped_early = True
                    break
                episode_grids = batch.to(context.device, non_blocking=True).long()
                optimizer.zero_grad(set_to_none=True)
                with bfloat16_autocast(context.device):
                    output = model(episode_grids, is_eval=False)
                output.loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.paper_experiment.training.gradient_clip_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                quantizer = getattr(model, "quantizer", None)
                if quantizer is not None:
                    quantizer.step()
                state = GridWorldTheorizerTrainingState(
                    completed_epochs=state.completed_epochs,
                    global_step=state.global_step + 1,
                    best_support_grid_accuracy=state.best_support_grid_accuracy,
                )
                if state.global_step % config.log_interval == 0 and metric_log is not None:
                    record = _output_values(output)
                    record["gradient_norm"] = float(gradient_norm.detach().float().item())
                    metric_log.log(
                        _metric_record(
                            "train",
                            record,
                            record,
                            epoch=epoch,
                            global_step=state.global_step,
                            learning_rate=scheduler.get_last_lr()[0],
                        )
                    )
                    _write_status(output_directory, "running", state, config.canonical)

            if stopped_early:
                break
            state = GridWorldTheorizerTrainingState(
                completed_epochs=epoch + 1,
                global_step=state.global_step,
                best_support_grid_accuracy=state.best_support_grid_accuracy,
            )

        _write_status(
            output_directory,
            "completed",
            state,
            config.canonical,
            stopped_early=stopped_early,
            best_checkpoint=str(best_checkpoint) if best_checkpoint else None,
            periodic_checkpoints=[str(path) for path in periodic_checkpoints],
        )
        succeeded = True
        return GridWorldTheorizerRunResult(
            output_directory=output_directory,
            state=state,
            stopped_early=stopped_early,
            periodic_checkpoints=tuple(periodic_checkpoints),
            best_checkpoint=best_checkpoint,
        )
    except BaseException as error:
        if output_directory is not None:
            _write_status(
                output_directory,
                "failed",
                state,
                config.canonical,
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


def _validate_run_mode(
    config: GridWorldTheorizerRunConfig,
    context: DistributedContext,
    source: GitSourceState,
) -> None:
    if context.world_size != 1:
        raise ValueError("GridWorld alpha runs require exactly one process and one GPU")
    if not config.canonical:
        return
    development_values = (
        config.max_train_steps,
        config.development_epochs,
        config.development_train_episodes,
        config.development_test_episodes,
        config.development_batch_size,
        config.development_num_workers,
        config.development_save_interval,
        config.development_log_interval,
    )
    if any(value is not None for value in development_values) or config.force_cpu:
        raise ValueError("canonical runs cannot use development overrides")
    if config.seed not in GRIDWORLD_THEORIZER_SEEDS:
        raise ValueError("canonical runs require model seed 42, 43, or 44")
    if context.device.type != "cuda":
        raise ValueError("canonical GridWorld theorizer runs require one CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise ValueError("canonical runs require exactly one visible CUDA device")
    if config.wandb_mode == "disabled":
        raise ValueError("canonical runs require online or offline W&B tracking")


def _prepare_artifact_evidence(
    config: GridWorldTheorizerRunConfig,
) -> dict[str, dict[str, Any]]:
    experiment = config.paper_experiment
    paths = {
        "train": Path(config.train_h5).expanduser().resolve(strict=True),
        "test": Path(config.test_h5).expanduser().resolve(strict=True),
        "observation": Path(config.observation_checkpoint).expanduser().resolve(strict=True),
    }
    evidence = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }
    observation_evidence = inspect_observation_checkpoint(
        paths["observation"],
        require_canonical=config.canonical,
        selection_policy=config.observation_selection,
        expected_sha256=str(evidence["observation"]["sha256"]),
    )
    evidence["observation"].update(observation_evidence.as_dict())
    if config.canonical:
        expected = {
            "train": experiment.train_artifact_sha256,
            "test": experiment.test_artifact_sha256,
        }
        for name, digest in expected.items():
            if evidence[name]["sha256"] != digest:
                raise ValueError(f"{name} artifact does not match the paper SHA-256")
        split_by_name = {split.name: split for split in experiment.profile.splits}
        expected_names = {
            "train": experiment.profile.artifact_filename(split_by_name["practice"]),
            "test": experiment.profile.artifact_filename(split_by_name["exam"]),
        }
        for name, filename in expected_names.items():
            if paths[name].name != filename:
                raise ValueError(f"canonical {name} filename must be {filename}")
    return evidence


def _validate_loader_contract(
    config: GridWorldTheorizerRunConfig,
    train: _TheorizerLoader,
    test: _TheorizerLoader,
) -> None:
    if config.canonical:
        expected_train = config.paper_experiment.training.train_episodes
        expected_test = config.paper_experiment.training.test_episodes
        if train.num_episodes != expected_train or test.num_episodes != expected_test:
            raise ValueError("canonical loader episode counts do not match the paper")
        if len(train.dataloader) != config.paper_experiment.training.train_batches_per_epoch:
            raise ValueError("canonical training loader must contain 391 batches")
        if len(test.dataloader) != config.paper_experiment.training.test_batches_per_epoch:
            raise ValueError("canonical test loader must contain 40 batches")


def _evaluate(
    model: nn.Module,
    loader: _TheorizerLoader,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    unweighted_sums: dict[str, float] = {}
    weighted_sums: dict[str, float] = {}
    batch_count = 0
    sample_count = 0
    with torch.no_grad():
        for batch in loader.batches():
            episode_grids = batch.to(device, non_blocking=True).long()
            with bfloat16_autocast(device):
                output = model(episode_grids, is_eval=True)
            values = _output_values(output)
            weight = episode_grids.shape[0]
            for name, value in values.items():
                unweighted_sums[name] = unweighted_sums.get(name, 0.0) + value
                weighted_sums[name] = weighted_sums.get(name, 0.0) + value * weight
            batch_count += 1
            sample_count += weight
    if not batch_count or not sample_count:
        raise RuntimeError("evaluation loader produced no batches")
    return (
        {name: value / batch_count for name, value in unweighted_sums.items()},
        {name: value / sample_count for name, value in weighted_sums.items()},
    )


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
    legacy: Mapping[str, float],
    weighted: Mapping[str, float],
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
    record.update({f"legacy_unweighted/{split}/{key}": value for key, value in legacy.items()})
    record.update({f"sample_weighted/{split}/{key}": value for key, value in weighted.items()})
    return record


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    state: GridWorldTheorizerTrainingState,
    evaluated_epoch: int,
    test_accuracy: float | None,
    resolved_config: Mapping[str, Any],
    train_generator: torch.Generator,
    test_generator: torch.Generator,
    overwrite: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    payload = {
        "format_version": 1,
        # Checkpoint metadata fields.
        "epoch": evaluated_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": state.global_step,
        "test_accuracy": test_accuracy,
        "args": dict(resolved_config),
        # Release resume/provenance fields.
        "training_state": asdict(state),
        "position": PAPER_THEORIZER_CHECKPOINT_POSITION,
        "quantizer_current_step": (
            model.quantizer.current_step if hasattr(model, "quantizer") else None
        ),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "train_generator_state": train_generator.get_state(),
        "test_generator_state": test_generator.get_state(),
    }
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("xb") as file:
        torch.save(payload, file)
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    train_generator: torch.Generator,
    test_generator: torch.Generator,
    device: torch.device,
    expected_experiment: str | None = None,
    expected_method: str | None = None,
    expected_seed: int | None = None,
) -> GridWorldTheorizerTrainingState:
    payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported theorizer checkpoint format")
    if payload.get("position") != PAPER_THEORIZER_CHECKPOINT_POSITION:
        raise ValueError("resume requires an after-eval-before-train checkpoint")
    arguments = payload.get("args")
    if not isinstance(arguments, dict):
        raise ValueError("theorizer checkpoint is missing resolved arguments")
    if expected_experiment is not None and arguments.get("experiment") != expected_experiment:
        raise ValueError("resume checkpoint experiment does not match the requested run")
    checkpoint_method = arguments.get("method", "neo")
    if expected_method is not None and checkpoint_method != expected_method:
        raise ValueError("resume checkpoint method does not match the requested run")
    if expected_seed is not None and arguments.get("seed") != expected_seed:
        raise ValueError("resume checkpoint seed does not match the requested run")
    restored_state = GridWorldTheorizerTrainingState(**payload["training_state"])
    if payload.get("epoch") != restored_state.completed_epochs:
        raise ValueError("resume checkpoint epoch is inconsistent with training_state")
    if payload.get("global_step") != restored_state.global_step:
        raise ValueError("resume checkpoint global_step is inconsistent with training_state")
    quantizer_step = payload.get("quantizer_current_step")
    if quantizer_step is not None and int(quantizer_step) != restored_state.global_step:
        raise ValueError("resume checkpoint quantizer step does not match global_step")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    if quantizer_step is not None:
        quantizer = getattr(model, "quantizer", None)
        if quantizer is None:
            raise ValueError("checkpoint contains quantizer state for a continuous model")
        quantizer.current_step = int(quantizer_step)
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    train_generator.set_state(payload["train_generator_state"].cpu())
    test_generator.set_state(payload["test_generator_state"].cpu())
    return restored_state


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
        "canonical": config.canonical,
        "method": config.method,
        "experiment": config.experiment,
        "seed": config.seed,
        "training": asdict(config.paper_experiment.training),
        "length_control_coefficient": config.paper_experiment.length_control_coefficient,
        "runtime": {
            "world_size": context.world_size,
            "precision": "bf16-mixed" if context.device.type == "cuda" else "float32",
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "epochs": config.epochs,
            "max_train_steps": config.max_train_steps,
            "checkpoint_position": PAPER_THEORIZER_CHECKPOINT_POSITION,
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
            "force_cpu": config.force_cpu,
        },
        "tracking": {
            "project": config.wandb_project,
            "entity": config.wandb_entity,
            "group": config.wandb_group or f"gridworld-{config.method}-{config.experiment}",
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
    canonical: bool,
    **extra: Any,
) -> None:
    write_json_atomic(
        output_directory / "status.json",
        {
            "status": status,
            "canonical": canonical,
            **asdict(state),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce one GridWorld alpha run")
    parser.add_argument("--method", choices=_TRAINING_METHODS, default="neo")
    parser.add_argument("--experiment", choices=available_theorizer_experiments(), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-h5", type=Path, required=True)
    parser.add_argument("--test-h5", type=Path, required=True)
    parser.add_argument("--observation-checkpoint", type=Path, required=True)
    parser.add_argument("--observation-selection", choices=OBSERVATION_SELECTION_POLICIES,
                        default="recorded-final",
                        help="Best reconstruction requires the complete 500-epoch run record beside the snapshot")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--wandb-project", default="LearningToTheorize")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--development-cpu", action="store_true")
    parser.add_argument("--development-max-train-steps", type=int)
    parser.add_argument("--development-epochs", type=int)
    parser.add_argument("--development-train-episodes", type=int)
    parser.add_argument("--development-test-episodes", type=int)
    parser.add_argument("--development-batch-size", type=int)
    parser.add_argument("--development-num-workers", type=int)
    parser.add_argument("--development-save-interval", type=int)
    parser.add_argument("--development-log-interval", type=int)
    return parser


def _config_from_arguments(arguments: argparse.Namespace) -> GridWorldTheorizerRunConfig:
    if not arguments.development:
        overrides = (
            arguments.development_max_train_steps,
            arguments.development_epochs,
            arguments.development_train_episodes,
            arguments.development_test_episodes,
            arguments.development_batch_size,
            arguments.development_num_workers,
            arguments.development_save_interval,
            arguments.development_log_interval,
        )
        if arguments.development_cpu or any(value is not None for value in overrides):
            raise ValueError("development overrides require --development")
    return GridWorldTheorizerRunConfig(
        method=arguments.method,
        experiment=arguments.experiment,
        seed=arguments.seed,
        train_h5=arguments.train_h5,
        test_h5=arguments.test_h5,
        observation_checkpoint=arguments.observation_checkpoint,
        observation_selection=arguments.observation_selection,
        output_root=arguments.output_root,
        run_name=arguments.run_name,
        wandb_project=arguments.wandb_project,
        wandb_entity=arguments.wandb_entity,
        wandb_group=arguments.wandb_group,
        wandb_mode=arguments.wandb_mode,
        canonical=not arguments.development,
        force_cpu=bool(arguments.development_cpu),
        max_train_steps=arguments.development_max_train_steps,
        development_epochs=arguments.development_epochs,
        development_train_episodes=arguments.development_train_episodes,
        development_test_episodes=arguments.development_test_episodes,
        development_batch_size=arguments.development_batch_size,
        development_num_workers=arguments.development_num_workers,
        development_save_interval=arguments.development_save_interval,
        development_log_interval=arguments.development_log_interval,
        resume_checkpoint=arguments.resume_checkpoint,
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
