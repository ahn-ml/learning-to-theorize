"""Distributed GridWorld observation pretraining."""

from __future__ import annotations

import argparse
import os
import platform
import re
import socket
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor, distributed as dist, nn
from torch.nn.parallel import DistributedDataParallel

from tasks.gridworld.data.artifacts import get_artifact_spec
from tasks.gridworld.data.dataset import (
    GridWorldObservationLoader,
    GridWorldObservationLoaderConfig,
    build_gridworld_observation_loader,
)
from tasks.gridworld.models.observation_pretraining import (
    GridWorldObservationPretrainingModel,
    GridWorldObservationPretrainingObjective,
    GridWorldObservationPretrainingOutput,
)
from tasks.gridworld.observation_pretraining import (
    GridWorldObservationPretrainingConfig,
)
from training import (
    EpochTrainingState,
    OptimizationConfig,
    build_adamw_optimizer,
    build_cosine_warmup_scheduler,
    optimization_step,
    save_training_checkpoint,
)
from training.runtime import (
    DistributedContext,
    ExperimentTrackingConfig,
    GitSourceState,
    MetricLog,
    WandbMode,
    barrier,
    bfloat16_autocast,
    broadcast_object,
    capture_git_source,
    create_run_directory,
    environment_integer,
    initialize_distributed,
    seed_everything,
    sha256_file,
    write_json_atomic,
    write_json_exclusive,
)


PAPER_TRAIN_ARTIFACT_SHA256 = get_artifact_spec(
    "vae-pretraining", "practice"
).sha256
PAPER_TEST_ARTIFACT_SHA256 = get_artifact_spec("vae-pretraining", "exam").sha256
PAPER_PRECISION = "bf16-mixed"
PAPER_DISTRIBUTED_BACKEND = "nccl"
PAPER_DISTRIBUTED_TIMEOUT_MINUTES = 30
PAPER_CHECKPOINT_POSITION = "after_eval_before_train"

_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_METRIC_NAMES = (
    "loss",
    "reconstruction_loss",
    "kl_loss",
    "pixel_accuracy",
    "grid_accuracy",
    "theorizer_reconstruction_loss",
    "transition_consistency_loss",
    "action_vq_loss",
)


@dataclass(frozen=True, slots=True)
class GridWorldObservationRunConfig:
    """Resolved observation-pretraining settings."""

    train_h5: Path
    test_h5: Path
    output_root: Path
    run_name: str = "gridworld-observation-paper-v1"
    wandb_project: str = "LearningToTheorize"
    wandb_entity: str | None = None
    wandb_group: str | None = "gridworld-observation-pretraining"
    wandb_mode: WandbMode = "online"
    canonical: bool = True
    pretraining_profile: str = "recovered"
    pretraining_sweep: bool = False
    force_cpu: bool = False
    max_train_steps: int | None = None
    pretraining: GridWorldObservationPretrainingConfig = field(
        default_factory=GridWorldObservationPretrainingConfig
    )

    def __post_init__(self) -> None:
        if self.pretraining_profile not in ("recovered", "appendix"):
            raise ValueError("unknown pretraining profile")
        if not _RUN_NAME.fullmatch(self.run_name):
            raise ValueError(
                "run_name must contain only letters, numbers, '.', '-', and '_'"
            )
        if not self.wandb_project.strip():
            raise ValueError("wandb_project must not be empty")
        if self.wandb_mode not in ("online", "offline", "disabled"):
            raise ValueError("wandb_mode must be online, offline, or disabled")
        if self.max_train_steps is not None and self.max_train_steps < 1:
            raise ValueError("max_train_steps must be positive when provided")


@dataclass(frozen=True, slots=True)
class GridWorldObservationRunResult:
    """Completed runner state returned to programmatic callers."""

    output_directory: Path
    state: EpochTrainingState
    stopped_early: bool
    checkpoints: tuple[Path, ...]


def paper_checkpoint_number(
    *, epoch: int, global_step: int, save_interval_epochs: int
) -> int | None:
    """Return the one-based checkpoint label before an epoch."""

    if epoch < 0 or global_step < 0 or save_interval_epochs < 1:
        raise ValueError("checkpoint epoch, step, and interval must be valid")
    return global_step + 1 if (epoch + 1) % save_interval_epochs == 0 else None


def save_best_reconstruction_checkpoint(
    output_directory: Path,
    *,
    score: float,
    best_score: float,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: EpochTrainingState,
    optimization: OptimizationConfig,
    metadata: Mapping[str, str | int | float | bool | None],
) -> float:
    """Keep the first maximum of globally aggregated validation accuracy.

    Saving does not stop training or replace the periodic snapshots.
    Selection uses observation reconstruction, never downstream or OOD scores.
    """
    if not 0.0 <= score <= 1.0:
        raise ValueError("validation grid accuracy must be finite and in [0, 1]")
    if score <= best_score:
        return best_score
    checkpoint = output_directory / "best_reconstruction" / f"checkpoint_{state.global_step + 1}.pt"
    save_training_checkpoint(
        checkpoint, model=model, optimizer=optimizer, scheduler=scheduler,
        state=state, optimization=optimization, metadata=metadata,
    )
    write_json_atomic(output_directory / "best_reconstruction.json", {
        "policy": "global-validation-reconstruction-first-maximum-v1",
        "metric": "grid_accuracy", "score": score, "direction": "max",
        "tie_break": "earliest", "aggregation": "global_sample_weighted",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_state": asdict(state),
        "selection_uses_downstream_scores": False,
        "selection_uses_ood_scores": False,
        "training_continues_after_selection": True,
    })
    return score


def run_gridworld_observation_pretraining(
    config: GridWorldObservationRunConfig,
) -> GridWorldObservationRunResult:
    """Run observation pretraining with torchrun or a single process."""

    context = initialize_distributed(
        backend=PAPER_DISTRIBUTED_BACKEND,
        timeout_minutes=PAPER_DISTRIBUTED_TIMEOUT_MINUTES,
        force_cpu=config.force_cpu,
    )
    train_loader: GridWorldObservationLoader | None = None
    test_loader: GridWorldObservationLoader | None = None
    metric_log: MetricLog | None = None
    output_directory: Path | None = None
    checkpoints: list[Path] = []
    state = EpochTrainingState(completed_epochs=0, global_step=0)
    global_step = 0
    best_reconstruction_score = 0.0
    completed_epochs = 0
    stopped_early = False
    succeeded = False
    try:
        source = capture_git_source(Path(__file__).resolve().parent)
        _validate_run_mode(config, context, source)
        artifacts = _prepare_artifact_evidence(config, context)
        output_directory = create_run_directory(
            config.output_root,
            config.run_name,
            source,
            context,
        )
        resolved_config = _resolved_config(config, context, source, artifacts)
        metric_log = _initialize_metric_log(
            config,
            context,
            source,
            output_directory,
            resolved_config,
            state,
        )

        # Reset random seeds after W&B initialization.
        # Keep that order so tracker internals cannot perturb model construction.
        seed_everything(config.pretraining.seed)
        model: nn.Module = GridWorldObservationPretrainingModel()

        loader_config = GridWorldObservationLoaderConfig(
            per_rank_batch_size=config.pretraining.per_rank_batch_size,
            world_size=config.pretraining.effective_world_size,
            rank=context.rank,
            num_workers=config.pretraining.data_loader_workers,
            seed=config.pretraining.seed,
            pin_memory=context.device.type == "cuda",
        )
        train_loader = build_gridworld_observation_loader(
            config.train_h5, loader_config, shuffle=True
        )
        test_loader = build_gridworld_observation_loader(
            config.test_h5, loader_config, shuffle=False
        )
        _validate_loader_contract(config, train_loader, test_loader)

        model = model.to(context.device)
        if context.is_distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=True,
                find_unused_parameters=True,
            )
        objective = GridWorldObservationPretrainingObjective(kl_weight=config.pretraining.kl_weight)

        def forward_objective(
            current_model: nn.Module,
            episode_grids: Tensor,
        ) -> GridWorldObservationPretrainingOutput:
            with bfloat16_autocast(context.device):
                return objective(current_model, episode_grids)

        optimization = config.pretraining.optimization()
        optimizer = build_adamw_optimizer(model, optimization)
        scheduler = build_cosine_warmup_scheduler(optimizer, optimization)

        for epoch in range(config.pretraining.epochs):
            evaluation, legacy_evaluation = _evaluate(
                model,
                objective,
                test_loader,
                epoch=epoch,
                context=context,
            )
            if context.is_global_zero and metric_log is not None:
                metric_log.log(
                    _metric_record(
                        "eval",
                        evaluation,
                        legacy_evaluation,
                        epoch=epoch,
                        global_step=global_step,
                        learning_rate=scheduler.get_last_lr()[0],
                    )
                )
                _write_status(
                    output_directory,
                    status="running",
                    state=EpochTrainingState(
                        completed_epochs=epoch,
                        global_step=global_step,
                    ),
                    canonical=config.canonical,
                    tracker_url=metric_log.tracker_url,
                    phase=PAPER_CHECKPOINT_POSITION,
                    current_epoch=epoch,
                    checkpoints=[str(path) for path in checkpoints],
                )

            barrier(context)
            best_payload: list[Any] = [None]
            if context.is_global_zero:
                try:
                    best_payload[0] = {"value": save_best_reconstruction_checkpoint(
                        output_directory,
                        score=evaluation["grid_accuracy"],
                        best_score=best_reconstruction_score,
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        state=EpochTrainingState(completed_epochs=epoch, global_step=global_step),
                        optimization=optimization,
                        metadata={
                            "domain": "gridworld", "stage": "observation_pretraining",
                            "position": PAPER_CHECKPOINT_POSITION,
                            "evaluated_epoch": epoch,
                            "historical_checkpoint_number": global_step + 1,
                            "source_commit": source.commit, "canonical": config.canonical,
                            "pretraining_profile": config.pretraining_profile,
                            "pretraining_sweep": config.pretraining_sweep,
                            "kl_weight": config.pretraining.kl_weight,
                        },
                    )}
                except BaseException as error:
                    best_payload[0] = {"error": f"{type(error).__name__}: {error}"}
            broadcast_object(best_payload, context)
            if not isinstance(best_payload[0], dict) or "error" in best_payload[0]:
                raise RuntimeError(f"best reconstruction checkpoint save failed: {best_payload[0]}")
            best_reconstruction_score = best_payload[0]["value"]
            checkpoint_number = paper_checkpoint_number(
                epoch=epoch,
                global_step=global_step,
                save_interval_epochs=config.pretraining.save_interval_epochs,
            )
            if checkpoint_number is not None:
                checkpoint_payload: list[Any] = [None]
                if context.is_global_zero:
                    try:
                        checkpoint_path = (
                            output_directory
                            / "checkpoints"
                            / f"checkpoint_{checkpoint_number}.pt"
                        )
                        save_training_checkpoint(
                            checkpoint_path,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            state=EpochTrainingState(
                                completed_epochs=epoch,
                                global_step=global_step,
                            ),
                            optimization=optimization,
                            metadata={
                                "domain": "gridworld",
                                "stage": "observation_pretraining",
                                "position": PAPER_CHECKPOINT_POSITION,
                                "evaluated_epoch": epoch,
                                "historical_checkpoint_number": checkpoint_number,
                                "source_commit": source.commit,
                                "canonical": config.canonical,
                                "pretraining_profile": config.pretraining_profile,
                                "pretraining_sweep": config.pretraining_sweep,
                                "kl_weight": config.pretraining.kl_weight,
                            },
                        )
                        checkpoints.append(checkpoint_path)
                        checkpoint_payload[0] = {"value": str(checkpoint_path)}
                    except BaseException as error:
                        checkpoint_payload[0] = {
                            "error": f"{type(error).__name__}: {error}"
                        }
                broadcast_object(checkpoint_payload, context)
                checkpoint_result = checkpoint_payload[0]
                if not isinstance(checkpoint_result, dict):
                    raise RuntimeError("rank zero did not report checkpoint status")
                if "error" in checkpoint_result:
                    raise RuntimeError(
                        f"checkpoint save failed on rank zero: "
                        f"{checkpoint_result['error']}"
                    )
            barrier(context)

            train_loader.set_epoch(epoch)
            model.train()
            for batch in train_loader.dataloader:
                if (
                    config.max_train_steps is not None
                    and global_step >= config.max_train_steps
                ):
                    stopped_early = True
                    break
                episode_grids = batch.to(context.device, non_blocking=True)
                step = optimization_step(
                    model,
                    episode_grids,
                    forward_objective,
                    optimizer,
                    scheduler,
                    global_step=global_step,
                    max_gradient_norm=optimization.max_gradient_norm,
                )
                global_step = step.global_step
                if global_step % config.pretraining.log_interval_steps == 0:
                    aggregate = _reduce_output(step.objective, context)
                    aggregate["gradient_norm"] = _reduce_max(
                        step.gradient_norm.detach(), context
                    )
                    legacy_train = (
                        _output_values(step.objective)
                        if context.is_global_zero
                        else None
                    )
                    if context.is_global_zero and metric_log is not None:
                        metric_log.log(
                            _metric_record(
                                "train",
                                aggregate,
                                legacy_train,
                                epoch=epoch,
                                global_step=global_step,
                                learning_rate=step.learning_rates[0],
                            )
                        )
                        _write_status(
                            output_directory,
                            status="running",
                            state=EpochTrainingState(
                                completed_epochs=epoch,
                                global_step=global_step,
                            ),
                            canonical=config.canonical,
                            tracker_url=metric_log.tracker_url,
                            phase="training",
                            current_epoch=epoch,
                            checkpoints=[str(path) for path in checkpoints],
                        )

            if stopped_early:
                break
            completed_epochs = epoch + 1

        state = EpochTrainingState(
            completed_epochs=completed_epochs,
            global_step=global_step,
        )
        if context.is_global_zero:
            _write_status(
                output_directory,
                status="completed",
                state=state,
                canonical=config.canonical,
                tracker_url=metric_log.tracker_url if metric_log else None,
                stopped_early=stopped_early,
                checkpoints=[str(path) for path in checkpoints],
            )
        succeeded = True
        return GridWorldObservationRunResult(
            output_directory=output_directory,
            state=state,
            stopped_early=stopped_early,
            checkpoints=tuple(checkpoints),
        )
    except BaseException as error:
        state = EpochTrainingState(
            completed_epochs=completed_epochs,
            global_step=global_step,
        )
        if context.is_global_zero and output_directory is not None:
            _write_status(
                output_directory,
                status="failed",
                state=state,
                canonical=config.canonical,
                error=f"{type(error).__name__}: {error}",
            )
        raise
    finally:
        if train_loader is not None:
            train_loader.dataset.close()
        if test_loader is not None:
            test_loader.dataset.close()
        if context.owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
        if metric_log is not None:
            metric_log.finish(exit_code=0 if succeeded else 1)


def _validate_run_mode(
    config: GridWorldObservationRunConfig,
    context: DistributedContext,
    source: GitSourceState,
) -> None:
    if config.pretraining.effective_world_size != context.world_size:
        raise ValueError(
            "configured effective_world_size does not match torchrun WORLD_SIZE"
        )
    if not config.canonical and not config.pretraining_sweep:
        return
    expected = GridWorldObservationPretrainingConfig(
        kl_weight=1e-5 if config.pretraining_profile == "appendix" else 0.0
    )
    if config.canonical and config.pretraining != expected:
        raise ValueError("canonical runs require the selected pretraining configuration")
    if config.max_train_steps is not None:
        raise ValueError("canonical runs cannot stop after a development step limit")
    if context.world_size != 4 or context.device.type != "cuda":
        raise ValueError("canonical GridWorld pretraining requires four CUDA ranks")
    if config.wandb_mode == "disabled":
        raise ValueError("canonical runs require online or offline W&B tracking")


def _initialize_metric_log(
    config: GridWorldObservationRunConfig,
    context: DistributedContext,
    source: GitSourceState,
    output_directory: Path,
    resolved_config: Mapping[str, Any],
    state: EpochTrainingState,
) -> MetricLog | None:
    metric_log: MetricLog | None = None
    payload: list[Any] = [None]
    if context.is_global_zero:
        try:
            write_json_exclusive(output_directory / "config.json", resolved_config)
            write_json_exclusive(output_directory / "source.json", asdict(source))
            metric_log = MetricLog(
                output_directory / "metrics.jsonl",
                ExperimentTrackingConfig(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    group=config.wandb_group,
                    name=f"{config.run_name}-{output_directory.name}",
                    mode=config.wandb_mode,
                    tags=(
                        "gridworld",
                        "observation-pretraining",
                        "paper-reproduction",
                    ),
                ),
                resolved_config,
            )
            _write_status(
                output_directory,
                status="running",
                state=state,
                canonical=config.canonical,
                tracker_url=metric_log.tracker_url,
            )
            payload[0] = {"value": True}
        except BaseException as error:
            if metric_log is not None:
                try:
                    metric_log.finish(exit_code=1)
                except Exception:
                    pass
                metric_log = None
            payload[0] = {"error": f"{type(error).__name__}: {error}"}
    broadcast_object(payload, context)
    result = payload[0]
    if not isinstance(result, dict):
        raise RuntimeError("rank zero did not report logging initialization")
    if "error" in result:
        raise RuntimeError(
            f"logging initialization failed on rank zero: {result['error']}"
        )
    return metric_log


def _prepare_artifact_evidence(
    config: GridWorldObservationRunConfig,
    context: DistributedContext,
) -> dict[str, dict[str, int | str | None]]:
    payload: list[Any] = [None]
    if context.is_global_zero:
        try:
            train = Path(config.train_h5).expanduser().resolve(strict=True)
            test = Path(config.test_h5).expanduser().resolve(strict=True)
            evidence = {
                "train": {
                    "path": str(train),
                    "bytes": train.stat().st_size,
                    "sha256": sha256_file(train) if (config.canonical or config.pretraining_sweep) else None,
                },
                "test": {
                    "path": str(test),
                    "bytes": test.stat().st_size,
                    "sha256": sha256_file(test) if (config.canonical or config.pretraining_sweep) else None,
                },
            }
            if config.canonical or config.pretraining_sweep:
                if evidence["train"]["sha256"] != PAPER_TRAIN_ARTIFACT_SHA256:
                    raise ValueError("training HDF5 does not match the paper artifact")
                if evidence["test"]["sha256"] != PAPER_TEST_ARTIFACT_SHA256:
                    raise ValueError("test HDF5 does not match the paper artifact")
            payload[0] = {"value": evidence}
        except BaseException as error:
            payload[0] = {"error": f"{type(error).__name__}: {error}"}
    broadcast_object(payload, context)
    result = payload[0]
    if not isinstance(result, dict):
        raise RuntimeError("rank zero did not provide artifact evidence")
    if "error" in result:
        raise RuntimeError(f"artifact validation failed on rank zero: {result['error']}")
    return result["value"]


def _validate_loader_contract(
    config: GridWorldObservationRunConfig,
    train: GridWorldObservationLoader,
    test: GridWorldObservationLoader,
) -> None:
    paper = config.pretraining
    if len(train.dataset) != paper.train_episodes:
        raise ValueError(
            f"expected {paper.train_episodes} training episodes, got {len(train.dataset)}"
        )
    if len(test.dataset) != paper.test_episodes:
        raise ValueError(
            f"expected {paper.test_episodes} test episodes, got {len(test.dataset)}"
        )
    if len(train.dataloader) != paper.train_steps_per_epoch:
        raise ValueError("training loader step count does not match resolved config")
    if len(test.dataloader) != paper.test_steps_per_epoch:
        raise ValueError("test loader step count does not match resolved config")


def _evaluate(
    model: nn.Module,
    objective: GridWorldObservationPretrainingObjective,
    loader: GridWorldObservationLoader,
    *,
    epoch: int,
    context: DistributedContext,
) -> tuple[dict[str, float], dict[str, float] | None]:
    loader.set_epoch(epoch)
    model.eval()
    weighted_sums = torch.zeros(len(_METRIC_NAMES) + 1, dtype=torch.float64, device=context.device)
    legacy_sums = {name: 0.0 for name in _METRIC_NAMES}
    legacy_batches = 0
    with torch.no_grad():
        for batch in loader.dataloader:
            episode_grids = batch.to(context.device, non_blocking=True)
            with bfloat16_autocast(context.device):
                output = objective(model, episode_grids)
            values = _output_values(output)
            weight = float(output.num_observations)
            for index, name in enumerate(_METRIC_NAMES):
                weighted_sums[index] += values[name] * weight
            weighted_sums[-1] += weight
            if context.is_global_zero:
                for name in _METRIC_NAMES:
                    legacy_sums[name] += values[name]
                legacy_batches += 1
    if context.is_distributed:
        dist.all_reduce(weighted_sums, op=dist.ReduceOp.SUM)
    if weighted_sums[-1].item() == 0.0:
        raise RuntimeError("evaluation loader produced no observations")
    aggregate = {
        name: (weighted_sums[index] / weighted_sums[-1]).item()
        for index, name in enumerate(_METRIC_NAMES)
    }
    legacy = None
    if context.is_global_zero:
        if legacy_batches == 0:
            raise RuntimeError("rank-zero evaluation loader produced no batches")
        legacy = {
            name: legacy_sums[name] / legacy_batches for name in _METRIC_NAMES
        }
    return aggregate, legacy


def _reduce_output(
    output: GridWorldObservationPretrainingOutput,
    context: DistributedContext,
) -> dict[str, float]:
    values = _output_values(output)
    weight = float(output.num_observations)
    packed = torch.tensor(
        [values[name] * weight for name in _METRIC_NAMES] + [weight],
        dtype=torch.float64,
        device=context.device,
    )
    if context.is_distributed:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return {
        name: (packed[index] / packed[-1]).item()
        for index, name in enumerate(_METRIC_NAMES)
    }


def _reduce_max(value: Tensor, context: DistributedContext) -> float:
    result = value.float().clone()
    if context.is_distributed:
        dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return result.item()


def _output_values(
    output: GridWorldObservationPretrainingOutput,
) -> dict[str, float]:
    return {
        name: float(getattr(output, name).detach().float().item())
        for name in _METRIC_NAMES
    }


def _metric_record(
    split: str,
    aggregate: Mapping[str, float],
    legacy_rank_zero: Mapping[str, float] | None,
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
    record.update({f"{split}/{key}": value for key, value in aggregate.items()})
    if legacy_rank_zero is not None:
        record.update(
            {
                f"legacy_rank0/{split}/{key}": value
                for key, value in legacy_rank_zero.items()
            }
        )
    return record


def _resolved_config(
    config: GridWorldObservationRunConfig,
    context: DistributedContext,
    source: GitSourceState,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "domain": "gridworld",
        "stage": "observation_pretraining",
        "canonical": config.canonical,
        "run_name": config.run_name,
        "pretraining_profile": config.pretraining_profile,
        "pretraining_sweep": config.pretraining_sweep,
        "pretraining": asdict(config.pretraining),
        "runtime": {
            "precision": PAPER_PRECISION if context.device.type == "cuda" else "float32",
            "backend": PAPER_DISTRIBUTED_BACKEND if context.is_distributed else None,
            "timeout_minutes": PAPER_DISTRIBUTED_TIMEOUT_MINUTES,
            "find_unused_parameters": True,
            "broadcast_buffers": True,
            "world_size": context.world_size,
            "force_cpu": config.force_cpu,
            "max_train_steps": config.max_train_steps,
            "checkpoint_position": PAPER_CHECKPOINT_POSITION,
            "autocast_scope": "forward_and_loss",
            "cublas_workspace_config": ":4096:8",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "deterministic_algorithms_warn_only": True,
        },
        "tracking": {
            "project": config.wandb_project,
            "entity": config.wandb_entity,
            "group": config.wandb_group,
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
            "torch": torch.__version__,
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
    *,
    status: str,
    state: EpochTrainingState,
    canonical: bool,
    **extra: Any,
) -> None:
    payload = {
        "status": status,
        "canonical": canonical,
        "completed_epochs": state.completed_epochs,
        "global_step": state.global_step,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    write_json_atomic(output_directory / "status.json", payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce GridWorld phase-one observation pretraining",
    )
    parser.add_argument("--pretraining-profile", choices=("recovered", "appendix"), default="recovered")
    parser.add_argument("--pretraining-overrides", type=Path,
                        help="JSON hyperparameter overrides; recorded as a non-paper sweep")
    parser.add_argument("--train-h5", type=Path, required=True)
    parser.add_argument("--test-h5", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", default="gridworld-observation-paper-v1")
    parser.add_argument("--wandb-project", default="LearningToTheorize")
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--wandb-group", default="gridworld-observation-pretraining"
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument(
        "--development",
        action="store_true",
        help="allow a dirty source, non-paper topology, and bounded smoke settings",
    )
    parser.add_argument("--development-max-train-steps", type=int)
    parser.add_argument("--development-cpu", action="store_true")
    parser.add_argument("--development-epochs", type=int)
    parser.add_argument("--development-train-episodes", type=int)
    parser.add_argument("--development-test-episodes", type=int)
    parser.add_argument("--development-batch-size", type=int)
    parser.add_argument("--development-num-workers", type=int)
    parser.add_argument("--development-save-interval", type=int)
    parser.add_argument("--development-log-interval", type=int)
    return parser


def _config_from_arguments(
    arguments: argparse.Namespace,
) -> GridWorldObservationRunConfig:
    development = bool(arguments.development)
    if not development:
        forbidden = (
            arguments.development_max_train_steps,
            arguments.development_epochs,
            arguments.development_train_episodes,
            arguments.development_test_episodes,
            arguments.development_batch_size,
            arguments.development_num_workers,
            arguments.development_save_interval,
            arguments.development_log_interval,
        )
        if any(value is not None for value in forbidden) or arguments.development_cpu:
            raise ValueError("development overrides require --development")
        pretraining = GridWorldObservationPretrainingConfig()
    else:
        if arguments.development_train_episodes is None:
            raise ValueError("--development-train-episodes is required")
        if arguments.development_test_episodes is None:
            raise ValueError("--development-test-episodes is required")
        world_size = environment_integer("WORLD_SIZE", 1)
        pretraining = GridWorldObservationPretrainingConfig(
            epochs=(
                1
                if arguments.development_epochs is None
                else arguments.development_epochs
            ),
            train_episodes=arguments.development_train_episodes,
            test_episodes=arguments.development_test_episodes,
            per_rank_batch_size=(
                2
                if arguments.development_batch_size is None
                else arguments.development_batch_size
            ),
            effective_world_size=world_size,
            data_loader_workers=(
                0
                if arguments.development_num_workers is None
                else arguments.development_num_workers
            ),
            save_interval_epochs=(
                1
                if arguments.development_save_interval is None
                else arguments.development_save_interval
            ),
            log_interval_steps=(
                1
                if arguments.development_log_interval is None
                else arguments.development_log_interval
            ),
        )
    pretraining = replace(pretraining, kl_weight=1e-5 if arguments.pretraining_profile == "appendix" else 0.0)
    override_path = getattr(arguments, "pretraining_overrides", None)
    if override_path is not None:
        if development:
            raise ValueError("pretraining sweeps cannot use development overrides")
        from tasks.gridworld.pretraining_sweep import apply_pretraining_overrides
        pretraining = apply_pretraining_overrides(pretraining, override_path)
    return GridWorldObservationRunConfig(
        pretraining_profile=arguments.pretraining_profile,
        pretraining_sweep=override_path is not None,
        train_h5=arguments.train_h5,
        test_h5=arguments.test_h5,
        output_root=arguments.output_root,
        run_name=arguments.run_name,
        wandb_project=arguments.wandb_project,
        wandb_entity=arguments.wandb_entity,
        wandb_group=arguments.wandb_group,
        wandb_mode=arguments.wandb_mode,
        canonical=not development and override_path is None,
        force_cpu=bool(arguments.development_cpu),
        max_train_steps=arguments.development_max_train_steps,
        pretraining=pretraining,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        config = _config_from_arguments(arguments)
        result = run_gridworld_observation_pretraining(config)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    if environment_integer("RANK", 0) == 0:
        print(result.output_directory)


if __name__ == "__main__":
    main()
