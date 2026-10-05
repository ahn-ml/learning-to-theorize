"""Distributed GridWorld observation pretraining."""

from __future__ import annotations

import argparse
import os
import platform
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor, distributed as dist, nn
from torch.nn.parallel import DistributedDataParallel

from tasks.gridworld.data.dataset import (
    GridWorldHDF5Dataset,
    GridWorldObservationLoader,
    GridWorldObservationLoaderConfig,
    GridWorldSingleObservationLoader,
    build_gridworld_observation_loader,
)
from tasks.gridworld.models.vae import VAE
from tasks.gridworld.observation_pretraining import (
    GridWorldObservationObjective,
    GridWorldObservationObjectiveOutput,
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


RUN_NAME = "gridworld-observation-pretraining"
DISTRIBUTED_BACKEND = "nccl"
DISTRIBUTED_TIMEOUT_MINUTES = 30
_METRIC_NAMES = (
    "loss",
    "reconstruction_loss",
    "kl_loss",
    "pixel_accuracy",
    "grid_accuracy",
)


@dataclass(frozen=True, slots=True)
class GridWorldObservationRunConfig:
    """Inputs, tracking, and recipe of one observation-pretraining run."""

    train_h5: Path
    test_h5: Path
    output_root: Path
    wandb_project: str = "LearningToTheorize"
    wandb_entity: str | None = None
    wandb_group: str | None = "gridworld-observation-pretraining"
    wandb_mode: WandbMode = "online"
    pretraining: GridWorldObservationPretrainingConfig = field(
        default_factory=GridWorldObservationPretrainingConfig
    )

    def __post_init__(self) -> None:
        if not self.wandb_project.strip():
            raise ValueError("wandb_project must not be empty")


@dataclass(frozen=True, slots=True)
class GridWorldObservationRunResult:
    """Completed runner state returned to programmatic callers."""

    output_directory: Path
    state: EpochTrainingState


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

    Saving does not stop training. Selection uses observation reconstruction,
    never downstream or OOD scores.
    """
    if not 0.0 <= score <= 1.0:
        raise ValueError("validation grid accuracy must be finite and in [0, 1]")
    if score <= best_score:
        return best_score
    relative_checkpoint = Path("best_reconstruction") / f"checkpoint_{state.global_step + 1}.pt"
    checkpoint = output_directory / relative_checkpoint
    save_training_checkpoint(
        checkpoint, model=model, optimizer=optimizer, scheduler=scheduler,
        state=state, optimization=optimization, metadata=metadata,
    )
    write_json_atomic(output_directory / "best_reconstruction.json", {
        "policy": "global-validation-reconstruction-first-maximum-v1",
        "metric": "grid_accuracy", "score": score, "direction": "max",
        "tie_break": "earliest", "aggregation": "global_sample_weighted",
        "checkpoint": relative_checkpoint.as_posix(),
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
    """Run observation pretraining under torchrun."""

    context = initialize_distributed(
        backend=DISTRIBUTED_BACKEND,
        timeout_minutes=DISTRIBUTED_TIMEOUT_MINUTES,
    )
    train_dataset: GridWorldHDF5Dataset | None = None
    test_loader: GridWorldObservationLoader | None = None
    metric_log: MetricLog | None = None
    output_directory: Path | None = None
    global_step = 0
    best_reconstruction_score = 0.0
    completed_epochs = 0
    succeeded = False
    try:
        if config.pretraining.effective_world_size != context.world_size:
            raise ValueError(
                f"observation pretraining requires {config.pretraining.effective_world_size} "
                f"processes, got WORLD_SIZE={context.world_size}"
            )
        source = capture_git_source(Path(__file__).resolve().parent)
        artifacts = _prepare_artifact_evidence(config, context)
        output_directory = create_run_directory(
            config.output_root,
            RUN_NAME,
            source,
            context,
        )
        resolved_config = _resolved_config(config, context, source, artifacts)
        metric_log = _initialize_metric_log(
            config,
            context,
            output_directory,
            resolved_config,
            source,
        )

        # Reset random seeds after W&B initialization.
        # Keep that order so tracker internals cannot perturb model construction.
        seed_everything(config.pretraining.seed)
        model: nn.Module = VAE()

        loader_config = GridWorldObservationLoaderConfig(
            per_rank_batch_size=config.pretraining.per_rank_batch_size,
            world_size=config.pretraining.effective_world_size,
            rank=context.rank,
            num_workers=config.pretraining.data_loader_workers,
            seed=config.pretraining.seed,
            pin_memory=context.device.type == "cuda",
        )
        train_dataset = GridWorldHDF5Dataset(config.train_h5)
        test_loader = build_gridworld_observation_loader(
            config.test_h5, loader_config, shuffle=False
        )
        train_loader = GridWorldSingleObservationLoader(train_dataset, loader_config)
        _validate_loader_contract(config.pretraining, train_loader, test_loader)

        model = model.to(context.device)
        if context.is_distributed:
            # The encoder's to_state layer is not used by the forward pass.
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=True,
                find_unused_parameters=True,
            )
        objective = GridWorldObservationObjective(kl_weight=config.pretraining.kl_weight)

        def forward_objective(
            current_model: nn.Module,
            episode_grids: Tensor,
        ) -> GridWorldObservationObjectiveOutput:
            with bfloat16_autocast(context.device):
                return objective(current_model, episode_grids)

        optimization = config.pretraining.optimization()
        optimizer = build_adamw_optimizer(model, optimization)
        scheduler = build_cosine_warmup_scheduler(optimizer, optimization)

        for epoch in range(config.pretraining.epochs):
            evaluation = _evaluate(
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
                    tracker_url=metric_log.tracker_url,
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
                            "evaluated_epoch": epoch, "source_commit": source.commit,
                        },
                    )}
                except BaseException as error:
                    best_payload[0] = {"error": f"{type(error).__name__}: {error}"}
            broadcast_object(best_payload, context)
            if not isinstance(best_payload[0], dict) or "error" in best_payload[0]:
                raise RuntimeError(f"best reconstruction checkpoint save failed: {best_payload[0]}")
            best_reconstruction_score = best_payload[0]["value"]
            barrier(context)

            train_loader.set_epoch(epoch)
            model.train()
            for batch in train_loader:
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
                    if context.is_global_zero and metric_log is not None:
                        metric_log.log(
                            _metric_record(
                                "train",
                                aggregate,
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
                            tracker_url=metric_log.tracker_url,
                        )
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
                tracker_url=metric_log.tracker_url if metric_log else None,
            )
        succeeded = True
        return GridWorldObservationRunResult(
            output_directory=output_directory,
            state=state,
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
                error=f"{type(error).__name__}: {error}",
            )
        raise
    finally:
        if train_dataset is not None:
            train_dataset.close()
        if test_loader is not None:
            test_loader.dataset.close()
        if context.owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
        if metric_log is not None:
            metric_log.finish(exit_code=0 if succeeded else 1)


def _initialize_metric_log(
    config: GridWorldObservationRunConfig,
    context: DistributedContext,
    output_directory: Path,
    resolved_config: Mapping[str, Any],
    source: GitSourceState,
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
                    name=f"{RUN_NAME}-{output_directory.name}",
                    mode=config.wandb_mode,
                    tags=("gridworld", "observation-pretraining"),
                ),
                resolved_config,
            )
            _write_status(
                output_directory,
                status="running",
                state=EpochTrainingState(completed_epochs=0, global_step=0),
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
) -> dict[str, dict[str, int | str]]:
    payload: list[Any] = [None]
    if context.is_global_zero:
        try:
            evidence = {}
            for name, value in (("train", config.train_h5), ("test", config.test_h5)):
                path = Path(value).expanduser().resolve(strict=True)
                evidence[name] = {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
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
    pretraining: GridWorldObservationPretrainingConfig,
    train: GridWorldSingleObservationLoader,
    test: GridWorldObservationLoader,
) -> None:
    if len(train.dataset) != pretraining.train_episodes:
        raise ValueError(
            f"expected {pretraining.train_episodes} training episodes, got {len(train.dataset)}"
        )
    if len(test.dataset) != pretraining.test_episodes:
        raise ValueError(
            f"expected {pretraining.test_episodes} test episodes, got {len(test.dataset)}"
        )
    if len(train) != pretraining.train_steps_per_epoch:
        raise ValueError("training loader step count does not match the recipe")
    if len(test.dataloader) != pretraining.test_steps_per_epoch:
        raise ValueError("test loader step count does not match the recipe")


def _evaluate(
    model: nn.Module,
    objective: GridWorldObservationObjective,
    loader: GridWorldObservationLoader,
    *,
    epoch: int,
    context: DistributedContext,
) -> dict[str, float]:
    loader.set_epoch(epoch)
    model.eval()
    weighted_sums = torch.zeros(len(_METRIC_NAMES) + 1, dtype=torch.float64, device=context.device)
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
    if context.is_distributed:
        dist.all_reduce(weighted_sums, op=dist.ReduceOp.SUM)
    if weighted_sums[-1].item() == 0.0:
        raise RuntimeError("evaluation loader produced no observations")
    return {
        name: (weighted_sums[index] / weighted_sums[-1]).item()
        for index, name in enumerate(_METRIC_NAMES)
    }


def _reduce_output(
    output: GridWorldObservationObjectiveOutput,
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
    output: GridWorldObservationObjectiveOutput,
) -> dict[str, float]:
    return {
        name: float(getattr(output, name).detach().float().item())
        for name in _METRIC_NAMES
    }


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
        "pretraining": asdict(config.pretraining),
        "runtime": {
            "precision": "bf16-mixed" if context.device.type == "cuda" else "float32",
            "backend": DISTRIBUTED_BACKEND if context.is_distributed else None,
            "timeout_minutes": DISTRIBUTED_TIMEOUT_MINUTES,
            "find_unused_parameters": True,
            "broadcast_buffers": True,
            "world_size": context.world_size,
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
    **extra: Any,
) -> None:
    payload = {
        "status": status,
        "completed_epochs": state.completed_epochs,
        "global_step": state.global_step,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    write_json_atomic(output_directory / "status.json", payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretrain the GridWorld observation VAE with the release recipe",
    )
    parser.add_argument("--train-h5", type=Path, required=True)
    parser.add_argument("--test-h5", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wandb-project", default="LearningToTheorize")
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--wandb-group", default="gridworld-observation-pretraining"
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline"),
        default="online",
    )
    return parser


def _config_from_arguments(
    arguments: argparse.Namespace,
) -> GridWorldObservationRunConfig:
    return GridWorldObservationRunConfig(
        train_h5=arguments.train_h5,
        test_h5=arguments.test_h5,
        output_root=arguments.output_root,
        wandb_project=arguments.wandb_project,
        wandb_entity=arguments.wandb_entity,
        wandb_group=arguments.wandb_group,
        wandb_mode=arguments.wandb_mode,
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
