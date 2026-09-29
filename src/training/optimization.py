"""Small, domain-independent optimization and checkpoint contracts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Mapping, Protocol, TypeVar

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


class LossOutput(Protocol):
    """Minimum result contract required by :func:`optimization_step`."""

    loss: Tensor


BatchT = TypeVar("BatchT")
OutputT = TypeVar("OutputT", bound=LossOutput)
Objective = Callable[[nn.Module, BatchT], OutputT]


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    """Optimizer and learning-rate schedule settings shared across domains."""

    total_steps: int
    learning_rate: float
    weight_decay: float
    max_gradient_norm: float
    warmup_ratio: float
    minimum_learning_rate_ratio: float

    def __post_init__(self) -> None:
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.max_gradient_norm <= 0.0:
            raise ValueError("max_gradient_norm must be positive")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 0.0 <= self.minimum_learning_rate_ratio <= 1.0:
            raise ValueError("minimum_learning_rate_ratio must be in [0, 1]")

    @property
    def warmup_steps(self) -> int:
        return int(self.warmup_ratio * self.total_steps)


def cosine_with_warmup_multiplier(
    step: int,
    config: OptimizationConfig,
) -> float:
    """Return the exact warmup/cosine multiplier used by the paper run."""

    if step < config.warmup_steps:
        return step / config.warmup_steps if config.warmup_steps else 1.0
    progress = (step - config.warmup_steps) / max(
        config.total_steps - config.warmup_steps,
        1,
    )
    return config.minimum_learning_rate_ratio + (
        1.0 - config.minimum_learning_rate_ratio
    ) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_adamw_optimizer(
    model: nn.Module,
    config: OptimizationConfig,
) -> AdamW:
    """Build the paper optimizer without adding framework dependencies."""

    return AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def build_program_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    programmer_learning_rate: float,
    executor_learning_rate: float,
    weight_decay: float,
) -> AdamW:
    """Build named programmer, executor and remaining-parameter groups.

    Group and parameter order are preserved when restoring optimizer state.
    Frozen observation parameters are excluded from the remaining group.
    """

    groups = [
        {
            "params": list(model.theory_programmer.parameters()),
            "lr": programmer_learning_rate,
            "name": "theory_programmer",
        },
        {
            "params": list(model.program_executor.parameters()),
            "lr": executor_learning_rate,
            "name": "program_executor",
        },
    ]
    remaining = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith(("theory_programmer.", "program_executor."))
        and parameter.requires_grad
    ]
    if remaining:
        groups.append({"params": remaining, "lr": learning_rate, "name": "other"})
    return AdamW(groups, weight_decay=weight_decay)


def build_cosine_warmup_scheduler(
    optimizer: Optimizer,
    config: OptimizationConfig,
) -> LambdaLR:
    """Build a scheduler that advances after each optimizer update."""

    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_with_warmup_multiplier(step, config),
    )


@dataclass(frozen=True, slots=True)
class OptimizationStep(Generic[OutputT]):
    """Result and state after one optimizer update."""

    objective: OutputT
    gradient_norm: Tensor
    global_step: int
    learning_rates: tuple[float, ...]


def optimization_step(
    model: nn.Module,
    batch: BatchT,
    objective: Objective[BatchT, OutputT],
    optimizer: Optimizer,
    scheduler: LRScheduler,
    *,
    global_step: int,
    max_gradient_norm: float,
) -> OptimizationStep[OutputT]:
    """Apply backward, gradient clipping, optimizer step, clear, and schedule."""

    model.train()
    output = objective(model, batch)
    output.loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
    optimizer.step()
    # PyTorch 2.7 defaults to set_to_none=True. Keep it explicit so later
    # versions cannot silently change the paper update contract.
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    return OptimizationStep(
        objective=output,
        gradient_norm=gradient_norm,
        global_step=global_step + 1,
        learning_rates=tuple(scheduler.get_last_lr()),
    )


@dataclass(frozen=True, slots=True)
class EpochTrainingState:
    """Unambiguous resume state for a release checkpoint."""

    completed_epochs: int
    global_step: int

    def __post_init__(self) -> None:
        if self.completed_epochs < 0 or self.global_step < 0:
            raise ValueError("completed_epochs and global_step must be non-negative")


CheckpointScalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class LoadedTrainingCheckpoint:
    """Training state and provenance restored from a release checkpoint."""

    state: EpochTrainingState
    optimization: OptimizationConfig
    metadata: dict[str, CheckpointScalar]


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    state: EpochTrainingState,
    optimization: OptimizationConfig,
    metadata: Mapping[str, CheckpointScalar] | None = None,
) -> None:
    """Save a weights-only-loadable checkpoint without overwriting a run."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_state_dict": _unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "training_state": asdict(state),
        "optimization": asdict(optimization),
        "metadata": dict(metadata or {}),
    }
    with checkpoint_path.open("xb") as file:
        torch.save(payload, file)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    map_location: str | torch.device = "cpu",
) -> LoadedTrainingCheckpoint:
    """Strictly restore a release checkpoint using PyTorch's safe loader."""

    payload: Any = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported training checkpoint format")
    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "training_state",
        "optimization",
        "metadata",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"training checkpoint is missing fields: {missing}")
    _unwrap_model(model).load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    return LoadedTrainingCheckpoint(
        state=EpochTrainingState(**payload["training_state"]),
        optimization=OptimizationConfig(**payload["optimization"]),
        metadata=dict(payload["metadata"]),
    )


def _unwrap_model(model: nn.Module) -> nn.Module:
    candidate = getattr(model, "module", model)
    return candidate if isinstance(candidate, nn.Module) else model
