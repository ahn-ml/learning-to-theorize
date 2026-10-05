"""Recipe and objective for GridWorld observation pretraining."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from training import OptimizationConfig


@dataclass(frozen=True, slots=True)
class GridWorldObservationPretrainingConfig:
    """GridWorld observation pretraining settings; the defaults are the release recipe."""

    seed: int = 42
    epochs: int = 50
    train_episodes: int = 500_000
    test_episodes: int = 5_000
    per_rank_batch_size: int = 512
    effective_world_size: int = 4
    data_loader_workers: int = 4
    log_interval_steps: int = 100
    learning_rate: float = 0.0007
    weight_decay: float = 0.01
    max_gradient_norm: float = 1.0
    warmup_ratio: float = 0.1
    minimum_learning_rate_ratio: float = 0.005
    kl_weight: float = 1e-5

    def __post_init__(self) -> None:
        positive_integer_fields = (
            self.epochs,
            self.train_episodes,
            self.test_episodes,
            self.per_rank_batch_size,
            self.effective_world_size,
            self.log_interval_steps,
        )
        if any(value < 1 for value in positive_integer_fields):
            raise ValueError("pretraining counts must be positive")
        if self.seed < 0 or self.data_loader_workers < 0:
            raise ValueError("seed and data_loader_workers must be non-negative")
        if self.kl_weight < 0.0:
            raise ValueError("kl_weight must be non-negative")

    def steps_per_epoch(self, num_episodes: int) -> int:
        """Match a non-dropping DistributedSampler and per-rank DataLoader."""

        if num_episodes < 1:
            raise ValueError("num_episodes must be positive")
        samples_per_rank = math.ceil(num_episodes / self.effective_world_size)
        return math.ceil(samples_per_rank / self.per_rank_batch_size)

    @property
    def train_steps_per_epoch(self) -> int:
        return self.steps_per_epoch(self.train_episodes)

    @property
    def test_steps_per_epoch(self) -> int:
        return self.steps_per_epoch(self.test_episodes)

    @property
    def total_steps(self) -> int:
        return self.epochs * self.train_steps_per_epoch

    def optimization(self) -> OptimizationConfig:
        return OptimizationConfig(
            total_steps=self.total_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_gradient_norm=self.max_gradient_norm,
            warmup_ratio=self.warmup_ratio,
            minimum_learning_rate_ratio=self.minimum_learning_rate_ratio,
        )


def flatten_episode_grids(episode_grids: Tensor) -> Tensor:
    """Flatten all input grids followed by all output grids."""

    if episode_grids.ndim != 4 or tuple(episode_grids.shape[1:]) != (4, 10, 10):
        raise ValueError(
            "expected episode grids shaped (B, 4, 10, 10), "
            f"got {tuple(episode_grids.shape)}"
        )
    if episode_grids.is_floating_point() or episode_grids.is_complex():
        raise TypeError("GridWorld observations must contain integer color indices")
    inputs = episode_grids[:, ::2].reshape(-1, 10, 10)
    outputs = episode_grids[:, 1::2].reshape(-1, 10, 10)
    return torch.cat([inputs, outputs], dim=0)


@dataclass(frozen=True, slots=True)
class GridWorldObservationObjectiveOutput:
    """Loss and exact reconstruction metrics for one flattened batch."""

    loss: Tensor
    reconstruction_loss: Tensor
    kl_loss: Tensor
    pixel_accuracy: Tensor
    grid_accuracy: Tensor
    num_observations: int


@dataclass(frozen=True, slots=True)
class GridWorldObservationObjective:
    """VAE objective: grid reconstruction plus a weighted KL term."""

    kl_weight: float

    def __post_init__(self) -> None:
        if self.kl_weight < 0.0:
            raise ValueError("kl_weight must be non-negative")

    def __call__(
        self,
        model: nn.Module,
        episode_grids: Tensor,
    ) -> GridWorldObservationObjectiveOutput:
        observations = flatten_episode_grids(episode_grids)
        model_output = model(observations)
        reconstruction_loss = F.cross_entropy(
            model_output.logits.reshape(-1, model_output.logits.shape[-1]),
            observations.reshape(-1).long(),
            reduction="mean",
        )
        kl_loss = model_output.posterior.kl_to_standard_normal()
        loss = reconstruction_loss + self.kl_weight * kl_loss
        with torch.no_grad():
            predictions = model_output.logits.argmax(dim=-1)
            correct = predictions.eq(observations)
            pixel_accuracy = correct.float().mean()
            grid_accuracy = correct.flatten(1).all(dim=1).float().mean()
        return GridWorldObservationObjectiveOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kl_loss=kl_loss,
            pixel_accuracy=pixel_accuracy,
            grid_accuracy=grid_accuracy,
            num_observations=observations.shape[0],
        )
