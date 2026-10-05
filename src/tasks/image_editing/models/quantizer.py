"""Map continuous operations onto a codebook of reusable discrete primitives.

Operations are L2-normalized and assigned to their nearest code; the codebook is
updated by an exponential moving average, and an entropy term encourages
confident, diverse code use.

Derived from the VQ implementation in taming-transformers / MAGVIT via the
paper-producing codebase (Apache-2.0, Bytedance Ltd.).
Copyright (2024) Bytedance Ltd. and/or its affiliates.
Includes modifications by the Learning to Theorize authors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.codebook_ema import CodebookEMA


@dataclass(frozen=True, slots=True)
class ActionQuantizerConfig:
    """Codebook settings for the Image Editing action space."""

    codebook_size: int = 16
    action_dim: int = 16
    commitment_weight: float = 0.25
    entropy_weight: float = 0.1
    entropy_temperature: float = 1.0
    ema_decay: float = 0.999
    ema_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if self.codebook_size < 1 or self.action_dim < 1:
            raise ValueError("codebook_size and action_dim must be positive")


@dataclass(frozen=True, slots=True)
class ActionQuantizerOutput:
    """Quantized operations and the losses consumed by the paper objective."""

    values: Tensor
    loss: Tensor
    loss_per_sample: Tensor
    indices: Tensor


class ActionQuantizer(nn.Module):
    """Quantize continuous operations against an EMA-updated codebook."""

    def __init__(self, config: ActionQuantizerConfig) -> None:
        super().__init__()
        self.config = config
        self._ema = CodebookEMA()

        self.embedding = nn.Embedding(config.codebook_size, config.action_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / config.codebook_size, 1.0 / config.codebook_size
        )
        self.register_buffer("ema_cluster_size", torch.zeros(config.codebook_size))
        self.register_buffer("ema_weight", self.embedding.weight.data.clone())
        # EMA replaces the codebook gradient entirely.
        self.embedding.requires_grad_(False)

    def step(self) -> None:
        """Commit pooled EMA statistics after the optimizer update, on all ranks."""
        self._ema.update(self.embedding.weight, self.ema_cluster_size,
                         self.ema_weight, self.config.ema_decay,
                         self.config.ema_epsilon)

    def _entropy_loss(self, affinity: Tensor) -> Tensor:
        flat = affinity.view(-1, affinity.shape[-1]) / self.config.entropy_temperature
        probabilities = F.softmax(flat, dim=-1)
        log_probabilities = F.log_softmax(flat + 1e-5, dim=-1)
        average = torch.mean(probabilities, dim=0)
        codebook_entropy = -torch.sum(average * torch.log(average + 1e-5))
        sample_entropy = -torch.mean(
            torch.sum(probabilities * log_probabilities, dim=-1)
        )
        return sample_entropy - codebook_entropy

    def get_codebook_entry(self, indices: Tensor) -> Tensor:
        """Look up normalized codebook vectors for integer indices."""

        if indices.ndim != 1:
            raise ValueError(f"indices must be 1-dimensional, got {tuple(indices.shape)}")
        return F.normalize(self.embedding(indices), dim=-1)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        actions: Tensor,
        *,
        training_mode: bool = True,
    ) -> ActionQuantizerOutput:
        """Quantize one operation. Signature matches ``models.quantizer``."""

        config = self.config
        flattened = F.normalize(actions.flatten(end_dim=-2).float(), dim=-1)
        codebook = F.normalize(self.embedding.weight, dim=-1)

        distances = (
            torch.sum(flattened**2, dim=1, keepdim=True)
            + torch.sum(codebook**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", flattened, codebook.T)
        )
        indices = torch.argmin(distances, dim=1)

        if self.training and training_mode:
            self._ema.accumulate(flattened, indices, config.codebook_size)

        quantized = F.normalize(self.get_codebook_entry(indices).view(actions.shape), dim=-1)
        actions = F.normalize(actions, dim=-1)

        # Both reductions train the programmer toward the selected code; the
        # codebook itself is updated only by the EMA.
        commitment_loss = config.commitment_weight * torch.mean(
            (quantized.detach() - actions) ** 2
        )
        commitment_loss_per_sample = config.commitment_weight * torch.mean(
            (quantized.detach() - actions) ** 2, dim=(1, 2)
        )
        entropy_loss = config.entropy_weight * self._entropy_loss(-distances)

        return ActionQuantizerOutput(
            # Straight-through: gradients flow to the programmer, not the codebook.
            values=actions + (quantized - actions).detach(),
            loss=commitment_loss + entropy_loss,
            loss_per_sample=commitment_loss_per_sample + entropy_loss,
            indices=indices,
        )


__all__ = [
    "ActionQuantizer",
    "ActionQuantizerConfig",
    "ActionQuantizerOutput",
]
