"""Map continuous operations onto a codebook of reusable discrete primitives.

Supports a learned or EMA-updated codebook and categorical sampling with a
cosine temperature schedule.

Derived from the VQ implementation in taming-transformers / MAGVIT via the
paper-producing codebase (Apache-2.0, Bytedance Ltd.).
Copyright (2024) Bytedance Ltd. and/or its affiliates.
Includes modifications by the Learning to Theorize authors.
"""

from __future__ import annotations

import math
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
    l2_normalize: bool = True
    stochastic: bool = False
    tau_start: float = 2.0
    tau_end: float = 0.1
    tau_steps: int = 3000
    entropy_weight: float = 0.1
    entropy_temperature: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if self.codebook_size < 1 or self.action_dim < 1:
            raise ValueError("codebook_size and action_dim must be positive")
        if self.tau_steps < 1:
            raise ValueError("tau_steps must be positive")


@dataclass(frozen=True, slots=True)
class ActionQuantizerOutput:
    """Quantized operations and values consumed by the paper objective."""

    values: Tensor
    loss: Tensor
    loss_per_sample: Tensor
    commitment_loss: Tensor
    codebook_loss: Tensor
    entropy_loss: Tensor
    sample_entropy: Tensor
    codebook_entropy: Tensor
    temperature: Tensor
    indices: Tensor
    logits: Tensor


class ActionQuantizer(nn.Module):
    """Quantize continuous operations against a learned or EMA-updated codebook."""

    def __init__(self, config: ActionQuantizerConfig) -> None:
        super().__init__()
        self.config = config
        self.current_step = 0
        self._ema = CodebookEMA()

        self.embedding = nn.Embedding(config.codebook_size, config.action_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / config.codebook_size, 1.0 / config.codebook_size
        )
        if config.use_ema:
            self.register_buffer("ema_cluster_size", torch.zeros(config.codebook_size))
            self.register_buffer("ema_weight", self.embedding.weight.data.clone())
            # EMA replaces the codebook gradient entirely.
            self.embedding.requires_grad_(False)

    def get_temperature(self) -> float:
        """Cosine-annealed categorical sampling temperature."""

        if self.current_step >= self.config.tau_steps:
            return self.config.tau_end
        cosine = math.cos(math.pi * self.current_step / self.config.tau_steps)
        return self.config.tau_end + 0.5 * (
            self.config.tau_start - self.config.tau_end
        ) * (1 + cosine)

    def step(self) -> None:
        """Commit pooled EMA statistics after the optimizer update, on all ranks."""
        if self.config.use_ema:
            self._ema.update(self.embedding.weight, self.ema_cluster_size,
                             self.ema_weight, self.config.ema_decay,
                             self.config.ema_epsilon)
        self.current_step += 1

    def _entropy_loss(self, affinity: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        flat = affinity.view(-1, affinity.shape[-1]) / self.config.entropy_temperature
        probabilities = F.softmax(flat, dim=-1)
        log_probabilities = F.log_softmax(flat + 1e-5, dim=-1)
        average = torch.mean(probabilities, dim=0)
        codebook_entropy = -torch.sum(average * torch.log(average + 1e-5))
        sample_entropy = -torch.mean(
            torch.sum(probabilities * log_probabilities, dim=-1)
        )
        return sample_entropy - codebook_entropy, sample_entropy, codebook_entropy

    def get_codebook_entry(self, indices: Tensor) -> Tensor:
        """Look up codebook vectors for integer indices."""

        if indices.ndim != 1:
            raise ValueError(f"indices must be 1-dimensional, got {tuple(indices.shape)}")
        entries = self.embedding(indices)
        if self.config.l2_normalize:
            entries = F.normalize(entries, dim=-1)
        return entries

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        actions: Tensor,
        *,
        training_mode: bool = True,
    ) -> ActionQuantizerOutput:
        """Quantize one operation. Signature matches ``models.quantizer``."""

        config = self.config
        is_eval = not training_mode
        flattened = actions.flatten(end_dim=-2).float()
        if config.l2_normalize:
            flattened = F.normalize(flattened, dim=-1)
            codebook = F.normalize(self.embedding.weight, dim=-1)
        else:
            codebook = self.embedding.weight

        distances = (
            torch.sum(flattened**2, dim=1, keepdim=True)
            + torch.sum(codebook**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", flattened, codebook.T)
        )

        if config.stochastic and config.l2_normalize:
            temperature = self.get_temperature()
            if is_eval:
                logits = -distances
                indices = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            else:
                logits = -distances / temperature
                indices = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
        else:
            temperature = 0.0
            logits = None
            indices = torch.argmin(distances, dim=1)

        if config.use_ema and self.training and training_mode:
            self._ema.accumulate(flattened, indices, config.codebook_size)

        quantized = self.get_codebook_entry(indices).view(actions.shape)
        if config.l2_normalize:
            quantized = F.normalize(quantized, dim=-1)
            actions = F.normalize(actions, dim=-1)

        # Both reductions train the programmer toward the selected code. The
        # codebook has its own loss (or EMA); detaching actions here would remove
        # the commitment gradient from the per-sample loss used by NEO.
        commitment_loss = config.commitment_weight * torch.mean(
            (quantized.detach() - actions) ** 2
        )
        commitment_loss_per_sample = config.commitment_weight * torch.mean(
            (quantized.detach() - actions) ** 2, dim=(1, 2)
        )
        codebook_loss = torch.mean((quantized - actions.detach()) ** 2)
        codebook_loss_per_sample = torch.mean(
            (quantized - actions.detach()) ** 2, dim=(1, 2)
        )
        if config.use_ema:
            codebook_loss = 0.0 * codebook_loss
            codebook_loss_per_sample = 0.0 * codebook_loss_per_sample

        if config.entropy_weight != 0:
            raw_entropy_loss, sample_entropy, codebook_entropy = self._entropy_loss(
                -distances
            )
            entropy_loss = config.entropy_weight * raw_entropy_loss
        else:
            zero = torch.tensor(0.0, device=actions.device, dtype=actions.dtype)
            entropy_loss, sample_entropy, codebook_entropy = zero, zero, zero

        return ActionQuantizerOutput(
            # Straight-through: gradients flow to the programmer, not the codebook.
            values=actions + (quantized - actions).detach(),
            loss=commitment_loss + codebook_loss + entropy_loss,
            loss_per_sample=commitment_loss_per_sample
            + codebook_loss_per_sample
            + entropy_loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            entropy_loss=entropy_loss,
            sample_entropy=sample_entropy,
            codebook_entropy=codebook_entropy,
            temperature=torch.tensor(temperature),
            indices=indices,
            logits=1 / distances if logits is None else logits,
        )


__all__ = [
    "ActionQuantizer",
    "ActionQuantizerConfig",
    "ActionQuantizerOutput",
]
