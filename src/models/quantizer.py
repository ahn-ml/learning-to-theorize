"""Vector quantization of latent operations into reusable primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.config import LatentProgramConfig


@dataclass(frozen=True, slots=True)
class ActionQuantizerOutput:
    """Quantized operations and values consumed by the paper objective."""

    values: Tensor
    loss: Tensor
    loss_per_sample: Tensor
    commitment_loss: Tensor
    codebook_loss: Tensor
    temperature: Tensor
    indices: Tensor
    logits: Tensor
    # Set only by the Arithmetic adapter.
    orthogonal_loss: Tensor | None = None
    entropy_loss: Tensor | None = None
    sample_entropy: Tensor | None = None
    codebook_entropy: Tensor | None = None


class ActionQuantizer(nn.Module):
    """Six-code stochastic vector quantizer used by every alpha setting."""

    def __init__(self, config: LatentProgramConfig) -> None:
        super().__init__()
        self.config = config
        self.commitment_cost = self.config.action_commitment_weight
        self.current_step = 0
        self.tau_start = self.config.action_tau_start
        self.tau_end = self.config.action_tau_end
        self.tau_steps = self.config.action_tau_steps

        self.embedding = nn.Embedding(
            self.config.action_codebook_size,
            self.config.action_dim,
        )
        self.embedding.weight.data.uniform_(
            -1.0 / self.config.action_codebook_size,
            1.0 / self.config.action_codebook_size,
        )

    def get_temperature(self) -> float:
        if self.current_step >= self.tau_steps:
            return self.tau_end
        cosine = math.cos(math.pi * self.current_step / self.tau_steps)
        return self.tau_end + 0.5 * (self.tau_start - self.tau_end) * (
            1.0 + cosine
        )

    def step(self) -> None:
        self.current_step += 1

    def forward(
        self,
        actions: Tensor,
        *,
        training_mode: bool = True,
    ) -> ActionQuantizerOutput:
        expected = (
            self.config.num_action_tokens,
            self.config.action_dim,
        )
        if actions.ndim != 3 or tuple(actions.shape[1:]) != expected:
            raise ValueError(
                f"actions must be shaped (B, {expected[0]}, {expected[1]}), "
                f"got {tuple(actions.shape)}"
            )

        with torch.autocast(device_type=actions.device.type, enabled=False):
            flattened = F.normalize(actions.flatten(end_dim=-2).float(), dim=-1)
            embedding = F.normalize(self.embedding.weight, dim=-1)
            distances = (
                flattened.square().sum(dim=1, keepdim=True)
                + embedding.square().sum(dim=1)
                - 2 * torch.einsum("bd,dn->bn", flattened, embedding.T)
            )

            current_temperature = self.get_temperature()
            if training_mode:
                logits = -distances / current_temperature
                probabilities = F.softmax(logits, dim=-1)
                indices = torch.multinomial(probabilities, 1).squeeze(-1)
            else:
                logits = -distances
                probabilities = F.softmax(logits, dim=-1)
                indices = probabilities.argmax(dim=-1)

            quantized = self.get_codebook_entry(indices).view(actions.shape)
            quantized = F.normalize(quantized, dim=-1)
            normalized_actions = F.normalize(actions, dim=-1)
            commitment_loss = self.commitment_cost * (
                quantized.detach() - normalized_actions
            ).square().mean()
            commitment_loss_per_sample = self.commitment_cost * (
                quantized.detach() - normalized_actions
            ).square().mean(dim=(1, 2))
            codebook_loss = (
                quantized - normalized_actions.detach()
            ).square().mean()
            codebook_loss_per_sample = (
                quantized - normalized_actions.detach()
            ).square().mean(dim=(1, 2))

            loss = commitment_loss + codebook_loss
            loss_per_sample = commitment_loss_per_sample + codebook_loss_per_sample
            straight_through = normalized_actions + (
                quantized - normalized_actions
            ).detach()

        return ActionQuantizerOutput(
            values=straight_through,
            loss=loss,
            loss_per_sample=loss_per_sample,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            temperature=torch.tensor(current_temperature),
            indices=indices,
            logits=logits,
        )

    def get_codebook_entry(self, indices: Tensor) -> Tensor:
        if indices.ndim != 1:
            raise ValueError("indices must be a rank-one tensor")
        return F.normalize(self.embedding(indices), dim=-1)


__all__ = [
    "ActionQuantizer",
    "ActionQuantizerOutput",
]
