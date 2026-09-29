"""Adapt the released arithmetic modules to the shared NEO interfaces."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.quantizer import ActionQuantizerOutput
from tasks.arithmetic_factorization.models.quantizer import VectorQuantizer
from tasks.arithmetic_factorization.models.state_network import (
    DigitDecoder,
    DigitEncoder,
)


@dataclass(frozen=True, slots=True)
class DeterministicPosterior:
    """Stand-in posterior for a deterministic observation encoder.

    The arithmetic encoder is a plain digit embedding, so there is no
    observation KL term. The shared rollout reports the value but never adds
    it to the loss, so a zero keeps the reported metric honest.
    """

    reference: Tensor

    def kl_to_standard_normal(self) -> Tensor:
        return torch.zeros(
            (), device=self.reference.device, dtype=self.reference.dtype
        )


class ObservationEncoder(nn.Module):
    """Digit encoder that reports a deterministic posterior to shared NEO."""

    def __init__(self, parameters) -> None:
        super().__init__()
        self.embedding_encoder = DigitEncoder(parameters)

    @property
    def embedding(self) -> nn.Embedding:
        """Expose the released parameter name for checkpoint compatibility."""

        return self.embedding_encoder.embedding

    def forward(self, grid: Tensor) -> tuple[Tensor, DeterministicPosterior]:
        states, _ = self.embedding_encoder(grid)
        return states, DeterministicPosterior(states)


class ObservationDecoder(nn.Module):
    """Digit decoder wrapper kept symmetric with the encoder adapter."""

    def __init__(self, parameters) -> None:
        super().__init__()
        self.digit_decoder = DigitDecoder(parameters)

    @property
    def output_layer(self) -> nn.Linear:
        """Expose the released parameter name for checkpoint compatibility."""

        return self.digit_decoder.output_layer

    def forward(self, state: Tensor) -> Tensor:
        return self.digit_decoder(state)


class ArithmeticActionQuantizer(nn.Module):
    """Released EMA vector quantizer exposed as a shared quantizer."""

    def __init__(self, quantizer: VectorQuantizer) -> None:
        super().__init__()
        self.quantizer = quantizer

    def get_temperature(self) -> float:
        return self.quantizer.get_temperature()

    def step(self) -> None:
        self.quantizer.step()

    def get_codebook_entry(self, indices: Tensor) -> Tensor:
        return self.quantizer.get_codebook_entry(indices)

    def forward(
        self, actions: Tensor, *, training_mode: bool = True
    ) -> ActionQuantizerOutput:
        values, result = self.quantizer(actions, training_mode=training_mode)
        return ActionQuantizerOutput(
            values=values,
            loss=result["quantizer_loss"],
            loss_per_sample=result["quantizer_loss_per_sample"],
            commitment_loss=result["commitment_loss"],
            codebook_loss=result["codebook_loss"],
            entropy_loss=result["entropy_loss"],
            sample_entropy=result["sample_entropy"],
            codebook_entropy=result["codebook_entropy"],
            temperature=result["temperature"],
            indices=result["min_encoding_indices"],
            logits=result["logit"],
            orthogonal_loss=result["orthogonal_loss"],
        )


__all__ = [
    "ArithmeticActionQuantizer",
    "DeterministicPosterior",
    "ObservationDecoder",
    "ObservationEncoder",
]
