"""A digit embedding and linear readout trained by reconstruction alone."""

import torch
from torch import Tensor, nn
from omegaconf import OmegaConf

from tasks.arithmetic_factorization.models.adapters import ObservationEncoder, ObservationDecoder
from tasks.arithmetic_factorization.objective import digit_cross_entropy
from tasks.arithmetic_factorization.observation_pretraining import ArithmeticObservationPretrainingConfig


class ArithmeticObservationModel(nn.Module):
    def __init__(self, config: ArithmeticObservationPretrainingConfig) -> None:
        super().__init__()
        parameters = OmegaConf.create({
            "grid_dim": config.num_digits, "num_colors": config.num_symbols,
            "state_dim": config.state_dim, "num_state_tokens": config.num_digits,
        })
        self.encoder = ObservationEncoder(parameters)
        self.decoder = ObservationDecoder(parameters)

    def forward(self, observations: Tensor) -> dict[str, Tensor]:
        if observations.ndim != 3 or observations.shape[1] != 1:
            raise ValueError("expected individual numbers with shape (batch, 1, digits)")
        states, _ = self.encoder(observations)
        logits = self.decoder(states)
        correct = logits.argmax(-1) == observations
        return {
            "loss": digit_cross_entropy(logits, observations),
            "digit_correct": correct.sum(),
            "number_correct": correct.flatten(1).all(1).sum(),
        }
