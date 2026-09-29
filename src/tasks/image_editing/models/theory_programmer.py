"""Theory programmer that selects the next latent operation.

The forward pass uses a six-layer MLP without length conditioning. FiLM and
length-embedding parameters are retained for checkpoint compatibility."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tasks.image_editing.models.blocks import FiLMBlock


@dataclass(frozen=True, slots=True)
class LatentProgramConfig:
    """Shape and depth settings shared by the programmer and the executor."""

    state_dim: int = 256
    action_dim: int = 16
    num_state_tokens: int = 1
    num_action_tokens: int = 1
    hidden_dim: int = 128
    num_film_layers: int = 4
    dropout: float = 0.0
    max_transition_length: int = 3
    transition_residual: bool = True

    def __post_init__(self) -> None:
        for name in ("state_dim", "action_dim", "num_state_tokens", "num_action_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.hidden_dim < 1 or self.num_film_layers < 1:
            raise ValueError("hidden_dim and num_film_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.max_transition_length < 1:
            raise ValueError("max_transition_length must be positive")

    @property
    def state_input_dim(self) -> int:
        return self.num_state_tokens * self.state_dim

    @property
    def action_input_dim(self) -> int:
        return self.num_action_tokens * self.action_dim


class TheoryProgrammer(nn.Module):
    """Infer one continuous operation from current and target latent states."""

    def __init__(self, config: LatentProgramConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim

        # Retained for checkpoint and random-number parity; never read in forward.
        self.length_embedding = nn.Embedding(config.max_transition_length + 1, hidden_dim)

        self.input_layer = nn.Sequential(
            nn.Linear(2 * config.state_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.film_blocks = nn.ModuleList(
            FiLMBlock(hidden_dim, hidden_dim, config.dropout)
            for _ in range(config.num_film_layers)
        )
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.action_input_dim),
        )

    def forward(self, current_state: Tensor, target_state: Tensor) -> Tensor:
        self._validate_states(current_state, target_state)
        batch_size = current_state.shape[0]
        combined = torch.cat(
            [
                current_state.reshape(batch_size, -1),
                target_state.reshape(batch_size, -1),
            ],
            dim=-1,
        )
        hidden = self.input_layer(combined)
        for block in self.film_blocks:
            # Unconditioned: the FiLM modulation of each block is inactive.
            hidden = block.transform(hidden)
        actions = self.output_layers(hidden)
        return actions.view(
            batch_size,
            self.config.num_action_tokens,
            self.config.action_dim,
        )

    def _validate_states(self, current_state: Tensor, target_state: Tensor) -> None:
        expected = (self.config.num_state_tokens, self.config.state_dim)
        if current_state.ndim != 3 or tuple(current_state.shape[1:]) != expected:
            raise ValueError(
                f"current_state must be shaped (B, {expected[0]}, {expected[1]}), "
                f"got {tuple(current_state.shape)}"
            )
        if target_state.shape != current_state.shape:
            raise ValueError(
                "target_state must match current_state, got "
                f"{tuple(target_state.shape)} and {tuple(current_state.shape)}"
            )


__all__ = ["LatentProgramConfig", "TheoryProgrammer"]
