"""Latent program execution for one quantized operation.

Unlike the theory programmer, the executor really is FiLM-conditioned: the
action embedding modulates every one of the four blocks, which is what makes a
primitive's effect depend on the primitive rather than on the state alone.
"""

from __future__ import annotations

from torch import Tensor, nn

from tasks.image_editing.models.blocks import FiLMBlock
from tasks.image_editing.models.theory_programmer import LatentProgramConfig


class ProgramExecutor(nn.Module):
    """Apply one FiLM-conditioned residual operation in latent state space."""

    def __init__(self, config: LatentProgramConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim

        self.state_layer = nn.Sequential(
            nn.Linear(config.state_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.action_proj = nn.Linear(config.action_input_dim, hidden_dim)
        self.film_blocks = nn.ModuleList(
            FiLMBlock(hidden_dim, hidden_dim, config.dropout)
            for _ in range(config.num_film_layers)
        )
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.state_input_dim),
        )

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        self._validate(state, action)
        batch_size = state.shape[0]
        hidden = self.state_layer(state.reshape(batch_size, -1))
        action_embedding = self.action_proj(action.reshape(batch_size, -1))
        for block in self.film_blocks:
            hidden = block(hidden, action_embedding)
        delta = self.output_layers(hidden).view(
            batch_size,
            self.config.num_state_tokens,
            self.config.state_dim,
        )
        if self.config.transition_residual:
            return state + delta
        return delta

    def _validate(self, state: Tensor, action: Tensor) -> None:
        expected_state = (self.config.num_state_tokens, self.config.state_dim)
        expected_action = (self.config.num_action_tokens, self.config.action_dim)
        if state.ndim != 3 or tuple(state.shape[1:]) != expected_state:
            raise ValueError(
                f"state must be shaped (B, {expected_state[0]}, {expected_state[1]}), "
                f"got {tuple(state.shape)}"
            )
        if action.ndim != 3 or tuple(action.shape[1:]) != expected_action:
            raise ValueError(
                f"action must be shaped (B, {expected_action[0]}, {expected_action[1]}), "
                f"got {tuple(action.shape)}"
            )
        if action.shape[0] != state.shape[0]:
            raise ValueError("state and action batch sizes must match")


__all__ = ["ProgramExecutor"]
