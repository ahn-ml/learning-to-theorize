"""Latent program execution for one quantized operation."""

from __future__ import annotations

from torch import Tensor, nn

from models.config import LatentProgramConfig


class ProgramExecutor(nn.Module):
    """Apply one FiLM-conditioned residual operation in latent state space."""

    def __init__(
        self,
        config: LatentProgramConfig,
    ) -> None:
        super().__init__()
        self.config = config
        hidden_dim = self.config.transition_hidden_dim
        state_input_dim = self.config.num_state_tokens * self.config.state_dim
        action_input_dim = self.config.num_action_tokens * self.config.action_dim

        self.state_layer = nn.Sequential(
            nn.Linear(state_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
        )
        self.action_proj = nn.Linear(action_input_dim, hidden_dim)
        self.film_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.film_beta = nn.Linear(hidden_dim, hidden_dim)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_dim, state_input_dim),
        )

        nn.init.ones_(self.film_gamma.weight.data.diagonal())
        nn.init.zeros_(self.film_gamma.bias.data)
        nn.init.zeros_(self.film_beta.weight.data)
        nn.init.zeros_(self.film_beta.bias.data)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        batch_size = state.shape[0]
        hidden = self.state_layer(state.reshape(batch_size, -1))
        action_embedding = self.action_proj(action.reshape(batch_size, -1))
        hidden = (
            self.film_gamma(action_embedding) * hidden
            + self.film_beta(action_embedding)
        )
        delta = self.output_layers(hidden).view(
            batch_size,
            self.config.num_state_tokens,
            self.config.state_dim,
        )
        if self.config.transition_residual:
            return state + delta
        return delta


__all__ = ["ProgramExecutor"]
