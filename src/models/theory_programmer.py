"""Theory programmer that selects the next latent operation."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from models.config import LatentProgramConfig


class TheoryProgrammer(nn.Module):
    """Infer one continuous operation from current and target latent states."""

    def __init__(
        self,
        config: LatentProgramConfig,
    ) -> None:
        super().__init__()
        self.config = config
        hidden_dim = self.config.policy_hidden_dim
        input_dim = 2 * self.config.num_state_tokens * self.config.state_dim
        output_dim = self.config.num_action_tokens * self.config.action_dim

        # The length embedding and FiLM layers are never read in forward; they
        # are kept for checkpoint keys and initialization (random-number) parity.
        self.length_embedding = nn.Embedding(
            self.config.max_transition_length + 1,
            hidden_dim,
        )
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
        )
        self.film_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.film_beta = nn.Linear(hidden_dim, hidden_dim)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Match the paper-producing FiLM initialization exactly.
        nn.init.ones_(self.film_gamma.weight.data.diagonal())
        nn.init.zeros_(self.film_gamma.bias.data)
        nn.init.zeros_(self.film_beta.weight.data)
        nn.init.zeros_(self.film_beta.bias.data)

    def forward(self, current_state: Tensor, target_state: Tensor) -> Tensor:
        batch_size = current_state.shape[0]
        combined = torch.cat(
            [
                current_state.reshape(batch_size, -1),
                target_state.reshape(batch_size, -1),
            ],
            dim=-1,
        )
        actions = self.output_layers(self.input_layer(combined))
        return actions.view(
            batch_size,
            self.config.num_action_tokens,
            self.config.action_dim,
        )


__all__ = ["TheoryProgrammer"]
