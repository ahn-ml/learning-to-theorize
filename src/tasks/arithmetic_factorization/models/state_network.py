import torch
import torch.nn as nn


class DigitEncoder(nn.Module):
    """Embed each digit independently."""

    def __init__(self, params):
        super().__init__()
        self.digits = params.grid_dim
        self.num_classes = params.num_colors
        self.state_dim = params.state_dim
        self.num_state_tokens = params.num_state_tokens
        assert self.num_state_tokens == self.digits, "Number of state tokens must match number of digits"

        self.embedding = nn.Embedding(self.num_classes, self.state_dim)

    def forward(self, grid: torch.Tensor) -> tuple[torch.Tensor, None]:
        """
        Args:
            grid: (batch_size, height, width) - integer grid values

        Returns:
            state: (batch_size, num_state_tokens, state_dim)
            vae_params: None
        """
        return self.embedding(grid.squeeze(1)), None


class DigitDecoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.digits = params.grid_dim
        self.num_classes = params.num_colors
        self.state_dim = params.state_dim
        self.num_state_tokens = params.num_state_tokens
        assert self.num_state_tokens == self.digits, "Number of state tokens must match number of digits"

        self.output_layer = nn.Linear(self.state_dim, self.num_classes)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch_size, num_state_tokens, state_dim)

        Returns:
            output: (batch_size, height, width, num_classes)
        """
        return self.output_layer(state).unsqueeze(1)
