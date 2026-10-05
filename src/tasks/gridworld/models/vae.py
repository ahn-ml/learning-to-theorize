"""Variational autoencoder for GridWorld observations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class VAEConfig:
    """Architecture settings for the paper's 10x10 GridWorld VAE."""

    grid_size: int = 10
    num_colors: int = 9
    state_dim: int = 32
    num_state_tokens: int = 1
    dropout: float = 0.1
    sample_posterior: bool = True

    def __post_init__(self) -> None:
        if self.grid_size != 10:
            raise ValueError("the paper GridWorld CNN supports exactly 10x10 grids")
        if self.num_colors < 1:
            raise ValueError("num_colors must be positive")
        if self.state_dim < 1 or self.num_state_tokens < 1:
            raise ValueError("state_dim and num_state_tokens must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class GaussianPosterior:
    """Diagonal Gaussian returned by the variational encoder."""

    mean: Tensor
    log_variance: Tensor

    def sample(self) -> Tensor:
        standard_deviation = torch.exp(0.5 * self.log_variance)
        return self.mean + torch.randn_like(standard_deviation) * standard_deviation

    def kl_to_standard_normal(self) -> Tensor:
        return -0.5 * torch.sum(
            1 + self.log_variance - self.mean.square() - self.log_variance.exp()
        ) / self.mean.shape[0]


class Encoder(nn.Module):
    """Encode an integer 10x10 grid into latent state tokens."""

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config
        embedding_dim = 16
        self.embedding = nn.Embedding(config.num_colors, embedding_dim)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(embedding_dim, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(config.dropout),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(config.dropout),
            nn.Conv2d(128, 256, 3, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(config.dropout),
        )
        flattened_dim = 256 * 2 * 2
        output_dim = config.num_state_tokens * config.state_dim

        # Unused by forward; it is part of the checkpoint state and of the
        # parameter initialization order.
        self.to_state = nn.Linear(flattened_dim, output_dim)
        self.to_mean = nn.Linear(flattened_dim, output_dim)
        self.to_logvar = nn.Linear(flattened_dim, output_dim)

    def forward(self, grid: Tensor) -> tuple[Tensor, GaussianPosterior]:
        self._validate_grid(grid)
        batch_size = grid.shape[0]
        embedded = self.embedding(grid)
        features = self.conv_layers(embedded.permute(0, 3, 1, 2)).reshape(
            batch_size, -1
        )

        posterior = GaussianPosterior(
            mean=self.to_mean(features).view(
                batch_size,
                self.config.num_state_tokens,
                self.config.state_dim,
            ),
            log_variance=self.to_logvar(features).view(
                batch_size,
                self.config.num_state_tokens,
                self.config.state_dim,
            ),
        )
        state = posterior.sample() if self.config.sample_posterior else posterior.mean
        return state, posterior

    def _validate_grid(self, grid: Tensor) -> None:
        expected_shape = (self.config.grid_size, self.config.grid_size)
        if grid.ndim != 3 or tuple(grid.shape[1:]) != expected_shape:
            raise ValueError(
                f"expected integer grids shaped (B, 10, 10), got {tuple(grid.shape)}"
            )
        if grid.is_floating_point() or grid.is_complex():
            raise TypeError("grids must contain integer color indices")


class Decoder(nn.Module):
    """Decode latent state tokens to per-cell color logits."""

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config
        flattened_dim = 256 * 2 * 2
        self.from_state = nn.Linear(
            config.num_state_tokens * config.state_dim,
            flattened_dim,
        )
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(config.dropout),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(config.dropout),
            nn.Conv2d(64, config.num_colors, 3, padding=1),
        )

    def forward(self, state: Tensor) -> Tensor:
        expected_shape = (self.config.num_state_tokens, self.config.state_dim)
        if state.ndim != 3 or tuple(state.shape[1:]) != expected_shape:
            raise ValueError(
                f"expected state shaped (B, {expected_shape[0]}, {expected_shape[1]}), "
                f"got {tuple(state.shape)}"
            )
        batch_size = state.shape[0]
        features = state.view(batch_size, -1)
        features = self.from_state(features)
        features = features.view(batch_size, 256, 2, 2)
        features = self.deconv_layers(features)
        return features.permute(0, 2, 3, 1)


@dataclass(frozen=True, slots=True)
class VAEOutput:
    logits: Tensor
    state: Tensor
    posterior: GaussianPosterior


class VAE(nn.Module):
    """GridWorld encoder and decoder pretrained before theory learning."""

    def __init__(self, config: VAEConfig | None = None) -> None:
        super().__init__()
        self.config = config or VAEConfig()
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

    def forward(self, grid: Tensor) -> VAEOutput:
        state, posterior = self.encoder(grid)
        return VAEOutput(
            logits=self.decoder(state),
            state=state,
            posterior=posterior,
        )


__all__ = [
    "Decoder",
    "Encoder",
    "GaussianPosterior",
    "VAE",
    "VAEConfig",
    "VAEOutput",
]
