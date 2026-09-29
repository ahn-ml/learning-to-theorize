"""ImageEditing observation VAE for NEO.

Four residual encoder stages produce 4x4x512 features; three residual decoder
stages reconstruct 32x32 images. Checkpoints use ``encoder.*`` and
``decoder.*`` parameter names. A variational encoder uses ``to_mean`` and
``to_logvar`` projections; a deterministic encoder uses ``to_state``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


IMAGE_SIZE = 32
NUM_CHANNELS = 3
BOTTLENECK_CHANNELS = 512
BOTTLENECK_SIZE = 4


@dataclass(frozen=True, slots=True)
class VAEConfig:
    """Architecture settings for the paper's 32x32 CIFAR-10 VAE."""

    image_size: int = IMAGE_SIZE
    state_dim: int = 256
    num_state_tokens: int = 1
    dropout: float = 0.1
    variational: bool = True
    sample_posterior: bool = True

    def __post_init__(self) -> None:
        if self.image_size != IMAGE_SIZE:
            raise ValueError("the paper Image Editing ResNet supports exactly 32x32 images")
        if self.state_dim < 1 or self.num_state_tokens < 1:
            raise ValueError("state_dim and num_state_tokens must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def latent_dim(self) -> int:
        return self.num_state_tokens * self.state_dim


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


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a projected skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip: nn.Module = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)
        out = self.dropout(self.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class ResidualUpBlock(nn.Module):
    """Transposed-convolution upsampling with a projected skip connection."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, 3, stride=2, padding=1, output_padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.conv = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.ConvTranspose2d(
            in_channels, out_channels, 1, stride=2, output_padding=1, bias=False
        )
        self.skip_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip_bn(self.skip(x))
        out = self.dropout(self.relu(self.bn1(self.upsample(x))))
        out = self.bn2(self.conv(out))
        out = out + identity
        return self.relu(out)


class Encoder(nn.Module):
    """Encode a 32x32 RGB image into latent state tokens."""

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv2d(NUM_CHANNELS, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = ResidualBlock(64, 64, stride=1, dropout=config.dropout)
        self.layer2 = ResidualBlock(64, 128, stride=2, dropout=config.dropout)
        self.layer3 = ResidualBlock(128, 256, stride=2, dropout=config.dropout)
        self.layer4 = ResidualBlock(256, BOTTLENECK_CHANNELS, stride=2, dropout=config.dropout)

        self.flattened_dim = BOTTLENECK_CHANNELS * BOTTLENECK_SIZE * BOTTLENECK_SIZE
        if config.variational:
            self.to_mean = nn.Linear(self.flattened_dim, config.latent_dim)
            self.to_logvar = nn.Linear(self.flattened_dim, config.latent_dim)
        else:
            self.to_state = nn.Linear(self.flattened_dim, config.latent_dim)

    def forward(self, image: Tensor) -> tuple[Tensor, GaussianPosterior | None]:
        image = self._to_float_nchw(image)
        batch_size = image.shape[0]

        features = self.relu(self.bn1(self.conv1(image)))
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        features = features.reshape(batch_size, -1)

        shape = (batch_size, self.config.num_state_tokens, self.config.state_dim)
        if not self.config.variational:
            state = self.to_state(features).view(shape)
            return F.normalize(state, dim=-1), None

        posterior = GaussianPosterior(
            mean=self.to_mean(features).view(shape),
            log_variance=self.to_logvar(features).view(shape),
        )
        state = posterior.sample() if self.config.sample_posterior else posterior.mean
        return state, posterior

    def _to_float_nchw(self, image: Tensor) -> Tensor:
        """Accept ``(B, H, W, C)`` or ``(B, C, H, W)`` bytes or floats.

        Convert to float and divide by 255 when the batch maximum exceeds 1.0.
        """

        if image.ndim != 4:
            raise ValueError(f"image must be 4-dimensional, got {tuple(image.shape)}")
        if image.shape[-1] == NUM_CHANNELS:
            image = image.permute(0, 3, 1, 2)
        if tuple(image.shape[1:]) != (
            NUM_CHANNELS,
            self.config.image_size,
            self.config.image_size,
        ):
            raise ValueError(
                f"image must be (B, 3, {self.config.image_size}, {self.config.image_size}) "
                f"or (B, {self.config.image_size}, {self.config.image_size}, 3), got "
                f"{tuple(image.shape)}"
            )
        if image.max() > 1.0:
            image = image.float() / 255.0
        return image.float()


class Decoder(nn.Module):
    """Decode latent state tokens into a 32x32 RGB image in ``[0, 1]``."""

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config
        self.flattened_dim = BOTTLENECK_CHANNELS * BOTTLENECK_SIZE * BOTTLENECK_SIZE
        self.from_state = nn.Linear(config.latent_dim, self.flattened_dim)
        self.layer1 = ResidualUpBlock(BOTTLENECK_CHANNELS, 256, dropout=config.dropout)
        self.layer2 = ResidualUpBlock(256, 128, dropout=config.dropout)
        self.layer3 = ResidualUpBlock(128, 64, dropout=config.dropout)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, NUM_CHANNELS, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, state: Tensor) -> Tensor:
        expected = (self.config.num_state_tokens, self.config.state_dim)
        if state.ndim != 3 or tuple(state.shape[1:]) != expected:
            raise ValueError(
                f"state must be (B, {expected[0]}, {expected[1]}), got {tuple(state.shape)}"
            )
        batch_size = state.shape[0]
        features = self.from_state(state.reshape(batch_size, -1))
        features = features.view(
            batch_size, BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE
        )
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        image = self.final_conv(features)
        return image.permute(0, 2, 3, 1)


class VAE(nn.Module):
    """Encoder/decoder pair sharing one :class:`VAEConfig`."""

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, GaussianPosterior | None]:
        state, posterior = self.encoder(image)
        return self.decoder(state), state, posterior


__all__ = [
    "BOTTLENECK_CHANNELS",
    "BOTTLENECK_SIZE",
    "Decoder",
    "Encoder",
    "GaussianPosterior",
    "IMAGE_SIZE",
    "NUM_CHANNELS",
    "ResidualBlock",
    "ResidualUpBlock",
    "VAE",
    "VAEConfig",
]
