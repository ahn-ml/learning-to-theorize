"""Image Editing observation semantics supplied to the shared NEO model.

Observations are 32x32 RGB images.  The decoder emits floats in ``[0, 1]``;
stored targets are ``uint8`` in ``[0, 255]`` and are rescaled here, matching the
paper-producing implementation which normalised the target rather than the
prediction.

Two places differ from a discrete task like GridWorld:

* **Grounding.** The valid-observation projection is 8-bit quantisation,
  ``round(x * 255) / 255``, which plays the role GridWorld's ``argmax`` plays.
* **Exact match.** There is none.  Pixel-exact agreement between a decoded image
  and its target never occurs, so length selection uses no exact-match
  override: :meth:`exact_match` always returns ``False``, which makes the shared
  rollout's override a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


IMAGE_SIZE = 32
NUM_CHANNELS = 3
GRIDS_PER_EPISODE = 4

#: Number of quantisation levels the grounding projection snaps to.
QUANTISATION_LEVELS = 255


@dataclass(frozen=True, slots=True)
class ImageEditingMetrics:
    """Mean absolute pixel error for a group of predictions."""

    l1: float


class ImageEditingObjective:
    """Pairing, L1 reconstruction, grounding, and metric rules for 32x32 images."""

    def validate_batch(self, batch: Tensor, *, is_eval: bool) -> None:
        if batch.ndim != 5:
            raise ValueError(
                "episode images must be shaped (B, pairs*2, 32, 32, 3), got "
                f"{tuple(batch.shape)}"
            )
        if batch.shape[1] < 2 or batch.shape[1] % 2:
            raise ValueError("episodes must contain an even, non-zero number of images")
        if tuple(batch.shape[2:]) != (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS):
            raise ValueError(
                f"Image Editing requires {IMAGE_SIZE}x{IMAGE_SIZE}x{NUM_CHANNELS} images"
            )
        if is_eval and batch.shape[1] != GRIDS_PER_EPISODE:
            raise ValueError("evaluation requires one support and one query pair per episode")

    def split_batch(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        """Split an episode batch into inputs and their edited targets."""

        trailing = batch.shape[-3:]
        inputs = batch[:, ::2].reshape(-1, *trailing)
        targets = batch[:, 1::2].reshape(-1, *trailing)
        return inputs, targets

    def _normalize_target(self, target: Tensor) -> Tensor:
        if target.dtype != torch.float32:
            return target.float() / 255.0
        return target

    def reconstruction_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(prediction, self._normalize_target(target), reduction="mean")

    def per_sample_reconstruction_loss(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor:
        target = self._normalize_target(target)
        elementwise = F.l1_loss(prediction, target, reduction="none")
        return elementwise.reshape(target.shape[0], -1).mean(dim=1)

    def decode_prediction(self, prediction: Tensor) -> Tensor:
        """Snap a decoded image onto the 8-bit grid of valid observations."""

        return torch.round(prediction * QUANTISATION_LEVELS) / QUANTISATION_LEVELS

    def exact_match(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Always ``False``; see the module docstring."""

        return torch.zeros(
            target.shape[0], dtype=torch.bool, device=prediction.device
        )

    def metrics(self, prediction: Tensor, target: Tensor) -> ImageEditingMetrics:
        target = self._normalize_target(target)
        with torch.no_grad():
            l1 = F.l1_loss(prediction, target, reduction="mean").item()
        return ImageEditingMetrics(l1=l1)

    def empty_metrics(self) -> ImageEditingMetrics:
        return ImageEditingMetrics(l1=0.0)


__all__ = [
    "GRIDS_PER_EPISODE",
    "IMAGE_SIZE",
    "ImageEditingMetrics",
    "ImageEditingObjective",
    "NUM_CHANNELS",
    "QUANTISATION_LEVELS",
]
