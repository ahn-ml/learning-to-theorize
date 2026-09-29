"""Loss, exact match, and metrics for arithmetic factorization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from tasks.arithmetic_factorization.models.model_utils import compute_accuracy

# Digits are drawn from a ten-symbol alphabet.
NUM_SYMBOLS = 10


@dataclass(frozen=True, slots=True)
class ArithmeticAccuracy:
    """Digit, whole-number, and macro-F1 accuracy for one evaluation pass."""

    digit_accuracy: float
    number_accuracy: float
    macro_f1: float


def digit_cross_entropy(prediction: Tensor, target: Tensor) -> Tensor:
    """Return the mean cross entropy over every digit position."""

    return F.cross_entropy(
        prediction.reshape(-1, prediction.shape[-1]),
        target.reshape(-1).to(torch.long),
    )


class ArithmeticObjective:
    """Task operations required by the shared latent-program rollout."""

    def validate_batch(self, batch: Tensor, *, is_eval: bool) -> None:
        if batch.dim() != 4:
            raise ValueError(
                f"arithmetic batches must be rank 4; got shape {tuple(batch.shape)}"
            )
        if batch.shape[1] % 2:
            raise ValueError(
                "arithmetic batches interleave input/output grids, so the grid "
                f"count must be even; got {batch.shape[1]}"
            )
        if is_eval and batch.shape[1] != 4:
            raise ValueError(
                "evaluation needs one support pair and one query pair per episode"
            )

    def split_batch(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        """Split interleaved grids into flattened inputs and targets."""

        height, width = batch.shape[-2:]
        inputs = batch[:, ::2].reshape(-1, height, width)
        targets = batch[:, 1::2].reshape(-1, height, width)
        return inputs, targets

    def reconstruction_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        return digit_cross_entropy(prediction, target)

    def per_sample_reconstruction_loss(
        self, prediction: Tensor, target: Tensor
    ) -> Tensor:
        per_digit = F.cross_entropy(
            prediction.reshape(-1, prediction.shape[-1]),
            target.reshape(-1).to(torch.long),
            reduction="none",
        )
        return per_digit.reshape(target.shape).mean(dim=(1, 2))

    def metrics(self, prediction: Tensor, target: Tensor) -> ArithmeticAccuracy:
        digit_accuracy, number_accuracy, macro_f1 = compute_accuracy(
            prediction, target, num_classes=NUM_SYMBOLS
        )
        return ArithmeticAccuracy(
            digit_accuracy=float(digit_accuracy),
            number_accuracy=float(number_accuracy),
            macro_f1=float(macro_f1),
        )

    def decode_prediction(self, prediction: Tensor) -> Tensor:
        """Collapse digit logits to the predicted digit grid."""

        return torch.argmax(prediction, dim=-1)

    def exact_match(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Return which episodes reproduce every digit of the target."""

        return (prediction == target).flatten(1).all(dim=1)

    def empty_metrics(self) -> ArithmeticAccuracy:
        return ArithmeticAccuracy(
            digit_accuracy=0.0, number_accuracy=0.0, macro_f1=0.0
        )


__all__ = [
    "NUM_SYMBOLS",
    "ArithmeticAccuracy",
    "ArithmeticObjective",
    "digit_cross_entropy",
]
