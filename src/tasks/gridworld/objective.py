"""GridWorld observation semantics supplied to the shared NEO model."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class GridWorldAccuracy:
    """Pixel, exact-grid, and macro-F1 values for a group of predictions."""

    pixel: float
    grid: float
    macro_f1: float


class GridWorldObjective:
    """Pairing, reconstruction, exact-match, and metric rules for 10x10 grids."""

    def validate_batch(self, batch: Tensor, *, is_eval: bool) -> None:
        if batch.ndim != 4:
            raise ValueError(
                "episode_grids must be shaped (B, pairs*2, 10, 10), got "
                f"{tuple(batch.shape)}"
            )
        if batch.shape[1] < 2 or batch.shape[1] % 2:
            raise ValueError("episode_grids must contain an even, non-zero number of grids")
        if tuple(batch.shape[2:]) != (10, 10):
            raise ValueError("GridWorld theorizer requires 10x10 grids")
        if batch.is_floating_point() or batch.is_complex():
            raise TypeError("episode_grids must contain integer color indices")
        num_flattened_pairs = batch.shape[0] * (batch.shape[1] // 2)
        if is_eval and num_flattened_pairs % 2:
            raise ValueError("evaluation requires paired support/query transformations")

    def split_batch(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        height, width = batch.shape[-2:]
        inputs = batch[:, ::2].reshape(-1, height, width)
        targets = batch[:, 1::2].reshape(-1, height, width)
        return inputs, targets

    def reconstruction_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        return grid_cross_entropy(prediction, target)

    def per_sample_reconstruction_loss(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor:
        self._validate_prediction(prediction, target)
        return F.cross_entropy(
            prediction.reshape(-1, prediction.shape[-1]),
            target.reshape(-1).long(),
            reduction="none",
        ).reshape(target.shape[0], target.shape[1], target.shape[2]).mean(dim=(1, 2))

    def metrics(self, prediction: Tensor, target: Tensor) -> GridWorldAccuracy:
        return grid_accuracy(prediction, target)

    def decode_prediction(self, prediction: Tensor) -> Tensor:
        return prediction.argmax(dim=-1)

    def exact_match(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "decoded predictions and targets do not align: "
                f"{tuple(prediction.shape)}, {tuple(target.shape)}"
            )
        return prediction.eq(target).all(dim=(1, 2))

    def empty_metrics(self) -> GridWorldAccuracy:
        return GridWorldAccuracy(pixel=0.0, grid=0.0, macro_f1=0.0)

    @staticmethod
    def _validate_prediction(prediction: Tensor, target: Tensor) -> None:
        if prediction.shape[:3] != target.shape:
            raise ValueError(
                "predictions and targets do not align: "
                f"{tuple(prediction.shape)}, {tuple(target.shape)}"
            )


def grid_cross_entropy(prediction: Tensor, target: Tensor) -> Tensor:
    GridWorldObjective._validate_prediction(prediction, target)
    return F.cross_entropy(
        prediction.reshape(-1, prediction.shape[-1]),
        target.reshape(-1).long(),
        reduction="mean",
    )


def grid_accuracy(prediction: Tensor, target: Tensor) -> GridWorldAccuracy:
    GridWorldObjective._validate_prediction(prediction, target)
    if target.shape[0] < 1:
        raise ValueError("accuracy requires at least one grid")
    decoded = prediction.argmax(dim=-1)
    correct = decoded.eq(target)
    pixel = correct.float().mean(dim=(1, 2)).sum().item() / target.shape[0]
    grid = correct.reshape(target.shape[0], -1).all(dim=1).sum().item()
    grid /= target.shape[0]

    prediction_flat = decoded.reshape(-1)
    target_flat = target.reshape(-1)
    scores: list[float] = []
    for color in range(prediction.shape[-1]):
        predicted_color = prediction_flat.eq(color)
        target_color = target_flat.eq(color)
        if not (target_color.any() or predicted_color.any()):
            continue
        true_positive = (predicted_color & target_color).sum().float()
        false_positive = (predicted_color & ~target_color).sum().float()
        false_negative = (~predicted_color & target_color).sum().float()
        precision = true_positive / (true_positive + false_positive + 1e-8)
        recall = true_positive / (true_positive + false_negative + 1e-8)
        score = 2 * precision * recall / (precision + recall + 1e-8)
        scores.append(score.item())
    macro_f1 = sum(scores) / len(scores) if scores else 0.0
    return GridWorldAccuracy(pixel=pixel, grid=grid, macro_f1=macro_f1)


__all__ = [
    "GridWorldAccuracy",
    "GridWorldObjective",
    "grid_accuracy",
    "grid_cross_entropy",
]
