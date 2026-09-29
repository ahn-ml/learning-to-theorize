"""Transformer blocks and accuracy metrics for Arithmetic."""

import math

import torch
import torch.nn as nn


def sinusoidal_embeddings(num_positions: int, d_model: int) -> torch.Tensor:
    """Return fixed position embeddings with shape (num_positions, d_model)."""
    position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * -(math.log(10000.0) / d_model)
    )
    embeddings = torch.zeros(num_positions, d_model)
    embeddings[:, 0::2] = torch.sin(position * div_term)
    embeddings[:, 1::2] = torch.cos(position * div_term)
    return embeddings


class EncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, batch_first=True)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        attention, _ = self.self_attn(
            src, src, src,
            need_weights=False,
            is_causal=False,
            average_attn_weights=False,
        )
        output = self.norm1(src + self.dropout1(attention))
        hidden = self.linear2(self.dropout(self.activation(self.linear1(output))))
        return self.norm2(output + self.dropout2(hidden))


class Encoder(nn.TransformerEncoder):
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output)
        if self.norm is not None:
            output = self.norm(output)
        return output


def compute_accuracy(pred, target, num_classes=10):
    """
    Helper to compute accuracy and F1 score.

    Args:
        pred: predictions (batch, height, width, num_classes)
        target: targets (batch, height, width)
        num_classes: number of classes (default 10 for ARC: 0-9)

    Returns:
        pixel_acc: per-pixel accuracy
        grid_acc: per-grid (exact match) accuracy
        macro_f1: macro-averaged F1 score across all classes
    """
    batch_size = pred.shape[0]
    if pred.shape != target.shape: # when using progressive decoding (no halting prediction, naive version)
        pred = pred.reshape(target.shape[0], -1, *pred.shape[1:])[:, -1] # take the last prediction for accuracy calculation

    # Get predicted classes and masks
    pred_classes = torch.argmax(pred, dim=-1) # (batch_size, height, width)

    # Calculate per pixel accuracy (soft accuracy) and per grid accuracy (hard accuracy)
    pixel_correct = (pred_classes == target).float().mean(dim=(1, 2)).sum().item()
    correct_per_sample = (pred_classes == target)
    grid_correct = correct_per_sample.view(batch_size, -1).all(dim=1).sum().item()

    pixel_acc = pixel_correct / batch_size
    grid_acc = grid_correct / batch_size

    # Compute F1 score (macro-averaged across classes)
    pred_flat = pred_classes.reshape(-1)
    target_flat = target.reshape(-1)

    # Per-class precision, recall, F1
    f1_scores = []
    for cls in range(num_classes):
        pred_cls = (pred_flat == cls)
        target_cls = (target_flat == cls)

        tp = (pred_cls & target_cls).sum().float()
        fp = (pred_cls & ~target_cls).sum().float()
        fn = (~pred_cls & target_cls).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Only include classes that appear in target (avoid inflating F1 with absent classes)
        if target_cls.sum() > 0 or pred_cls.sum() > 0:
            f1_scores.append(f1.item())

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return pixel_acc, grid_acc, macro_f1
