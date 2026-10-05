"""Transformer blocks and accuracy metrics for Arithmetic."""

import torch
import torch.nn as nn


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


def compute_accuracy(pred, target):
    """
    Return digit and whole-number accuracy.

    Args:
        pred: digit logits (batch, height, width, num_classes)
        target: digits (batch, height, width)

    Returns:
        pixel_acc: per-digit accuracy
        grid_acc: per-number (exact match) accuracy
    """
    batch_size = pred.shape[0]
    pred_classes = torch.argmax(pred, dim=-1) # (batch_size, height, width)

    # Calculate per pixel accuracy (soft accuracy) and per grid accuracy (hard accuracy)
    pixel_correct = (pred_classes == target).float().mean(dim=(1, 2)).sum().item()
    correct_per_sample = (pred_classes == target)
    grid_correct = correct_per_sample.view(batch_size, -1).all(dim=1).sum().item()

    pixel_acc = pixel_correct / batch_size
    grid_acc = grid_correct / batch_size
    return pixel_acc, grid_acc
