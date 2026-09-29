"""Batched ID validation matching the released final transfer protocol."""
from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def released_transfer_batch(model, data: Tensor, *, max_steps: int) -> dict[str, Tensor]:
    """Infer on support, re-encode each prediction, then replay the support program.

    Uses the shortest exactly solved support prefix, otherwise all ``max_steps``.
    Query targets only score the selected prediction. FP32 matches final evaluation;
    it deliberately differs from the latent recurrence used by the training loss.
    """
    if data.ndim != 4 or data.shape[1] != 4 or max_steps < 1:
        raise ValueError('expected (B,4,H,W) episodes and a positive horizon')
    with torch.autocast(device_type=data.device.type, enabled=False):
        support, target, query, answer = data.unbind(1)
        current, _ = model.encoder(support)
        goal, _ = model.encoder(target)
        actions, predictions = [], []
        for _ in range(max_steps):
            action = model.theory_programmer(current, goal)
            action = model.quantizer(action, training_mode=False).values
            actions.append(action)
            current = model.program_executor(current, action)
            prediction = model.decoder(current).argmax(-1)
            predictions.append(prediction)
            current, _ = model.encoder(prediction)
        stops = torch.full((len(data),), max_steps - 1, device=data.device, dtype=torch.long)
        for step in range(max_steps - 1, -1, -1):
            solved = predictions[step].eq(target).flatten(1).all(1)
            stops[solved] = step
        rows = torch.arange(len(data), device=data.device)
        support_prediction = torch.stack(predictions)[stops, rows]
        current, _ = model.encoder(query)
        predictions = []
        for action in actions:
            current = model.program_executor(current, action)
            prediction = model.decoder(current).argmax(-1)
            predictions.append(prediction)
            current, _ = model.encoder(prediction)
        query_prediction = torch.stack(predictions)[stops, rows]
    return {
        'support_prediction': support_prediction,
        'query_prediction': query_prediction,
        'selected_lengths': stops + 1,
        'support_solved': support_prediction.eq(target).flatten(1).all(1),
        'query_solved': query_prediction.eq(answer).flatten(1).all(1),
        'support_digit_correct': support_prediction.eq(target).flatten(1).float().mean(1),
    }
