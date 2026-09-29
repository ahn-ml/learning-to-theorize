"""Shared latent rollout and support-only stopping for Arithmetic NEO/NEO-S."""
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class LatentCandidates:
    indices: Tensor  # candidates, steps, action tokens
    lengths: Tensor
    support_prediction: Tensor
    query_prediction: Tensor
    support_correct: Tensor
    query_correct: Tensor


@torch.no_grad()
def latent_candidates(model, episode: Tensor, *, max_steps: int,
                      candidates: int = 1, temperature: float = 1.0,
                      greedy: bool = False,
                      hard_grounding: bool = True) -> LatentCandidates:
    """Share support actions with query and preserve the matched stopping rule.

    ``hard_grounding`` reencodes decoded support/query states between steps.
    It leaves scoring and support-only length selection unchanged.

    Greedy uses the model's normal VQ path. Sampling uses the released normalized
    distance logits at the given temperature. MDL has the model's shortest-exact
    override. Query labels never select actions, lengths, or winning candidates.
    Unsolved candidates retain their MDL prediction; NEO-S filters them at voting.
    """
    if episode.ndim != 3 or episode.shape[0] != 4:
        raise ValueError('expected one (4,H,W) episode')
    if max_steps < 1 or candidates < 1 or temperature <= 0:
        raise ValueError('positive horizon, candidate count and temperature required')
    if model.training:
        raise ValueError('requires an evaluation-mode discrete NEO')
    with torch.autocast(device_type=episode.device.type, enabled=False):
        batch = episode.unsqueeze(0).expand(candidates, -1, -1, -1)
        inputs, targets = model.objective.split_batch(batch)
        current, _ = model.encoder(inputs)
        goal, _ = model.encoder(targets)
        predictions, losses, indices = [], [], []
        for step in range(max_steps):
            raw = model.theory_programmer(current, goal)[::2]
            if greedy:
                quantized = model.quantizer(raw, training_mode=False)
                action, codes = quantized.values, quantized.indices
            else:
                vectors = F.normalize(raw.flatten(end_dim=-2).float(), dim=-1)
                codes_normalized = F.normalize(model.quantizer.quantizer.embedding.weight, dim=-1)
                distance = (vectors.square().sum(1, keepdim=True)
                            + codes_normalized.square().sum(1)
                            - 2 * torch.einsum('bd,dn->bn', vectors, codes_normalized.T))
                codes = torch.multinomial((-distance / temperature).softmax(-1), 1).squeeze(-1)
                action = model.quantizer.get_codebook_entry(codes).view_as(raw)
            indices.append(codes.reshape(candidates, -1))
            current = model.program_executor(current, action.repeat_interleave(2, 0))
            logits = model.decoder(current)
            predictions.append(model.objective.decode_prediction(logits))
            losses.append(model.objective.per_sample_reconstruction_loss(logits, targets)[::2]
                          * model.length_control_coefficient ** (step + 1))
            if hard_grounding:
                current, _ = model.encoder(predictions[-1])
        stops = torch.stack(losses, 1).argmin(1)
        for step in range(max_steps - 1, -1, -1):
            exact = model.objective.exact_match(predictions[step][::2], targets[::2])
            stops[exact] = step
        predictions = torch.stack(predictions)
        rows = torch.arange(candidates, device=episode.device) * 2
        support, query = predictions[stops, rows], predictions[stops, rows + 1]
        return LatentCandidates(torch.stack(indices, 1), stops + 1, support, query,
                                model.objective.exact_match(support, targets[::2]),
                                model.objective.exact_match(query, targets[1::2]))
