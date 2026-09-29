"""Paper NEO-S sampling and majority-selection semantics for GridWorld."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from models.neo import NEO


@dataclass(frozen=True, slots=True)
class SampledGridWorldTheories:
    """Programs sampled from one support transition and their stop lengths."""

    action_indices: tuple[tuple[tuple[int, ...], ...], ...]
    support_solved_at_step: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GridWorldScalingBudgetMetrics:
    """Pass and majority-selected solved counts for one sampling budget."""

    budget: int
    total: int
    pass_self_solved: int
    select_self_solved: int
    pass_transfer_solved: int
    select_transfer_solved: int

    def _rate(self, count: int) -> float:
        return count / max(1, self.total)

    @property
    def pass_self(self) -> float:
        return self._rate(self.pass_self_solved)

    @property
    def select_self(self) -> float:
        return self._rate(self.select_self_solved)

    @property
    def pass_transfer(self) -> float:
        return self._rate(self.pass_transfer_solved)

    @property
    def select_transfer(self) -> float:
        return self._rate(self.select_transfer_solved)


@dataclass(frozen=True, slots=True)
class GridWorldTestTimeScalingResult:
    """NEO-S results for several nested sampling budgets."""

    max_steps: int
    sample_temperature: float
    budgets: tuple[GridWorldScalingBudgetMetrics, ...]

    def legacy_dict(self) -> dict[str, object]:
        """Return the exact result-key convention used by ``evaluate_ood.py``."""

        result: dict[str, object] = {
            "K_values": [metrics.budget for metrics in self.budgets],
            "sample_temperature": self.sample_temperature,
        }
        for metrics in self.budgets:
            budget = metrics.budget
            result.update(
                {
                    f"pass@{budget}_self": metrics.pass_self,
                    f"select@{budget}_self": metrics.select_self,
                    f"pass@{budget}_trans": metrics.pass_transfer,
                    f"select@{budget}_trans": metrics.select_transfer,
                    f"total@{budget}": metrics.total,
                }
            )
        return result


@dataclass(slots=True)
class _ScalingCounts:
    total: int = 0
    pass_self: int = 0
    select_self: int = 0
    pass_transfer: int = 0
    select_transfer: int = 0


def sample_gridworld_theories(
    model: NEO,
    support_input: Tensor,
    support_target: Tensor,
    *,
    max_steps: int,
    num_theories: int,
    sample_temperature: float,
    device: torch.device,
    hard_grounding: bool = True,
) -> SampledGridWorldTheories:
    """Sample ``num_theories`` VQ programs in batches."""

    if max_steps < 1 or num_theories < 1:
        raise ValueError("max_steps and num_theories must be positive")
    if sample_temperature <= 0:
        raise ValueError("sample_temperature must be positive")

    with torch.no_grad():
        support_in = support_input.unsqueeze(0).to(device).long()
        support_out = support_target.unsqueeze(0).to(device).long()
        current, _ = model.encoder(support_in)
        target, _ = model.encoder(support_out)
        current = current.expand(num_theories, -1, -1).clone()
        target = target.expand(num_theories, -1, -1)
        sequences: list[list[tuple[int, ...]]] = [
            [] for _ in range(num_theories)
        ]
        solved_at_step = torch.full(
            (num_theories,), -1, dtype=torch.long, device=device
        )

        for step in range(max_steps):
            action = model.theory_programmer(current, target)
            flattened = F.normalize(action.flatten(end_dim=-2).float(), dim=-1)
            embedding = F.normalize(model.quantizer.embedding.weight, dim=-1)
            distances = (
                flattened.square().sum(dim=1, keepdim=True)
                + embedding.square().sum(dim=1)
                - 2 * torch.einsum("bd,dn->bn", flattened, embedding.T)
            )
            probabilities = F.softmax(-distances / sample_temperature, dim=-1)
            indices = torch.multinomial(probabilities, 1).squeeze(-1)
            indices_by_theory = indices.view(num_theories, -1)
            for theory_index, row in enumerate(indices_by_theory):
                sequences[theory_index].append(
                    tuple(int(value) for value in row.detach().cpu().tolist())
                )

            values = model.quantizer.get_codebook_entry(indices).view(action.shape)
            current = model.program_executor(current, values)
            prediction = model.decoder(current).argmax(dim=-1)
            matches = prediction.eq(support_out.expand_as(prediction)).all(dim=(1, 2))
            newly_solved = matches & solved_at_step.eq(-1)
            solved_at_step[newly_solved] = step + 1
            if hard_grounding:
                current, _ = model.encoder(prediction)

    return SampledGridWorldTheories(
        action_indices=tuple(tuple(sequence) for sequence in sequences),
        support_solved_at_step=tuple(
            int(value) for value in solved_at_step.detach().cpu().tolist()
        ),
    )


def apply_gridworld_theories(
    model: NEO,
    query_input: Tensor,
    query_target: Tensor,
    theories: SampledGridWorldTheories,
    *,
    device: torch.device,
    hard_grounding: bool = True,
) -> tuple[bool, ...]:
    """Replay sampled programs and score each at its support solution length."""

    num_theories = len(theories.action_indices)
    if num_theories == 0:
        return ()
    num_steps = len(theories.action_indices[0])
    if any(len(sequence) != num_steps for sequence in theories.action_indices):
        raise ValueError("all sampled theories must have the same rollout length")
    if len(theories.support_solved_at_step) != num_theories:
        raise ValueError("support stop lengths must match the number of theories")

    with torch.no_grad():
        query_in = query_input.unsqueeze(0).to(device).long()
        query_out = query_target.unsqueeze(0).to(device).long()
        current, _ = model.encoder(query_in)
        current = current.expand(num_theories, -1, -1).clone()
        correct_at_stop = torch.zeros(num_theories, dtype=torch.bool, device=device)
        stop_steps = torch.tensor(
            theories.support_solved_at_step, dtype=torch.long, device=device
        )

        for step in range(num_steps):
            flat_indices = torch.tensor(
                [
                    token
                    for sequence in theories.action_indices
                    for token in sequence[step]
                ],
                dtype=torch.long,
                device=device,
            )
            values = model.quantizer.get_codebook_entry(flat_indices).view(
                num_theories, -1, model.quantizer.embedding.embedding_dim
            )
            current = model.program_executor(current, values)
            prediction = model.decoder(current).argmax(dim=-1)
            matches = prediction.eq(query_out.expand_as(prediction)).all(dim=(1, 2))
            correct_at_stop |= matches & stop_steps.eq(step + 1)
            if hard_grounding:
                current, _ = model.encoder(prediction)

    return tuple(bool(value) for value in correct_at_stop.cpu().tolist())


def _budget_outcome(
    theories: SampledGridWorldTheories,
    query_correct: Sequence[bool],
    budget: int,
) -> tuple[bool, bool, bool, bool]:
    """Compute pass/select self/transfer metrics for each sampling budget."""

    if budget < 1 or budget > len(theories.action_indices):
        raise ValueError("budget must fit within the sampled theory count")
    if len(query_correct) != len(theories.action_indices):
        raise ValueError("query correctness must match the sampled theory count")

    prefix = range(budget)
    solved = [
        index for index in prefix if theories.support_solved_at_step[index] >= 1
    ]
    pass_self = bool(solved)
    pass_transfer = any(query_correct[index] for index in prefix)
    if not solved:
        return pass_self, False, pass_transfer, False

    fingerprints = [
        theories.action_indices[index][
            : theories.support_solved_at_step[index]
        ]
        for index in solved
    ]
    winning_fingerprint = Counter(fingerprints).most_common(1)[0][0]
    winning_index = next(
        index
        for index, fingerprint in zip(solved, fingerprints, strict=True)
        if fingerprint == winning_fingerprint
    )
    return pass_self, True, pass_transfer, bool(query_correct[winning_index])


def evaluate_gridworld_test_time_scaling(
    model: NEO,
    batches: Iterable[Tensor],
    *,
    max_steps: int,
    budgets: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
    sample_temperature: float = 0.3,
    device: torch.device,
    hard_grounding: bool = True,
    num_samples: int | None = None,
) -> GridWorldTestTimeScalingResult:
    """Evaluate the paper NEO-S nested-budget and majority-vote protocol."""

    resolved_budgets = tuple(int(value) for value in budgets)
    if not resolved_budgets or any(value < 1 for value in resolved_budgets):
        raise ValueError("budgets must contain positive integers")
    if tuple(sorted(set(resolved_budgets))) != resolved_budgets:
        raise ValueError("budgets must be unique and strictly increasing")
    if num_samples is not None and num_samples < 1:
        raise ValueError("num_samples must be positive when provided")

    model.eval()
    counts = {budget: _ScalingCounts() for budget in resolved_budgets}
    total = 0
    for batch in batches:
        if batch.ndim != 4 or batch.shape[1] < 4:
            raise ValueError("evaluation batches must have shape (batch, >=4, H, W)")
        for episode in batch:
            theories = sample_gridworld_theories(
                model,
                episode[0],
                episode[1],
                max_steps=max_steps,
                num_theories=resolved_budgets[-1],
                sample_temperature=sample_temperature,
                device=device,
                hard_grounding=hard_grounding,
            )
            query_correct = apply_gridworld_theories(
                model,
                episode[2],
                episode[3],
                theories,
                device=device,
                hard_grounding=hard_grounding,
            )
            for budget in resolved_budgets:
                pass_self, select_self, pass_transfer, select_transfer = (
                    _budget_outcome(theories, query_correct, budget)
                )
                current = counts[budget]
                current.total += 1
                current.pass_self += int(pass_self)
                current.select_self += int(select_self)
                current.pass_transfer += int(pass_transfer)
                current.select_transfer += int(select_transfer)
            total += 1
            if num_samples is not None and total >= num_samples:
                return _finish_scaling(
                    counts, max_steps=max_steps, sample_temperature=sample_temperature
                )
    return _finish_scaling(
        counts, max_steps=max_steps, sample_temperature=sample_temperature
    )


def _finish_scaling(
    counts: dict[int, _ScalingCounts],
    *,
    max_steps: int,
    sample_temperature: float,
) -> GridWorldTestTimeScalingResult:
    return GridWorldTestTimeScalingResult(
        max_steps=max_steps,
        sample_temperature=sample_temperature,
        budgets=tuple(
            GridWorldScalingBudgetMetrics(
                budget=budget,
                total=value.total,
                pass_self_solved=value.pass_self,
                select_self_solved=value.select_self,
                pass_transfer_solved=value.pass_transfer,
                select_transfer_solved=value.select_transfer,
            )
            for budget, value in counts.items()
        ),
    )


__all__ = [
    "GridWorldScalingBudgetMetrics",
    "GridWorldTestTimeScalingResult",
    "SampledGridWorldTheories",
    "apply_gridworld_theories",
    "evaluate_gridworld_test_time_scaling",
    "sample_gridworld_theories",
]
