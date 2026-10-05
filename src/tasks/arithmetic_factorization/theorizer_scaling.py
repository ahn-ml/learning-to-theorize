"""NEO-S sampling and majority selection for arithmetic.

NEO-S samples several candidate theories for one support transition, replays
each on the query, and selects one by majority vote over the sampled operation
sequences. It is not a search: the candidates are drawn independently and
never pruned or expanded.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from models.neo import NEO

# Sampling budgets and temperature for the arithmetic scaling results.
PAPER_SCALING_BUDGETS: tuple[int, ...] = (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
)
PAPER_SCALING_TEMPERATURE = 1.0


@dataclass(frozen=True, slots=True)
class SampledArithmeticTheories:
    """Theories sampled from one support transition and their stop lengths."""

    action_indices: tuple[tuple[tuple[int, ...], ...], ...]
    support_solved_at_step: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.action_indices) != len(self.support_solved_at_step):
            raise ValueError("every sampled theory needs one stop length")

    @property
    def support_correct(self) -> tuple[bool, ...]:
        """Whether each theory ever reproduced the support target."""

        return tuple(step >= 1 for step in self.support_solved_at_step)


@dataclass(frozen=True, slots=True)
class ArithmeticScalingBudgetMetrics:
    """Pass and majority-selected solved counts for one sampling budget."""

    budget: int
    total: int
    pass_self_solved: int
    select_self_solved: int
    pass_transfer_solved: int
    select_transfer_solved: int
    support_solved_total: int
    vote_episodes: int
    vote_count_total: int
    vote_ratio_total: float

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

    @property
    def average_support_solved(self) -> float:
        return self.support_solved_total / max(1, self.total)

    # Vote statistics average over the episodes that produced a vote, not over
    # every episode.

    @property
    def average_vote_count(self) -> float:
        return self.vote_count_total / max(1, self.vote_episodes)

    @property
    def average_vote_ratio(self) -> float:
        return self.vote_ratio_total / max(1, self.vote_episodes)


@dataclass(frozen=True, slots=True)
class ArithmeticTestTimeScalingResult:
    """NEO-S results for several nested sampling budgets."""

    max_steps: int
    sample_temperature: float
    budgets: tuple[ArithmeticScalingBudgetMetrics, ...]

    def legacy_dict(self) -> dict[str, object]:
        """Return the results keyed as ``pass@K_self``, ``select@K_trans``, etc."""

        result: dict[str, object] = {
            "K_values": [metrics.budget for metrics in self.budgets],
            "sample_temperature": self.sample_temperature,
            "max_steps_used": self.max_steps,
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
                    f"avg_support_solved@{budget}": metrics.average_support_solved,
                    f"avg_vote_count@{budget}": metrics.average_vote_count,
                    f"avg_vote_ratio@{budget}": metrics.average_vote_ratio,
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
    support_solved: int = 0
    vote_episodes: int = 0
    vote_count: int = 0
    vote_ratio: float = 0.0


def select_winning_theory(
    theories: SampledArithmeticTheories,
    solved_indices: Sequence[int],
) -> tuple[int, int, float]:
    """Pick one theory from those that explained the support transition.

    A discrete bottleneck votes on the operation sequence truncated to the
    step that solved the support, so theories that found the same program
    reinforce each other.
    """

    if not solved_indices:
        raise ValueError("at least one theory must have solved the support")

    fingerprints = [
        theories.action_indices[index][: theories.support_solved_at_step[index]]
        for index in solved_indices
    ]
    winning_fingerprint, vote_count = Counter(fingerprints).most_common(1)[0]
    winning_index = next(
        index
        for index, fingerprint in zip(solved_indices, fingerprints, strict=True)
        if fingerprint == winning_fingerprint
    )
    return winning_index, vote_count, vote_count / len(solved_indices)


def budget_outcome(
    theories: SampledArithmeticTheories,
    query_correct: Sequence[bool],
    budget: int,
) -> tuple[bool, bool, bool, bool, int, int, float]:
    """Score one nested budget: oracle pass and majority-selected outcomes."""

    if budget < 1:
        raise ValueError("budget must be positive")
    if len(query_correct) != len(theories.action_indices):
        raise ValueError("query correctness must match the sampled theory count")

    prefix = range(min(budget, len(theories.action_indices)))
    support_correct = theories.support_correct
    pass_self = any(support_correct[index] for index in prefix)
    pass_transfer = any(query_correct[index] for index in prefix)
    solved_indices = [index for index in prefix if support_correct[index]]
    if not solved_indices:
        return pass_self, False, pass_transfer, False, 0, 0, 0.0

    winning_index, vote_count, vote_ratio = select_winning_theory(
        theories, solved_indices
    )
    return (
        pass_self,
        True,
        pass_transfer,
        bool(query_correct[winning_index]),
        len(solved_indices),
        vote_count,
        vote_ratio,
    )


def evaluate_test_time_scaling(
    model: NEO,
    batches: Iterable,
    *,
    max_steps: int,
    budgets: Sequence[int] = PAPER_SCALING_BUDGETS,
    sample_temperature: float = PAPER_SCALING_TEMPERATURE,
    device: str = "cuda",
    num_samples: int | None = None,
    hard_grounding: bool = True,
) -> ArithmeticTestTimeScalingResult:
    """Evaluate NEO-S with the same grounding and stopping rule as NEO."""

    from tasks.arithmetic_factorization.data.dataset import unpack_batch
    from tasks.arithmetic_factorization.latent_inference import latent_candidates

    ordered = sorted(set(int(budget) for budget in budgets))
    if not ordered or ordered[0] < 1:
        raise ValueError("positive sampling budgets are required")
    largest = ordered[-1]
    counts = {budget: _ScalingCounts() for budget in ordered}
    seen = 0

    for batch in batches:
        data, _ = unpack_batch(batch, device)
        for index in range(data.shape[0]):
            if num_samples is not None and seen >= num_samples:
                break
            candidates = latent_candidates(
                model, data[index], max_steps=max_steps,
                candidates=largest, temperature=sample_temperature,
                hard_grounding=hard_grounding,
            )
            sequences = list(candidates.indices.unbind(0))
            solved_at_step = torch.where(candidates.support_correct, candidates.lengths, -1).tolist()
            query_correct = (candidates.query_correct & candidates.support_correct).tolist()
            indices = tuple(
                tuple(tuple(int(v) for v in step.tolist()) for step in sequence)
                for sequence in sequences
            )
            theories = SampledArithmeticTheories(
                action_indices=indices,
                support_solved_at_step=tuple(int(step) for step in solved_at_step),
            )
            for budget in ordered:
                outcome = budget_outcome(
                    theories,
                    query_correct,
                    budget,
                )
                entry = counts[budget]
                entry.total += 1
                entry.pass_self += int(outcome[0])
                entry.select_self += int(outcome[1])
                entry.pass_transfer += int(outcome[2])
                entry.select_transfer += int(outcome[3])
                entry.support_solved += outcome[4]
                if outcome[4]:
                    entry.vote_episodes += 1
                    entry.vote_count += outcome[5]
                    entry.vote_ratio += outcome[6]
            seen += 1
        if num_samples is not None and seen >= num_samples:
            break

    return ArithmeticTestTimeScalingResult(
        max_steps=max_steps,
        sample_temperature=sample_temperature,
        budgets=tuple(
            ArithmeticScalingBudgetMetrics(
                budget=budget,
                total=counts[budget].total,
                pass_self_solved=counts[budget].pass_self,
                select_self_solved=counts[budget].select_self,
                pass_transfer_solved=counts[budget].pass_transfer,
                select_transfer_solved=counts[budget].select_transfer,
                support_solved_total=counts[budget].support_solved,
                vote_episodes=counts[budget].vote_episodes,
                vote_count_total=counts[budget].vote_count,
                vote_ratio_total=counts[budget].vote_ratio,
            )
            for budget in ordered
        ),
    )


__all__ = [
    "PAPER_SCALING_BUDGETS",
    "PAPER_SCALING_TEMPERATURE",
    "ArithmeticScalingBudgetMetrics",
    "ArithmeticTestTimeScalingResult",
    "SampledArithmeticTheories",
    "budget_outcome",
    "evaluate_test_time_scaling",
    "select_winning_theory",
]
