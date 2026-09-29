"""Canonical program enumeration and the alpha split over programs.

A *program* is a canonically ordered multiset of primitive names.  Canonical
ordering collapses orderings that the generator treats as equivalent, so
``("brightness_plus", "hue_plus")`` is enumerated once rather than twice.

The alpha split partitions programs, not episodes: ``alpha`` is the fraction of
splittable programs kept in distribution, and everything held out becomes the
compositional-OOD program set.  Two program families are never held out:

* **anchors** -- a repeatable primitive applied twice, which is what lets the
  model ground a primitive's identity independently of composition; and
* **non-repeatable length-1 programs** -- the only place those primitives appear
  on their own.

The permutation uses :func:`numpy.random.seed` and
:func:`numpy.random.permutation` with :data:`PAPER_SPLIT_SEED`. Dataset generation
uses this RandomState sequence consistently across all splits.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np

from tasks.image_editing.data.primitives import (
    OPPOSING_PRIMITIVES,
    non_repeatable_primitives,
    primitive_names,
    repeatable_primitives,
)


Program = tuple[str, ...]

#: Seed for the program-level alpha split in the released alpha profiles.
#:
#: Alpha 0.66 and 0.33 use seed 43 for generation and program splitting.
#: Alpha 1.00 includes all programs and requires no split.
PAPER_SPLIT_SEED = 43

#: Longest program the alpha profiles enumerate.
PAPER_MAX_PROGRAM_LENGTH = 2

#: Program lengths held out for the length-OOD evaluation.
PAPER_MIN_PROGRAM_LENGTH_OOD = 3
PAPER_MAX_PROGRAM_LENGTH_OOD = 4


#: How often a primitive may appear in one program before the edit stops being
#: legible.  Four rotations return to the original image, and a third
#: ``brightness_minus`` leaves the image too dark for the change test to pass.
#: Primitives absent from this mapping may repeat freely.
REPEAT_LIMITS: dict[str, int] = {
    "rotation": 3,
    "brightness_minus": 2,
}


def _repeat_limit(name: str) -> int:
    if name in non_repeatable_primitives():
        return 1
    return REPEAT_LIMITS.get(name, PAPER_MAX_PROGRAM_LENGTH_OOD)


def is_valid_program(program: Program) -> bool:
    """Return whether the generator is allowed to compose this program.

    The same rules cover in-distribution (length 1-2) and length-OOD (length
    3-4) programs: the repeat caps in :data:`REPEAT_LIMITS` cannot bind at
    length 2, so the shorter enumeration is unaffected by them.
    """

    counts = Counter(program)
    for name, count in counts.items():
        if count > _repeat_limit(name):
            return False
    for first, second in OPPOSING_PRIMITIVES:
        if counts.get(first, 0) > 0 and counts.get(second, 0) > 0:
            return False
    return True


def enumerate_programs(
    max_length: int = PAPER_MAX_PROGRAM_LENGTH,
    *,
    min_length: int = 1,
) -> tuple[Program, ...]:
    """Return every valid canonical program of length ``min_length..max_length``."""

    if min_length < 1:
        raise ValueError("min_length must be positive")
    if max_length < min_length:
        raise ValueError("max_length must not be smaller than min_length")
    names = primitive_names()
    programs: list[Program] = []
    for length in range(min_length, max_length + 1):
        for combination in combinations_with_replacement(names, length):
            program = tuple(combination)
            if is_valid_program(program):
                programs.append(program)
    return tuple(programs)


def enumerate_length_ood_programs() -> tuple[Program, ...]:
    """Return the length-OOD program set: valid programs of length 3 and 4."""

    return enumerate_programs(
        PAPER_MAX_PROGRAM_LENGTH_OOD,
        min_length=PAPER_MIN_PROGRAM_LENGTH_OOD,
    )


def anchor_programs() -> tuple[Program, ...]:
    """Return ``(p, p)`` for each repeatable primitive, in canonical order."""

    return tuple((name, name) for name in repeatable_primitives())


def non_repeatable_unit_programs() -> tuple[Program, ...]:
    """Return the length-1 programs for primitives that cannot repeat."""

    return tuple((name,) for name in non_repeatable_primitives())


def splittable_programs() -> tuple[Program, ...]:
    """Return the programs the alpha split may hold out.

    These are the repeatable length-1 programs plus every valid length-2
    program that is not an anchor.
    """

    reserved = set(anchor_programs()) | set(non_repeatable_unit_programs())
    return tuple(
        program
        for program in enumerate_programs(PAPER_MAX_PROGRAM_LENGTH)
        if program not in reserved
    )


@dataclass(frozen=True, slots=True)
class ProgramSplit:
    """One resolved alpha partition over canonical programs."""

    alpha: float
    in_distribution: tuple[Program, ...]
    compositional_ood: tuple[Program, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        overlap = set(self.in_distribution) & set(self.compositional_ood)
        if overlap:
            raise ValueError(f"programs appear in both splits: {sorted(overlap)}")

    @property
    def num_in_distribution(self) -> int:
        return len(self.in_distribution)

    @property
    def num_compositional_ood(self) -> int:
        return len(self.compositional_ood)


def split_programs(alpha: float, seed: int = PAPER_SPLIT_SEED) -> ProgramSplit:
    """Partition canonical programs into in-distribution and compositional OOD.

    Anchors and non-repeatable length-1 programs are always in distribution;
    the remaining programs are shuffled once and cut at ``int(n * alpha)``.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")

    splittable = splittable_programs()
    total = len(splittable)
    num_in_distribution = int(total * alpha)

    # Use the global RandomState for reproducible dataset generation.
    np.random.seed(seed)
    permutation = np.random.permutation(total)

    kept = sorted(permutation[:num_in_distribution])
    held_out = sorted(permutation[num_in_distribution:])

    in_distribution = (
        anchor_programs()
        + non_repeatable_unit_programs()
        + tuple(splittable[index] for index in kept)
    )
    compositional_ood = tuple(splittable[index] for index in held_out)
    return ProgramSplit(
        alpha=alpha,
        in_distribution=in_distribution,
        compositional_ood=compositional_ood,
    )


__all__ = [
    "PAPER_MAX_PROGRAM_LENGTH",
    "PAPER_MAX_PROGRAM_LENGTH_OOD",
    "PAPER_MIN_PROGRAM_LENGTH_OOD",
    "PAPER_SPLIT_SEED",
    "Program",
    "ProgramSplit",
    "REPEAT_LIMITS",
    "anchor_programs",
    "enumerate_length_ood_programs",
    "enumerate_programs",
    "is_valid_program",
    "non_repeatable_unit_programs",
    "split_programs",
    "splittable_programs",
]
