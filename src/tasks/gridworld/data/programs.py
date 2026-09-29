"""Canonical program enumeration and compositional-OOD splitting."""

from __future__ import annotations

import copy
import math
import random
from functools import lru_cache
from itertools import product

import numpy as np

from tasks.gridworld.data.primitives import (
    PRIMITIVES,
    PRIMITIVE_NAMES,
    ObjectInfo,
)


Program = tuple[str, ...]
ANCHORS: tuple[Program, ...] = (
    ("move_up", "move_up", "move_up"),
    ("move_down", "move_down", "move_down"),
    ("move_right", "move_right", "move_right"),
    ("move_left", "move_left", "move_left"),
)


def _fingerprint_cases() -> tuple[tuple[np.ndarray, ObjectInfo], ...]:
    specs = (
        ((20, 20), ((8, 9), (9, 9), (10, 9), (10, 10)), 1),
        ((20, 20), ((7, 10), (7, 11), (8, 10), (9, 10), (9, 11)), 2),
        ((20, 20), ((8, 12), (9, 11), (9, 12), (10, 10), (10, 11)), 3),
        ((20, 20), ((9, 8), (9, 9), (9, 10), (10, 10)), 4),
        ((15, 25), ((6, 12), (7, 11), (7, 12), (7, 13), (8, 12)), 5),
    )
    cases: list[tuple[np.ndarray, ObjectInfo]] = []
    for shape, cells, label in specs:
        grid = np.zeros(shape, dtype=np.int32)
        for y, x in cells:
            grid[y, x] = label
        cases.append(
            (
                grid,
                {"num_objects": 1, "grids": [list(cells)], "labels": [label]},
            )
        )
    return tuple(cases)


def _apply_program(grid: np.ndarray, object_info: ObjectInfo, program: Program) -> np.ndarray:
    current_grid = grid.copy()
    current_info = copy.deepcopy(object_info)
    for primitive_name in program:
        current_grid, current_info = PRIMITIVES[primitive_name](current_grid, current_info)
    return current_grid


def _fingerprint(program: Program) -> tuple[tuple[tuple[int, ...], bytes], ...]:
    return tuple(
        (result.shape, result.tobytes())
        for grid, object_info in _fingerprint_cases()
        for result in (_apply_program(grid, object_info, program),)
    )


@lru_cache(maxsize=None)
def canonical_programs(min_length: int, max_length: int) -> tuple[Program, ...]:
    """Return the shortest program representatives for a length range."""

    if min_length < 1:
        raise ValueError("min_length must be positive")
    if max_length < min_length:
        raise ValueError("max_length must not be smaller than min_length")

    identity = tuple((grid.shape, grid.tobytes()) for grid, _ in _fingerprint_cases())
    groups: dict[tuple[tuple[tuple[int, ...], bytes], ...], list[Program]] = {}
    for length in range(1, max_length + 1):
        for program in product(PRIMITIVE_NAMES, repeat=length):
            groups.setdefault(_fingerprint(program), []).append(program)

    representatives: list[Program] = []
    for fingerprint, equivalent_programs in groups.items():
        if fingerprint == identity:
            continue
        representative = min(equivalent_programs, key=lambda item: (len(item), item))
        if min_length <= len(representative) <= max_length:
            representatives.append(representative)
    return tuple(sorted(representatives, key=lambda item: (len(item), item)))


def alpha_split(
    programs: tuple[Program, ...],
    alpha: float,
    seed: int = 1127,
) -> tuple[tuple[Program, ...], tuple[Program, ...]]:
    """Apply the paper's four-anchor plus non-anchor subsampling rule."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    program_set = set(programs)
    missing = [anchor for anchor in ANCHORS if anchor not in program_set]
    if missing:
        raise ValueError(f"program set does not contain the required anchors: {missing}")

    train = set(ANCHORS)
    remaining = [program for program in programs if program not in train]
    additional_count = math.ceil(alpha * len(remaining))
    train.update(random.Random(seed).sample(remaining, additional_count))
    train_programs = tuple(sorted(train, key=lambda item: (len(item), item)))
    ood_programs = tuple(
        sorted(
            (program for program in programs if program not in train),
            key=lambda item: (len(item), item),
        )
    )
    return train_programs, ood_programs
