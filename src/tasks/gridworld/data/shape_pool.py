"""Deterministic generation of the shape pool used by the paper."""

from __future__ import annotations

import hashlib
import io
import pickle
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


Cell = tuple[int, int]
Shape = tuple[Cell, ...]
CANONICAL_SHAPE_POOL_SHA256 = (
    "0edbaaef204e3c7e160dcee3e765b2ffdc2dac424724fbdc89ad514e719e6754"
)


@dataclass(frozen=True, slots=True)
class ShapePool:
    """A normalized, deterministically ordered collection of shapes."""

    shapes: tuple[Shape, ...]
    total_attempts: int | None = None
    filtered_candidates: int | None = None

    def __post_init__(self) -> None:
        if not self.shapes:
            raise ValueError("shape pool must not be empty")
        if len(self.shapes) != len(set(self.shapes)):
            raise ValueError("shape pool contains duplicates")


class _BuiltinsOnlyUnpickler(pickle.Unpickler):
    """Reject pickle opcodes that construct imported Python objects."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"global object loading is disabled: {module}.{name}")


def _well_connected(shape: Shape, min_adjacent_faces: int) -> bool:
    if len(shape) <= 1:
        return False
    cells = set(shape)
    return all(
        sum((y + dy, x + dx) in cells for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))
        >= min_adjacent_faces
        for y, x in shape
    )


def _generate_single_shape(
    python_rng: random.Random,
    numpy_rng: np.random.RandomState,
    *,
    grid_size: int,
    budget_ratio: float,
    wall_margin: int,
    min_size: int,
    expand_probability: float,
) -> Shape | None:
    buffer_grid = np.zeros((grid_size, grid_size), dtype=bool)
    for y in range(grid_size):
        for x in range(grid_size):
            if (
                y < wall_margin
                or y >= grid_size - wall_margin
                or x < wall_margin
                or x >= grid_size - wall_margin
            ):
                buffer_grid[y, x] = True

    valid_positions = [
        (y, x)
        for y in range(wall_margin, grid_size - wall_margin)
        for x in range(wall_margin, grid_size - wall_margin)
        if not buffer_grid[y, x]
    ]
    if not valid_positions:
        return None

    start = python_rng.choice(valid_positions)
    max_object_size = int(grid_size * grid_size * budget_ratio)
    budget = python_rng.randint(min_size, max_object_size)
    visited: set[Cell] = set()
    queue = [start]
    object_cells: list[Cell] = []

    while queue and len(object_cells) < budget:
        y, x = queue.pop(0)
        if not (
            wall_margin <= y < grid_size - wall_margin
            and wall_margin <= x < grid_size - wall_margin
        ):
            continue
        if (y, x) in visited or buffer_grid[y, x]:
            continue

        visited.add((y, x))
        object_cells.append((y, x))
        neighbors = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
        if numpy_rng.random_sample() < 0.5:
            neighbors += [
                (y - 1, x - 1),
                (y - 1, x + 1),
                (y + 1, x - 1),
                (y + 1, x + 1),
            ]
        python_rng.shuffle(neighbors)
        for neighbor in neighbors:
            if python_rng.random() < expand_probability:
                queue.append(neighbor)

    if len(object_cells) < min_size:
        return None
    min_y = min(y for y, _ in object_cells)
    min_x = min(x for _, x in object_cells)
    return tuple(sorted((y - min_y, x - min_x) for y, x in object_cells))


def generate_shape_pool(
    *,
    num_shapes: int,
    grid_size: int,
    budget_ratio: float,
    wall_margin: int,
    min_size: int,
    min_adjacent_faces: int,
    seed: int,
    expand_probability: float = 1.0,
    max_attempts_per_shape: int = 100,
) -> ShapePool:
    """Generate shapes using Python and NumPy random generators."""

    if num_shapes < 1:
        raise ValueError("num_shapes must be positive")
    if grid_size < 1 or min_size < 1:
        raise ValueError("grid_size and min_size must be positive")
    if not 0.0 < budget_ratio <= 1.0:
        raise ValueError("budget_ratio must be in (0, 1]")
    if int(grid_size * grid_size * budget_ratio) < min_size:
        raise ValueError("budget_ratio does not permit min_size cells")
    if wall_margin < 0 or wall_margin * 2 >= grid_size:
        raise ValueError("wall_margin leaves no usable grid")
    if min_adjacent_faces < 0 or min_adjacent_faces > 4:
        raise ValueError("min_adjacent_faces must be in [0, 4]")

    python_rng = random.Random(seed)
    numpy_rng = np.random.RandomState(seed)
    unique_shapes: set[Shape] = set()
    total_attempts = 0
    filtered_candidates = 0
    while len(unique_shapes) < num_shapes:
        for _ in range(max_attempts_per_shape):
            total_attempts += 1
            shape = _generate_single_shape(
                python_rng,
                numpy_rng,
                grid_size=grid_size,
                budget_ratio=budget_ratio,
                wall_margin=wall_margin,
                min_size=min_size,
                expand_probability=expand_probability,
            )
            if shape is None or shape in unique_shapes:
                continue
            if _well_connected(shape, min_adjacent_faces):
                unique_shapes.add(shape)
                break
            filtered_candidates += 1

    shapes = tuple(sorted(unique_shapes, key=lambda shape: (len(shape), shape)))
    return ShapePool(shapes, total_attempts, filtered_candidates)


@lru_cache(maxsize=1)
def generate_canonical_shape_pool() -> ShapePool:
    """Generate the exact 1,000-shape pool used by every paper dataset."""

    return generate_shape_pool(
        num_shapes=1_000,
        grid_size=10,
        budget_ratio=0.20,
        wall_margin=0,
        min_size=6,
        min_adjacent_faces=2,
        seed=1127,
    )


def shape_pool_pickle_bytes(shape_pool: ShapePool) -> bytes:
    """Serialize with the exact paper protocol and built-in-only schema."""

    payload = {
        "shapes": [list(shape) for shape in shape_pool.shapes],
        "num_shapes": len(shape_pool.shapes),
        "metadata": {
            "description": "Pre-generated shape pool for TaskGenerator",
            "format": "Each shape is a list of (y, x) coordinates normalized to origin (0,0)",
        },
    }
    return pickle.dumps(payload, protocol=4)


def save_shape_pool(path: str | Path, shape_pool: ShapePool) -> str:
    """Write a paper-compatible pickle, refusing to overwrite an artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = shape_pool_pickle_bytes(shape_pool)
    with destination.open("xb") as file:
        file.write(data)
    return hashlib.sha256(data).hexdigest()


def _read_pickle(source: BinaryIO) -> Any:
    return _BuiltinsOnlyUnpickler(source).load()


def load_shape_pool(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> ShapePool:
    """Load and validate the paper's built-in-only shape-pool pickle."""

    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"shape-pool SHA-256 mismatch: expected {expected_sha256}, got {digest}")
    payload = _read_pickle(io.BytesIO(data))
    if not isinstance(payload, dict) or set(payload) != {"shapes", "num_shapes", "metadata"}:
        raise ValueError("unexpected shape-pool schema")
    shapes = tuple(tuple((int(y), int(x)) for y, x in shape) for shape in payload["shapes"])
    if payload["num_shapes"] != len(shapes):
        raise ValueError("shape-pool count metadata does not match its contents")
    return ShapePool(shapes)
