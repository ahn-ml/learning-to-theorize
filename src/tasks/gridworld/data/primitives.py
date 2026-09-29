"""The four movement primitives used in the GridWorld experiments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray


Grid = NDArray[np.integer[Any]]
ObjectInfo = dict[str, Any]
Primitive = Callable[[Grid, ObjectInfo], tuple[Grid, ObjectInfo]]


def _move(grid: Grid, object_info: ObjectInfo, axis: int, delta: int) -> tuple[Grid, ObjectInfo]:
    """Move object zero, preserving the boundary expansion."""

    grid_shape = grid.shape
    chain: list[int] = []
    visited: set[int] = set()
    queue = [0]
    while queue:
        object_index = queue.pop(0)
        if object_index in visited:
            continue
        visited.add(object_index)
        chain.append(object_index)
        for cell in object_info["grids"][object_index]:
            target = list(cell)
            target[axis] += delta
            target_cell = tuple(target)
            for candidate, cells in enumerate(object_info["grids"]):
                if candidate not in visited and target_cell in cells:
                    queue.append(candidate)

    boundary = 0 if delta < 0 else grid_shape[axis] - 1
    needs_expansion = any(
        cell[axis] == boundary
        for object_index in chain
        for cell in object_info["grids"][object_index]
    )
    new_shape = list(grid_shape)
    if needs_expansion:
        new_shape[axis] += 1
    new_grid = np.zeros(tuple(new_shape), dtype=grid.dtype)

    new_cells_by_object: list[list[tuple[int, int]]] = []
    for object_index, cells in enumerate(object_info["grids"]):
        new_cells: list[tuple[int, int]] = []
        for cell in cells:
            coordinates = list(cell)
            if object_index in chain:
                if not (needs_expansion and delta < 0):
                    coordinates[axis] += delta
            elif needs_expansion and delta < 0:
                coordinates[axis] += 1
            new_cells.append((coordinates[0], coordinates[1]))
        new_cells_by_object.append(new_cells)

    new_info: ObjectInfo = {
        "num_objects": object_info["num_objects"],
        "grids": new_cells_by_object,
        "labels": object_info["labels"].copy(),
    }
    for label, cells in zip(new_info["labels"], new_info["grids"]):
        for y, x in cells:
            new_grid[y, x] = label
    return new_grid, new_info


def move_up(grid: Grid, object_info: ObjectInfo) -> tuple[Grid, ObjectInfo]:
    return _move(grid, object_info, axis=0, delta=-1)


def move_right(grid: Grid, object_info: ObjectInfo) -> tuple[Grid, ObjectInfo]:
    return _move(grid, object_info, axis=1, delta=1)


def move_down(grid: Grid, object_info: ObjectInfo) -> tuple[Grid, ObjectInfo]:
    return _move(grid, object_info, axis=0, delta=1)


def move_left(grid: Grid, object_info: ObjectInfo) -> tuple[Grid, ObjectInfo]:
    return _move(grid, object_info, axis=1, delta=-1)


PRIMITIVE_NAMES = ("move_up", "move_right", "move_down", "move_left")
PRIMITIVES: dict[str, Primitive] = {
    "move_up": move_up,
    "move_right": move_right,
    "move_down": move_down,
    "move_left": move_left,
}
