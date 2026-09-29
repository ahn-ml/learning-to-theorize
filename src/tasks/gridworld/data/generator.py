"""Paper-exact GridWorld episode generation without PyTorch coupling."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from data import Episode, Example
from tasks.gridworld.data.primitives import PRIMITIVES, ObjectInfo
from tasks.gridworld.data.profiles import GridWorldProfile, SplitSpec
from tasks.gridworld.data.programs import (
    Program,
    alpha_split,
    canonical_programs,
)
from tasks.gridworld.data.shape_pool import Shape, ShapePool


Grid = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class GridWorldEpisodeMetadata:
    """Ground-truth information reserved for generation and diagnostics."""

    program: Program


@dataclass(frozen=True, slots=True)
class GeneratedEpisode:
    """An episode plus its artifact split and stable index."""

    split: SplitSpec
    index: int
    episode: Episode[Grid, GridWorldEpisodeMetadata]


class GridWorldGenerator:
    """Generate a complete profile in a reproducible random-draw order."""

    def __init__(self, profile: GridWorldProfile, shape_pool: ShapePool) -> None:
        self.profile = profile
        self.shape_pool = shape_pool
        all_programs = canonical_programs(
            profile.min_program_length,
            profile.max_program_length,
        )
        if profile.split_mode == "alpha":
            assert profile.alpha is not None
            self.train_programs, self.ood_programs = alpha_split(
                all_programs,
                profile.alpha,
                profile.split_seed,
            )
        else:
            self.train_programs = all_programs
            self.ood_programs = ()
        if not self.train_programs:
            raise ValueError("profile produced no training programs")
        self._rng = random.Random(profile.seed)
        self._seen: set[tuple[bytes, bytes, Program]] = set()

    def iter_profile(self) -> Iterator[GeneratedEpisode]:
        """Yield every split sequentially; iteration order is part of the contract."""

        self._rng.seed(self.profile.seed)
        self._seen.clear()
        for split in self.profile.splits:
            programs = self.ood_programs if split.use_ood_programs else self.train_programs
            if split.num_episodes and not programs:
                raise ValueError(f"split {split.name} requests episodes but has no programs")
            for index in range(split.num_episodes):
                yield GeneratedEpisode(split, index, self._generate_one(programs))

    def _generate_one(
        self,
        programs: tuple[Program, ...],
        max_try: int = 50,
    ) -> Episode[Grid, GridWorldEpisodeMetadata]:
        for _ in range(max_try):
            program = self._rng.choice(programs)
            episode = self._generate_episode(program, max_try=max_try)
            if episode is not None:
                return episode
        raise RuntimeError(f"failed to generate an episode after {max_try} attempts")

    def _generate_episode(
        self,
        program: Program,
        *,
        max_try: int,
    ) -> Episode[Grid, GridWorldEpisodeMetadata] | None:
        support: list[Example[Grid]] = []
        query: list[Example[Grid]] = []
        for _ in range(max_try):
            for _ in range(self.profile.num_supports + 1):
                success = False
                for _ in range(max_try):
                    # The last candidate controls the enclosing retry loop.
                    # Keep this reset here to preserve the random-draw sequence.
                    success = False
                    colors = [
                        self._rng.choice(list(range(1, self.profile.num_colors + 1)))
                        for _ in range(self.profile.num_objects)
                    ]
                    input_grid, object_info = self._generate_input(colors)
                    if input_grid is None or object_info is None:
                        continue
                    if any(len(shape) == 1 for shape in object_info["grids"]):
                        continue
                    # Paper-exact behavior: mark the current candidate
                    # successful before validating the transformed output.
                    success = True
                    output_grid = self._apply_program(input_grid, object_info, program)
                    if output_grid is None:
                        continue
                    if input_grid.shape == output_grid.shape and np.array_equal(
                        input_grid, output_grid
                    ):
                        continue
                    if any(dimension > self.profile.grid_size for dimension in output_grid.shape):
                        continue
                    key = (input_grid.tobytes(), output_grid.tobytes(), program)
                    if key in self._seen:
                        continue

                    input_grid = self._pad(input_grid)
                    output_grid = self._pad(output_grid)
                    example = Example(input=input_grid, output=output_grid)
                    if len(support) < self.profile.num_supports:
                        support.append(example)
                    else:
                        query.append(example)
                    self._seen.add(key)
                    break
                if not success:
                    break
            if len(support) == self.profile.num_supports and len(query) == 1:
                return Episode(
                    support=tuple(support),
                    query=tuple(query),
                    metadata=GridWorldEpisodeMetadata(program),
                )
        return None

    def _generate_input(
        self,
        colors: list[int],
        max_attempts: int = 50,
    ) -> tuple[Grid | None, ObjectInfo | None]:
        size = self.profile.grid_size
        for _ in range(max_attempts):
            grid = np.zeros((size, size), dtype=np.uint8)
            buffer_grid = np.zeros((size, size), dtype=bool)
            object_info: ObjectInfo = {"num_objects": 0, "grids": [], "labels": []}
            for object_index in range(self.profile.num_objects):
                shape = self._rng.choice(self.shape_pool.shapes)
                placed = self._place_shape(
                    grid,
                    shape,
                    colors[object_index],
                    buffer_grid,
                    max_attempts,
                )
                if placed is None:
                    break
                grid, cells, buffer_grid = placed
                object_info["num_objects"] += 1
                object_info["grids"].append(cells)
                object_info["labels"].append(colors[object_index])
            if object_info["num_objects"] == self.profile.num_objects:
                return grid, object_info
        return None, None

    def _place_shape(
        self,
        grid: Grid,
        shape: Shape,
        color: int,
        buffer_grid: NDArray[np.bool_],
        max_attempts: int,
    ) -> tuple[Grid, list[tuple[int, int]], NDArray[np.bool_]] | None:
        height, width = grid.shape
        min_y = min(y for y, _ in shape)
        min_x = min(x for _, x in shape)
        shape_height = max(y for y, _ in shape) - min_y + 1
        shape_width = max(x for _, x in shape) - min_x + 1
        if shape_height > height or shape_width > width:
            return None

        for _ in range(max_attempts):
            offset_y = self._rng.randint(0, height - shape_height)
            offset_x = self._rng.randint(0, width - shape_width)
            cells = [
                (y - min_y + offset_y, x - min_x + offset_x)
                for y, x in shape
            ]
            if not all(not buffer_grid[y, x] for y, x in cells):
                continue
            new_grid = grid.copy()
            new_buffer = buffer_grid.copy()
            for y, x in cells:
                new_grid[y, x] = color
                for delta_y in (-1, 0, 1):
                    for delta_x in (-1, 0, 1):
                        neighbor_y, neighbor_x = y + delta_y, x + delta_x
                        if 0 <= neighbor_y < height and 0 <= neighbor_x < width:
                            new_buffer[neighbor_y, neighbor_x] = True
            return new_grid, cells, new_buffer
        return None

    @staticmethod
    def _apply_program(grid: Grid, object_info: ObjectInfo, program: Program) -> Grid | None:
        current_grid = grid.copy()
        current_info: ObjectInfo = {
            "num_objects": object_info["num_objects"],
            "grids": [list(cells) for cells in object_info["grids"]],
            "labels": object_info["labels"].copy(),
        }
        try:
            for primitive_name in program:
                current_grid, current_info = PRIMITIVES[primitive_name](current_grid, current_info)
        except (IndexError, KeyError, ValueError):
            return None
        return current_grid

    def _pad(self, grid: Grid) -> Grid:
        pad_height = max(0, self.profile.grid_size - grid.shape[0])
        pad_width = max(0, self.profile.grid_size - grid.shape[1])
        return np.pad(
            grid,
            ((0, pad_height), (0, pad_width)),
            mode="constant",
            constant_values=255,
        )
