"""Write the HDF5 artifacts used by the GridWorld experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from data import Episode
from tasks.gridworld.data.generator import (
    GridWorldEpisodeMetadata,
    GridWorldGenerator,
)
from tasks.gridworld.data.profiles import GridWorldProfile, SplitSpec
from tasks.gridworld.data.shape_pool import ShapePool


def _h5py() -> Any:
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("h5py is required for GridWorld paper HDF5 data") from error
    return h5py


def episode_to_hdf5_arrays(
    episode: Episode[np.ndarray, GridWorldEpisodeMetadata],
    *,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Convert an explicit episode to the paper's flat-array HDF5 schema."""

    if len(episode.query) != 1:
        raise ValueError("GridWorld paper artifacts require exactly one query")
    if episode.metadata is None:
        raise ValueError("GridWorld paper artifacts require program metadata")
    grids = np.zeros((2 * len(episode.support) + 1, grid_size, grid_size), dtype=np.int32)
    for index, example in enumerate(episode.support):
        grids[2 * index] = example.input
        grids[2 * index + 1] = example.output
    grids[-1] = episode.query[0].input
    answer = np.asarray(episode.query[0].output, dtype=np.int64)
    transitions = str([list(episode.metadata.program), []])
    return grids, answer, transitions


class GridWorldHDF5Writer:
    """Write one paper HDF5 artifact without overwriting existing data."""

    def __init__(self, path: str | Path, split: SplitSpec) -> None:
        h5py = _h5py()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "x")
        self._file.attrs["dataset_length"] = split.num_episodes
        self._file.attrs["batch_size"] = split.batch_size
        self._file.attrs["shuffle"] = False
        self._next_index = 0
        self._expected_count = split.num_episodes

    def write(self, episode: Episode[np.ndarray, GridWorldEpisodeMetadata], grid_size: int) -> None:
        if self._next_index >= self._expected_count:
            raise RuntimeError("writer received more episodes than declared")
        grids, answer, transitions = episode_to_hdf5_arrays(episode, grid_size=grid_size)
        group = self._file.create_group(f"sample_{self._next_index}")
        group.create_dataset("grids", data=grids)
        group.create_dataset("answer", data=answer)
        group.attrs["transitions"] = transitions
        self._next_index += 1

    def close(self) -> None:
        if self._file is None:
            return
        count = self._next_index
        expected = self._expected_count
        self._file.close()
        self._file = None
        if count != expected:
            raise RuntimeError(f"writer closed after {count} episodes; expected {expected}")

    def abort(self) -> None:
        """Close an incomplete artifact after a generation error."""

        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "GridWorldHDF5Writer":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def write_profile(
    profile: GridWorldProfile,
    shape_pool: ShapePool,
    output_directory: str | Path,
) -> tuple[Path, ...]:
    """Generate every profile split in the paper's order and HDF5 format."""

    output_directory = Path(output_directory)
    paths = tuple(
        output_directory / profile.artifact_filename(split)
        for split in profile.splits
        if split.num_episodes
    )
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing artifacts: {existing}")

    generator = GridWorldGenerator(profile, shape_pool)
    current_split: SplitSpec | None = None
    writer: GridWorldHDF5Writer | None = None
    try:
        for generated in generator.iter_profile():
            if generated.split != current_split:
                if writer is not None:
                    writer.close()
                current_split = generated.split
                path = output_directory / profile.artifact_filename(generated.split)
                writer = GridWorldHDF5Writer(path, generated.split)
            assert writer is not None
            writer.write(generated.episode, profile.grid_size)
        if writer is not None:
            writer.close()
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    return paths
