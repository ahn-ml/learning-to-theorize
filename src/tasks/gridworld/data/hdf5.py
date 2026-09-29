"""Read, write, and verify the HDF5 artifacts used by the paper experiments."""

from __future__ import annotations

import ast
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data import Episode, Example
from tasks.gridworld.data.generator import (
    GeneratedEpisode,
    GridWorldEpisodeMetadata,
    GridWorldGenerator,
)
from tasks.gridworld.data.profiles import GridWorldProfile, SplitSpec
from tasks.gridworld.data.shape_pool import ShapePool


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Summary returned by a semantic artifact verification."""

    profile: str
    compared_by_split: dict[str, int]
    complete: bool


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


def hdf5_arrays_to_episode(
    grids: np.ndarray,
    answer: np.ndarray,
    transitions: str,
) -> Episode[np.ndarray, GridWorldEpisodeMetadata]:
    """Parse one paper HDF5 sample without using ``eval``."""

    if grids.ndim != 3 or grids.shape[0] < 3 or grids.shape[0] % 2 != 1:
        raise ValueError(f"invalid GridWorld HDF5 grids shape: {grids.shape}")
    parsed = ast.literal_eval(transitions)
    if (
        not isinstance(parsed, list)
        or len(parsed) != 2
        or not isinstance(parsed[0], list)
        or parsed[1] != []
        or not all(isinstance(name, str) for name in parsed[0])
    ):
        raise ValueError("invalid GridWorld HDF5 transitions metadata")
    support = tuple(
        Example(input=grids[index].copy(), output=grids[index + 1].copy())
        for index in range(0, grids.shape[0] - 1, 2)
    )
    query = (Example(input=grids[-1].copy(), output=answer.copy()),)
    return Episode(
        support=support,
        query=query,
        metadata=GridWorldEpisodeMetadata(tuple(parsed[0])),
    )


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


def _check_root_contract(file: Any, split: SplitSpec) -> None:
    if int(file.attrs.get("dataset_length", -1)) != split.num_episodes:
        raise AssertionError(f"{split.name}: dataset_length mismatch")
    if int(file.attrs.get("batch_size", -1)) != split.batch_size:
        raise AssertionError(f"{split.name}: batch_size mismatch")
    if bool(file.attrs.get("shuffle", True)):
        raise AssertionError(f"{split.name}: shuffle must be false")


def verify_profile(
    profile: GridWorldProfile,
    shape_pool: ShapePool,
    data_directory: str | Path,
    *,
    max_total_episodes: int | None = None,
) -> VerificationResult:
    """Regenerate and compare canonical content in sequential RNG order."""

    if max_total_episodes is not None and max_total_episodes < 1:
        raise ValueError("max_total_episodes must be positive")
    h5py = _h5py()
    data_directory = Path(data_directory)
    compared = {split.name: 0 for split in profile.splits}
    split_by_name = {split.name: split for split in profile.splits}
    with ExitStack() as stack:
        files = {
            split.name: stack.enter_context(
                h5py.File(data_directory / profile.artifact_filename(split), "r")
            )
            for split in profile.splits
            if split.num_episodes
        }
        for name, file in files.items():
            _check_root_contract(file, split_by_name[name])

        total = 0
        for generated in GridWorldGenerator(profile, shape_pool).iter_profile():
            if max_total_episodes is not None and total >= max_total_episodes:
                return VerificationResult(profile.name, compared, complete=False)
            _compare_generated(files[generated.split.name], generated, profile.grid_size)
            compared[generated.split.name] += 1
            total += 1
    return VerificationResult(profile.name, compared, complete=True)


def _compare_generated(file: Any, generated: GeneratedEpisode, grid_size: int) -> None:
    group_name = f"sample_{generated.index}"
    if group_name not in file:
        raise AssertionError(f"{generated.split.name}/{group_name}: missing sample")
    expected_grids, expected_answer, expected_transitions = episode_to_hdf5_arrays(
        generated.episode,
        grid_size=grid_size,
    )
    group = file[group_name]
    actual_grids = group["grids"][:]
    actual_answer = group["answer"][:]
    actual_transitions = group.attrs["transitions"]
    if actual_grids.dtype != expected_grids.dtype or not np.array_equal(
        actual_grids, expected_grids
    ):
        raise AssertionError(f"{generated.split.name}/{group_name}: grids mismatch")
    if actual_answer.dtype != expected_answer.dtype or not np.array_equal(
        actual_answer, expected_answer
    ):
        raise AssertionError(f"{generated.split.name}/{group_name}: answer mismatch")
    if actual_transitions != expected_transitions:
        raise AssertionError(f"{generated.split.name}/{group_name}: transitions mismatch")
