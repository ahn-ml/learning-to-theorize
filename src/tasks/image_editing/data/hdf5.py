"""Read, write, and verify the Image Editing HDF5 artifacts.

Dataset layout:

.. code-block:: text

    /                      attrs: dataset_length, batch_size, shuffle
    /sample_<i>            attrs: transitions
    /sample_<i>/grids      uint8 (4, 32, 32, 3)   -- x1, y1, x2, y2
    /sample_<i>/answer     uint8 (32, 32, 3)      -- y2, duplicated for loaders

``transitions`` is stored as ``repr`` of a one-element list of programs, e.g.
``"[['brightness_plus', 'hue_plus']]"``.  It is parsed back with
:func:`ast.literal_eval`; the generator only ever writes literal lists of ``str``.

``answer`` duplicates ``grids[3]`` for loaders that read the query target directly.

This module is torch-free.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import h5py
import numpy as np

from tasks.image_editing.data.primitives import IMAGE_SIZE, NUM_CHANNELS
from tasks.image_editing.data.programs import Program


GRIDS_PER_EPISODE = 4
GRIDS_SHAPE = (GRIDS_PER_EPISODE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
ANSWER_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
COMPRESSION = "gzip"


@dataclass(frozen=True, slots=True)
class Episode:
    """One support pair and one query pair sharing a program."""

    grids: np.ndarray
    program: Program

    def __post_init__(self) -> None:
        if self.grids.shape != GRIDS_SHAPE or self.grids.dtype != np.uint8:
            raise ValueError(
                f"grids must be uint8 {GRIDS_SHAPE}, got "
                f"{self.grids.dtype} {tuple(self.grids.shape)}"
            )
        if not self.program:
            raise ValueError("an episode must carry a non-empty program")

    @property
    def answer(self) -> np.ndarray:
        """The query output, ``grids[3]``."""

        return self.grids[3]


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    """Root-level attributes of one artifact."""

    num_episodes: int
    batch_size: int
    shuffle: bool


def encode_transitions(program: Program) -> str:
    """Render a program the way the paper generator wrote it."""

    return str([list(program)])


def decode_transitions(raw: str | bytes) -> Program:
    """Parse the stored ``transitions`` attribute back into a program."""

    text = raw.decode() if isinstance(raw, bytes) else raw
    parsed = ast.literal_eval(text)
    if not isinstance(parsed, list) or len(parsed) != 1:
        raise ValueError(f"transitions must hold exactly one program, got {text!r}")
    program = parsed[0]
    if not isinstance(program, list) or not all(isinstance(x, str) for x in program):
        raise ValueError(f"program must be a list of primitive names, got {text!r}")
    return tuple(program)


def write_artifact(
    path: Path,
    episodes: Sequence[Episode],
    *,
    batch_size: int,
    shuffle: bool,
) -> None:
    """Write one artifact, refusing to clobber an existing file."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as handle:
        handle.attrs["dataset_length"] = len(episodes)
        handle.attrs["batch_size"] = batch_size
        handle.attrs["shuffle"] = shuffle
        for index, episode in enumerate(episodes):
            group = handle.create_group(f"sample_{index}")
            group.create_dataset("grids", data=episode.grids, compression=COMPRESSION)
            group.create_dataset(
                "answer", data=episode.answer, compression=COMPRESSION
            )
            group.attrs["transitions"] = encode_transitions(episode.program)


def read_metadata(path: Path) -> ArtifactMetadata:
    """Read the root attributes without touching per-episode data."""

    with h5py.File(Path(path), "r") as handle:
        return ArtifactMetadata(
            num_episodes=int(handle.attrs["dataset_length"]),
            batch_size=int(handle.attrs["batch_size"]),
            shuffle=bool(handle.attrs["shuffle"]),
        )


def read_episode(handle: h5py.File, index: int) -> Episode:
    """Read one episode from an open artifact."""

    group = handle[f"sample_{index}"]
    return Episode(
        grids=np.asarray(group["grids"], dtype=np.uint8),
        program=decode_transitions(group.attrs["transitions"]),
    )


def iter_episodes(path: Path, limit: int | None = None) -> Iterator[Episode]:
    """Yield episodes in stored order."""

    with h5py.File(Path(path), "r") as handle:
        total = int(handle.attrs["dataset_length"])
        stop = total if limit is None else min(total, limit)
        for index in range(stop):
            yield read_episode(handle, index)


def read_programs(path: Path) -> tuple[Program, ...]:
    """Read every stored program, in artifact order."""

    with h5py.File(Path(path), "r") as handle:
        total = int(handle.attrs["dataset_length"])
        return tuple(
            decode_transitions(handle[f"sample_{index}"].attrs["transitions"])
            for index in range(total)
        )


__all__ = [
    "ANSWER_SHAPE",
    "ArtifactMetadata",
    "COMPRESSION",
    "Episode",
    "GRIDS_PER_EPISODE",
    "GRIDS_SHAPE",
    "decode_transitions",
    "encode_transitions",
    "iter_episodes",
    "read_episode",
    "read_metadata",
    "read_programs",
    "write_artifact",
]
