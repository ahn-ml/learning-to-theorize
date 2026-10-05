"""Write the Image Editing HDF5 artifacts.

Dataset layout:

.. code-block:: text

    /                      attrs: dataset_length, batch_size, shuffle
    /sample_<i>            attrs: transitions
    /sample_<i>/grids      uint8 (4, 32, 32, 3)   -- x1, y1, x2, y2
    /sample_<i>/answer     uint8 (32, 32, 3)      -- y2, a copy of grids[3]

``transitions`` is stored as ``repr`` of a one-element list of programs, e.g.
``"[['brightness_plus', 'hue_plus']]"``.

This module is torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from tasks.image_editing.data.primitives import IMAGE_SIZE, NUM_CHANNELS
from tasks.image_editing.data.programs import Program


GRIDS_PER_EPISODE = 4
GRIDS_SHAPE = (GRIDS_PER_EPISODE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
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


def encode_transitions(program: Program) -> str:
    """Render a program the way the paper generator wrote it."""

    return str([list(program)])


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


__all__ = [
    "COMPRESSION",
    "Episode",
    "GRIDS_PER_EPISODE",
    "GRIDS_SHAPE",
    "encode_transitions",
    "write_artifact",
]
