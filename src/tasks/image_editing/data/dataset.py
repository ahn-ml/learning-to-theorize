"""Process-safe, model-facing loader for the Image Editing HDF5 artifacts.

This is the only data module coupled to PyTorch.  Model-facing tensor semantics:

* ``grids``  -- ``uint8`` ``(B, 4, 32, 32, 3)``, ordered support input, support
  output, query input, query output;
* ``answer`` -- ``uint8`` ``(B, 32, 32, 3)``, a copy of ``grids[:, 3]`` kept for
  artifact fidelity; and
* ``programs`` -- the ground-truth program per episode, used for analysis only.

Normalisation and channel reordering belong to the model's encoder, not here, so
the loader hands back the stored bytes unchanged.

The HDF5 handle is opened lazily per process.  Opening it in ``__init__`` would
share one handle across forked DataLoader workers, which h5py does not support.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from tasks.image_editing.data.hdf5 import decode_transitions
from tasks.image_editing.data.programs import Program


@dataclass(frozen=True, slots=True)
class EpisodeBatch:
    """One collated batch of episodes."""

    grids: torch.Tensor
    answer: torch.Tensor
    programs: tuple[Program, ...]

    def to(self, device: torch.device | str) -> "EpisodeBatch":
        return EpisodeBatch(
            grids=self.grids.to(device),
            answer=self.answer.to(device),
            programs=self.programs,
        )

    def __len__(self) -> int:
        return self.grids.shape[0]


class ImageEditingDataset(Dataset):
    """Read episodes from one released artifact."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"artifact not found: {self.path}")
        with h5py.File(self.path, "r") as handle:
            self._length = int(handle.attrs["dataset_length"])
        self._handle: h5py.File | None = None

    def __len__(self) -> int:
        return self._length

    @property
    def handle(self) -> h5py.File:
        """Open the artifact on first use in this process."""

        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, Program]:
        if not 0 <= index < self._length:
            raise IndexError(f"index {index} out of range for {self._length} episodes")
        group = self.handle[f"sample_{index}"]
        grids = torch.from_numpy(np.asarray(group["grids"], dtype=np.uint8))
        answer = torch.from_numpy(np.asarray(group["answer"], dtype=np.uint8))
        program = decode_transitions(group.attrs["transitions"])
        return grids, answer, program

    def __getstate__(self) -> dict:
        # Never pickle an open h5py handle into a DataLoader worker.
        state = {"path": self.path, "_length": self._length, "_handle": None}
        return state

    def __setstate__(self, state: dict) -> None:
        self.path = state["path"]
        self._length = state["_length"]
        self._handle = None

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def collate_episodes(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor, Program]],
) -> EpisodeBatch:
    """Stack episodes without changing dtype or layout."""

    grids = torch.stack([item[0] for item in batch])
    answer = torch.stack([item[1] for item in batch])
    programs = tuple(item[2] for item in batch)
    return EpisodeBatch(grids=grids, answer=answer, programs=programs)


__all__ = [
    "EpisodeBatch",
    "ImageEditingDataset",
    "collate_episodes",
]
