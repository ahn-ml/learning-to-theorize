"""Process-safe, model-facing loader for the Image Editing HDF5 artifacts.

``grids`` are ``uint8`` ``(B, 4, 32, 32, 3)``, ordered support input, support
output, query input, query output.  Normalisation and channel reordering belong
to the model's encoder, not here, so the loader hands back the stored bytes
unchanged.

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


@dataclass(frozen=True, slots=True)
class EpisodeBatch:
    """One collated batch of episodes."""

    grids: torch.Tensor

    def to(self, device: torch.device | str) -> "EpisodeBatch":
        return EpisodeBatch(grids=self.grids.to(device))

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

    def __getitem__(self, index: int) -> torch.Tensor:
        if not 0 <= index < self._length:
            raise IndexError(f"index {index} out of range for {self._length} episodes")
        group = self.handle[f"sample_{index}"]
        return torch.from_numpy(np.asarray(group["grids"], dtype=np.uint8))

    def __getstate__(self) -> dict:
        # Never pickle an open h5py handle into a DataLoader worker.
        return {"path": self.path, "_length": self._length, "_handle": None}

    def __setstate__(self, state: dict) -> None:
        self.path = state["path"]
        self._length = state["_length"]
        self._handle = None


def collate_episodes(batch: Sequence[torch.Tensor]) -> EpisodeBatch:
    """Stack episodes without changing dtype or layout."""

    return EpisodeBatch(grids=torch.stack(list(batch)))


__all__ = [
    "EpisodeBatch",
    "ImageEditingDataset",
    "collate_episodes",
]
