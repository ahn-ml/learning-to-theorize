"""Model-facing HDF5 loading for arithmetic factorization episodes."""

from __future__ import annotations

import ast
import random
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# Grids are stored with this sentinel in padded positions.
PAD_SENTINEL = 255

# Loader worker processes per rank.
PAPER_DATA_LOADER_WORKERS = 4


class ArithmeticEpisodes(Dataset):
    """One released HDF5 artifact of support/query factorization pairs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.suffix != ".h5":
            raise ValueError(f"unsupported data format: {self.path}")
        self._file = h5py.File(self.path, "r")
        self.length = int(self._file.attrs["dataset_length"])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        group = self._file[f"sample_{index}"]
        sample: dict[str, Any] = {}
        for key in group.keys():
            sample[key] = torch.from_numpy(group[key][:])
        for key in group.attrs.keys():
            value = group.attrs[key]
            if (
                isinstance(value, (bytes, str))
                and str(value).startswith("[")
                and str(value).endswith("]")
            ):
                sample[key] = ast.literal_eval(value)
            else:
                sample[key] = value
        return sample


def collate_episodes(batch: Sequence[dict[str, Any]]) -> tuple[Tensor, Tensor, list]:
    """Stack episode grids and answers into one batch."""

    grids = torch.stack([item["grids"] for item in batch])
    answers = torch.stack([item["answer"] for item in batch])
    return grids, answers, []


def unpack_batch(batch: tuple[Tensor, Tensor, list], device: Any) -> tuple[Tensor, list]:
    """Assemble the interleaved support/query tensor the model consumes.

    The returned tensor is ``(B, 4, 1, digits)`` ordered as support input,
    support output, query input, query target. The model reads even rows as
    support and odd rows as query after flattening.
    """

    input_grids, target_grids, transitions = batch
    supports = input_grids[:, :-1, :, :].to(device)
    query = input_grids[:, -1:, :, :].to(device)
    target = target_grids.unsqueeze(1).to(device)

    supports = torch.where(supports == PAD_SENTINEL, torch.zeros_like(supports), supports)
    query = torch.where(query == PAD_SENTINEL, torch.zeros_like(query), query)
    target = torch.where(target == PAD_SENTINEL, torch.zeros_like(target), target)

    return torch.cat([supports, query, target], dim=1), transitions


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(
    path: str | Path,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
    num_workers: int = PAPER_DATA_LOADER_WORKERS,
) -> DataLoader:
    """Build a reproducible loader over one released artifact."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ArithmeticEpisodes(path),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_episodes,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


__all__ = [
    "PAD_SENTINEL",
    "PAPER_DATA_LOADER_WORKERS",
    "ArithmeticEpisodes",
    "build_dataloader",
    "collate_episodes",
    "unpack_batch",
]
