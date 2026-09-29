"""Safe HDF5 input pipeline for GridWorld training and evaluation."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def _h5py() -> Any:
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("h5py is required for GridWorld observation data") from error
    return h5py


class GridWorldHDF5Dataset(Dataset[tuple[Tensor, Tensor]]):
    """Read a paper HDF5 artifact without sharing handles across workers."""

    def __init__(self, path: str | Path) -> None:
        self._file: Any | None = None
        self._owner_process_id: int | None = None
        self.path = Path(path).expanduser().resolve(strict=True)
        if self.path.suffix != ".h5":
            raise ValueError(f"expected an HDF5 artifact ending in .h5: {self.path}")
        h5py = _h5py()
        with h5py.File(self.path, "r") as file:
            self.num_episodes = int(file.attrs.get("dataset_length", -1))
            self.generation_batch_size = int(file.attrs.get("batch_size", -1))
            self.generation_shuffle = bool(file.attrs.get("shuffle", True))
            if self.num_episodes < 1:
                raise ValueError("paper HDF5 dataset_length must be positive")
            if self.generation_batch_size < 1:
                raise ValueError("paper HDF5 batch_size must be positive")
            if self.generation_shuffle:
                raise ValueError("paper GridWorld artifacts must record shuffle=false")
            self._read_sample(file, 0)
            if self.num_episodes > 1:
                self._read_sample(file, self.num_episodes - 1)

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if not 0 <= index < self.num_episodes:
            raise IndexError(index)
        grids, answer = self._read_sample(self._open_for_current_process(), index)
        return torch.from_numpy(grids), torch.from_numpy(answer)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._owner_process_id = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        state["_owner_process_id"] = None
        return state

    def __del__(self) -> None:
        self.close()

    def _open_for_current_process(self) -> Any:
        process_id = os.getpid()
        if self._file is not None and self._owner_process_id != process_id:
            self.close()
        if self._file is None:
            self._file = _h5py().File(self.path, "r")
            self._owner_process_id = process_id
        return self._file

    @staticmethod
    def _read_sample(file: Any, index: int) -> tuple[np.ndarray, np.ndarray]:
        group_name = f"sample_{index}"
        if group_name not in file:
            raise ValueError(f"paper HDF5 is missing {group_name}")
        group = file[group_name]
        if "grids" not in group or "answer" not in group:
            raise ValueError(f"{group_name} must contain grids and answer")
        grids = group["grids"][:]
        answer = group["answer"][:]
        if grids.shape != (3, 10, 10) or grids.dtype != np.dtype("int32"):
            raise ValueError(
                f"{group_name}/grids must have shape (3, 10, 10) and dtype int32"
            )
        if answer.shape != (10, 10) or answer.dtype != np.dtype("int64"):
            raise ValueError(
                f"{group_name}/answer must have shape (10, 10) and dtype int64"
            )
        if int(grids.min()) < 0 or int(grids.max()) > 8:
            raise ValueError(f"{group_name}/grids contains a color outside [0, 8]")
        if int(answer.min()) < 0 or int(answer.max()) > 8:
            raise ValueError(f"{group_name}/answer contains a color outside [0, 8]")
        return grids, answer


def collate_gridworld_observation_batch(
    samples: Sequence[tuple[Tensor, Tensor]],
) -> Tensor:
    """Collate paper HDF5 arrays into the model-facing four-grid tensor."""

    if not samples:
        raise ValueError("cannot collate an empty GridWorld batch")
    grids = torch.stack([sample[0] for sample in samples])
    answers = torch.stack([sample[1] for sample in samples])
    return assemble_paper_episode_grids(grids, answers)


def assemble_paper_episode_grids(grids: Tensor, answers: Tensor) -> Tensor:
    """Turn paper HDF5 ``grids``/``answer`` arrays into four-grid episodes."""

    if grids.ndim != 4 or tuple(grids.shape[1:]) != (3, 10, 10):
        raise ValueError(
            "expected paper grids shaped (B, 3, 10, 10), "
            f"got {tuple(grids.shape)}"
        )
    if answers.ndim != 3 or tuple(answers.shape[1:]) != (10, 10):
        raise ValueError(
            "expected paper answers shaped (B, 10, 10), "
            f"got {tuple(answers.shape)}"
        )
    if grids.shape[0] != answers.shape[0]:
        raise ValueError("paper grids and answers must have the same batch size")
    if (
        grids.is_floating_point()
        or grids.is_complex()
        or answers.is_floating_point()
        or answers.is_complex()
    ):
        raise TypeError("GridWorld observations must contain integer color indices")
    return torch.cat([grids, answers.unsqueeze(1)], dim=1)


def seed_observation_worker(worker_id: int) -> None:
    """Seed NumPy and Python random generators for each worker."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class _ProcessSafeHDF5DataLoader(DataLoader[Tensor]):
    """Ensure the parent has no live HDF5 handle when workers are created."""

    def __iter__(self) -> Any:
        dataset = self.dataset
        if isinstance(dataset, GridWorldHDF5Dataset):
            dataset.close()
        return super().__iter__()


@dataclass(frozen=True, slots=True)
class GridWorldObservationLoaderConfig:
    """One rank's explicit DataLoader and DistributedSampler settings."""

    per_rank_batch_size: int = 512
    world_size: int = 1
    rank: int = 0
    num_workers: int = 4
    seed: int = 42
    pin_memory: bool = True

    def __post_init__(self) -> None:
        if self.per_rank_batch_size < 1 or self.world_size < 1:
            raise ValueError("batch size and world size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        if self.num_workers < 0 or self.seed < 0:
            raise ValueError("num_workers and seed must be non-negative")


@dataclass(frozen=True, slots=True)
class GridWorldObservationLoader:
    """DataLoader plus the sampler state that the training loop must advance."""

    dataset: GridWorldHDF5Dataset
    sampler: DistributedSampler[tuple[Tensor, Tensor]]
    dataloader: DataLoader[Tensor]

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.sampler.set_epoch(epoch)


def build_gridworld_observation_loader(
    path: str | Path,
    config: GridWorldObservationLoaderConfig,
    *,
    shuffle: bool,
) -> GridWorldObservationLoader:
    """Build the explicit equivalent of the paper run's Fabric data setup."""

    dataset = GridWorldHDF5Dataset(path)
    sampler = DistributedSampler(
        dataset,
        num_replicas=config.world_size,
        rank=config.rank,
        shuffle=shuffle,
        seed=config.seed,
        drop_last=False,
    )
    generator = torch.Generator().manual_seed(config.seed)
    dataloader = _ProcessSafeHDF5DataLoader(
        dataset,
        batch_size=config.per_rank_batch_size,
        sampler=sampler,
        collate_fn=collate_gridworld_observation_batch,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        worker_init_fn=seed_observation_worker,
        generator=generator,
        drop_last=False,
        persistent_workers=False,
    )
    return GridWorldObservationLoader(dataset, sampler, dataloader)
