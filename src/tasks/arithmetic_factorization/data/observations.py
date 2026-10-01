"""Unpaired four-digit observations for digit-autoencoder pretraining."""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

OBSERVATION_FILENAME = "arithmetic_observations.h5"


def generate_observations(path: Path) -> None:
    """Store each number from 0000 through 9999 once, without transformation pairs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    numbers = np.arange(10000, dtype=np.int64)
    digits = ((numbers[:, None] // np.array([1000, 100, 10, 1])) % 10).astype(np.uint8)
    with h5py.File(path, "x") as file:
        file.create_dataset("observations", data=digits[:, None, :])


def build_observation_loader(path: Path, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    with h5py.File(path, "r") as file:
        observations = file["observations"][:]
    if observations.shape != (10000, 1, 4) or not np.issubdtype(observations.dtype, np.integer):
        raise ValueError("expected 10000 individual four-digit observations")
    if observations.min() < 0 or observations.max() > 9:
        raise ValueError("observations must contain digits 0 through 9")
    return DataLoader(
        torch.from_numpy(observations).long(), batch_size=batch_size,
        shuffle=shuffle, generator=torch.Generator().manual_seed(seed),
        num_workers=0, pin_memory=True,
    )
