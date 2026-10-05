"""Deterministic support/query episode generation for Image Editing.

One episode holds two CIFAR-10 images edited by the *same* program: the first
pair is the support demonstration, the second is the query.  A pair is kept only
if every step visibly changes the image and the final result visibly differs
from the input, which is what stops the model from being rewarded for
near-identity programs.

Generation uses one ``np.random.seed`` per dataset and consumes programs in
canonical order. Each attempt draws from :func:`numpy.random.choice`, including
rejected attempts, so the random-draw sequence remains reproducible.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

from tasks.image_editing.data.hdf5 import Episode
from tasks.image_editing.data.primitives import PRIMITIVES
from tasks.image_editing.data.programs import Program


#: Per-pixel difference below which a channel counts as unchanged.
CHANGE_PIXEL_THRESHOLD = 15.0 / 255.0

#: Minimum mean squared difference for a step to count as a real edit.
CHANGE_MSE_THRESHOLD = 0.005

#: Maximum fraction of unchanged pixels tolerated after one step.
CHANGE_PIXEL_ACCURACY_THRESHOLD = 0.775

#: Same bound applied to the input/output pair as a whole.
FINAL_PIXEL_ACCURACY_THRESHOLD = 0.775

#: Image-pair resampling attempts inside one episode attempt.
PAIR_ATTEMPTS = 100

#: Length-OOD generation resamples only once, which is why its artifacts fall
#: about 15% short of the requested episode count.
LENGTH_OOD_PAIR_ATTEMPTS = 1

#: Episode attempts per program are capped at this multiple of the target.
ATTEMPT_MULTIPLIER = 10


def load_cifar10(cifar_dir: Path, split: str) -> np.ndarray:
    """Load raw CIFAR-10 images as ``uint8`` ``(N, 32, 32, 3)``."""

    cifar_dir = Path(cifar_dir)
    if split == "train":
        names = [f"data_batch_{index}" for index in range(1, 6)]
    elif split == "test":
        names = ["test_batch"]
    else:
        raise ValueError(f"unknown CIFAR-10 split {split!r}; use 'train' or 'test'")

    batches = []
    for name in names:
        with (cifar_dir / name).open("rb") as stream:
            payload = pickle.load(stream, encoding="bytes")
        batches.append(payload[b"data"])
    data = np.concatenate(batches, axis=0)
    return data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


def _significantly_different(before: np.ndarray, after: np.ndarray, bound: float) -> bool:
    previous = before.astype(np.float32) / 255.0
    current = after.astype(np.float32) / 255.0
    unchanged = (np.abs(current - previous) < CHANGE_PIXEL_THRESHOLD).all(axis=-1)
    pixel_accuracy = unchanged.mean()
    mse = np.mean((current - previous) ** 2)
    return bool(pixel_accuracy < bound and mse >= CHANGE_MSE_THRESHOLD)


def apply_and_check(image: np.ndarray, program: Program) -> tuple[np.ndarray, bool]:
    """Run a program, rejecting it if any step or the whole edit is invisible."""

    current = image.copy()
    for name in program:
        try:
            primitive = PRIMITIVES[name]
        except KeyError as error:
            raise ValueError(f"unknown image-editing primitive {name!r}") from error
        previous = current.copy()
        current = primitive(current)
        if not _significantly_different(
            previous, current, CHANGE_PIXEL_ACCURACY_THRESHOLD
        ):
            return current, False

    changed = _significantly_different(
        image, current, FINAL_PIXEL_ACCURACY_THRESHOLD
    )
    return current, changed


def _sample_episode(
    images: np.ndarray,
    program: Program,
    pair_attempts: int,
) -> Episode | None:
    """Draw two images and edit both, or give up after ``pair_attempts``."""

    num_images = len(images)
    for _ in range(pair_attempts):
        first, second = np.random.choice(num_images, size=2, replace=False)
        support_input = images[first]
        query_input = images[second]
        support_output, support_ok = apply_and_check(support_input, program)
        query_output, query_ok = apply_and_check(query_input, program)
        if support_ok and query_ok:
            grids = np.stack(
                [support_input, support_output, query_input, query_output], axis=0
            ).astype(np.uint8)
            return Episode(grids=grids, program=program)
    return None


def generate_episodes(
    images: np.ndarray,
    programs: Sequence[Program],
    episodes_per_program: int,
    *,
    seed: int,
    pair_attempts: int = PAIR_ATTEMPTS,
) -> tuple[list[Episode], list[tuple[Program, int]]]:
    """Generate episodes for each program in order, under one seeded stream.

    Also returns ``(program, produced)`` for every program that fell short of
    ``episodes_per_program``, which is expected for the length-OOD profile.
    """

    if episodes_per_program < 1:
        raise ValueError("episodes_per_program must be positive")

    # Use the global RandomState for reproducible dataset generation.
    np.random.seed(seed)

    episodes: list[Episode] = []
    shortfalls: list[tuple[Program, int]] = []
    attempt_budget = episodes_per_program * ATTEMPT_MULTIPLIER

    for program in programs:
        produced = 0
        attempts = 0
        while produced < episodes_per_program and attempts < attempt_budget:
            attempts += 1
            episode = _sample_episode(images, program, pair_attempts)
            if episode is not None:
                episodes.append(episode)
                produced += 1
        if produced < episodes_per_program:
            shortfalls.append((program, produced))
    return episodes, shortfalls


__all__ = [
    "ATTEMPT_MULTIPLIER",
    "CHANGE_MSE_THRESHOLD",
    "CHANGE_PIXEL_ACCURACY_THRESHOLD",
    "CHANGE_PIXEL_THRESHOLD",
    "FINAL_PIXEL_ACCURACY_THRESHOLD",
    "LENGTH_OOD_PAIR_ATTEMPTS",
    "PAIR_ATTEMPTS",
    "apply_and_check",
    "generate_episodes",
    "load_cifar10",
]
