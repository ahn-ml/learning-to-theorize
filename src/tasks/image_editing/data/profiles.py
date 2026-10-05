"""Resolved data profiles used by the Image Editing paper experiments.

Each profile names one alpha setting and the artifacts it owns.  Episode counts
are recorded as they exist in the released HDF5 files; the paper reports *pairs*
(``2 x episodes``), because every episode holds one support pair and one query
pair.

The generator derives each split seed from the profile seed:

* alpha 1.00 was generated with ``--seed 42`` -> train 42, ID test 43;
* alpha 0.66 and 0.33 were generated with ``--seed 43`` -> train 43,
  compositional OOD 143, ID test 44, and the *program split* also keyed on 43;
* the length-OOD artifacts were generated with ``--seed 42``.

The following filenames contain evaluation episodes despite the ``train`` suffix:

* ``cifar10_ood_train_alpha*.h5`` is the compositional-OOD **test** set; and
* ``cifar10_length_ood_train_L3-4_N1000.h5`` is the length-OOD **test** set.

Evaluation uses the files listed above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tasks.image_editing.data.generator import (
    LENGTH_OOD_PAIR_ATTEMPTS,
    PAIR_ATTEMPTS,
)
from tasks.image_editing.data.programs import (
    PAPER_MAX_PROGRAM_LENGTH,
    Program,
    enumerate_length_ood_programs,
    enumerate_programs,
    split_programs,
)


SplitName = Literal["train", "id_test", "comp_ood_test", "length_ood_test"]
ProgramSource = Literal["in_distribution", "compositional_ood", "length_ood"]

#: CIFAR-10 split each artifact draws its source images from.
ImageSource = Literal["train", "test"]

#: Batch-size attribute written into every released artifact.
ARTIFACT_BATCH_SIZE = 16

#: Offsets the generator applies to ``--seed`` for each artifact it writes.
TRAIN_SEED_OFFSET = 0
ID_TEST_SEED_OFFSET = 1
COMP_OOD_SEED_OFFSET = 100


@dataclass(frozen=True, slots=True)
class SplitSpec:
    """One generated artifact within a profile."""

    name: SplitName
    filename: str
    program_source: ProgramSource
    image_source: ImageSource
    episodes_per_program: int
    num_episodes: int
    shuffle: bool
    seed: int
    pair_attempts: int = PAIR_ATTEMPTS

    def __post_init__(self) -> None:
        if self.episodes_per_program < 1:
            raise ValueError("episodes_per_program must be positive")
        if self.num_episodes < 1:
            raise ValueError("num_episodes must be positive")
        if self.pair_attempts < 1:
            raise ValueError("pair_attempts must be positive")


@dataclass(frozen=True, slots=True)
class ImageEditingProfile:
    """A fully resolved, deterministic Image Editing generation profile."""

    name: str
    alpha: float | None
    min_program_length: int
    max_program_length: int
    splits: tuple[SplitSpec, ...]
    base_seed: int
    split_seed: int | None
    batch_size: int = ARTIFACT_BATCH_SIZE

    def __post_init__(self) -> None:
        if self.min_program_length < 1:
            raise ValueError("min_program_length must be positive")
        if self.max_program_length < self.min_program_length:
            raise ValueError("max_program_length must not be smaller than min_program_length")
        if self.alpha is not None and not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        names = [split.name for split in self.splits]
        if len(names) != len(set(names)):
            raise ValueError("split names must be unique")

    def get_split(self, name: SplitName) -> SplitSpec:
        for split in self.splits:
            if split.name == name:
                return split
        available = ", ".join(split.name for split in self.splits)
        raise ValueError(f"profile {self.name!r} has no split {name!r}; has: {available}")


def _alpha_profile(
    alpha: float,
    *,
    base_seed: int,
    train_filename: str,
    id_test_filename: str,
    train_episodes_per_program: int,
    id_test_episodes_per_program: int,
    id_test_shuffle: bool,
    comp_ood: tuple[str, int] | None,
) -> ImageEditingProfile:
    split = split_programs(alpha, seed=base_seed)
    num_id = split.num_in_distribution
    splits = [
        SplitSpec(
            name="train",
            filename=train_filename,
            program_source="in_distribution",
            image_source="train",
            episodes_per_program=train_episodes_per_program,
            num_episodes=num_id * train_episodes_per_program,
            shuffle=True,
            seed=base_seed + TRAIN_SEED_OFFSET,
        ),
        SplitSpec(
            name="id_test",
            filename=id_test_filename,
            program_source="in_distribution",
            image_source="test",
            episodes_per_program=id_test_episodes_per_program,
            num_episodes=num_id * id_test_episodes_per_program,
            shuffle=id_test_shuffle,
            seed=base_seed + ID_TEST_SEED_OFFSET,
        ),
    ]
    if comp_ood is not None:
        filename, episodes_per_program = comp_ood
        splits.append(
            SplitSpec(
                name="comp_ood_test",
                filename=filename,
                program_source="compositional_ood",
                image_source="train",
                episodes_per_program=episodes_per_program,
                num_episodes=split.num_compositional_ood * episodes_per_program,
                shuffle=True,
                seed=base_seed + COMP_OOD_SEED_OFFSET,
            )
        )
    return ImageEditingProfile(
        name=f"paper-alpha-{alpha:.2f}",
        alpha=alpha,
        min_program_length=1,
        max_program_length=PAPER_MAX_PROGRAM_LENGTH,
        splits=tuple(splits),
        base_seed=base_seed,
        split_seed=base_seed if alpha < 1.0 else None,
    )


#: Length-OOD generation requests 1,000 episodes per program, but resamples a
#: rejected pair only once, so roughly 85% of the request is realised.  The
#: realised count below is the one present in the released artifact.
LENGTH_OOD_TEST_EPISODES = 196_943


_PROFILES: dict[str, ImageEditingProfile] = {}

for _profile in (
    _alpha_profile(
        1.00,
        base_seed=42,
        train_filename="cifar10_transforms_train_L2_N15000_canonical.h5",
        id_test_filename="cifar10_transforms_test_L2_N3000_canonical.h5",
        train_episodes_per_program=15_000,
        id_test_episodes_per_program=3_000,
        id_test_shuffle=False,
        comp_ood=None,
    ),
    _alpha_profile(
        0.66,
        base_seed=43,
        train_filename="cifar10_iid_train_alpha066_N20000_canonical.h5",
        id_test_filename="cifar10_iid_test_alpha066_N4000_canonical.h5",
        train_episodes_per_program=20_000,
        id_test_episodes_per_program=4_000,
        id_test_shuffle=True,
        comp_ood=("cifar10_ood_train_alpha066_N20000_canonical.h5", 20_000),
    ),
    _alpha_profile(
        0.33,
        base_seed=43,
        train_filename="cifar10_iid_train_alpha033_N20000_canonical.h5",
        id_test_filename="cifar10_iid_test_alpha033_N4000_canonical.h5",
        train_episodes_per_program=20_000,
        id_test_episodes_per_program=4_000,
        id_test_shuffle=True,
        comp_ood=("cifar10_ood_train_alpha033_N20000_canonical.h5", 20_000),
    ),
):
    _PROFILES[_profile.name] = _profile


_PROFILES["paper-length-ood"] = ImageEditingProfile(
    name="paper-length-ood",
    alpha=None,
    min_program_length=3,
    max_program_length=4,
    base_seed=42,
    split_seed=None,
    splits=(
        SplitSpec(
            name="length_ood_test",
            filename="cifar10_length_ood_train_L3-4_N1000.h5",
            program_source="length_ood",
            image_source="train",
            episodes_per_program=1_000,
            num_episodes=LENGTH_OOD_TEST_EPISODES,
            shuffle=True,
            seed=42,
            pair_attempts=LENGTH_OOD_PAIR_ATTEMPTS,
        ),
    ),
)


def resolve_programs(profile: ImageEditingProfile, split: SplitSpec) -> tuple[Program, ...]:
    """Return the programs one split generates from, in generation order."""

    if split.program_source == "length_ood":
        return enumerate_length_ood_programs()
    if profile.split_seed is None:
        if split.program_source != "in_distribution":
            raise ValueError(
                f"profile {profile.name!r} has no compositional-OOD program set"
            )
        return enumerate_programs(profile.max_program_length)
    assert profile.alpha is not None
    partition = split_programs(profile.alpha, seed=profile.split_seed)
    if split.program_source == "in_distribution":
        return partition.in_distribution
    return partition.compositional_ood


def available_profiles() -> tuple[str, ...]:
    """Return stable CLI profile names."""

    return tuple(_PROFILES)


def get_profile(name: str) -> ImageEditingProfile:
    """Look up a resolved paper profile."""

    try:
        return _PROFILES[name]
    except KeyError as error:
        choices = ", ".join(available_profiles())
        raise ValueError(
            f"unknown Image Editing profile {name!r}; choose one of: {choices}"
        ) from error


__all__ = [
    "ARTIFACT_BATCH_SIZE",
    "COMP_OOD_SEED_OFFSET",
    "ID_TEST_SEED_OFFSET",
    "ImageEditingProfile",
    "ImageSource",
    "LENGTH_OOD_TEST_EPISODES",
    "ProgramSource",
    "SplitName",
    "SplitSpec",
    "TRAIN_SEED_OFFSET",
    "available_profiles",
    "get_profile",
    "resolve_programs",
]
