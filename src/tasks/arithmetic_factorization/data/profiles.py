"""Resolved paper data profiles for arithmetic factorization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SplitName = Literal["train", "test", "composition_ood", "length_ood"]

# Every alpha reuses one held-out split of longer factorizations.
LENGTH_OOD_FILENAME = "arith_x2357_even_len456_15317_test.h5"
LENGTH_OOD_DIGEST = "74b3de868ccade1e8e5be89f78847775"

# Factorizations are chains over this multiplier set with even-length balance.
PAPER_MULTIPLIERS: tuple[int, ...] = (2, 3, 5, 7)
PAPER_TRAIN_LENGTHS: tuple[int, ...] = (1, 2, 3)
PAPER_LENGTH_OOD_LENGTHS: tuple[int, ...] = (4, 5, 6)


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """One released HDF5 artifact and the digest that identifies it."""

    filename: str
    digest: str
    num_samples: int


@dataclass(frozen=True, slots=True)
class ArithmeticProfile:
    """Training, in-distribution, and out-of-distribution splits for one alpha."""

    name: str
    alpha: str
    held_out_compositions: int
    train: ArtifactSpec
    test: ArtifactSpec
    length_ood: ArtifactSpec
    composition_ood: ArtifactSpec | None = None

    def artifact(self, split: SplitName) -> ArtifactSpec:
        """Return the artifact backing one split of this profile."""

        if split == "composition_ood" and self.composition_ood is None:
            raise ValueError(
                f"profile {self.name!r} has no composition_ood split; "
                "alpha 1.00 trains on the full composition space"
            )
        artifact = getattr(self, split, None)
        if not isinstance(artifact, ArtifactSpec):
            raise ValueError(f"unknown arithmetic split: {split!r}")
        return artifact

    @property
    def splits(self) -> tuple[SplitName, ...]:
        """Splits this profile provides, in canonical evaluation order."""

        if self.composition_ood is None:
            return ("train", "test", "length_ood")
        return ("train", "test", "composition_ood", "length_ood")


_LENGTH_OOD = ArtifactSpec(LENGTH_OOD_FILENAME, LENGTH_OOD_DIGEST, 15317)

_PROFILES: tuple[ArithmeticProfile, ...] = (
    ArithmeticProfile(
        name="paper-alpha-1.00",
        alpha="1.00",
        held_out_compositions=0,
        train=ArtifactSpec(
            "arith_x2357_even_len123_279611_train.h5",
            "d4c48f5df41ae20412ce024d73144b2a",
            279611,
        ),
        test=ArtifactSpec(
            "arith_x2357_even_len123_27961_test.h5",
            "fd8216bdef9d75f0c7161bfec1ff4bf9",
            27961,
        ),
        length_ood=_LENGTH_OOD,
    ),
    ArithmeticProfile(
        name="paper-alpha-0.66",
        alpha="0.66",
        held_out_compositions=8,
        train=ArtifactSpec(
            "arith_x2357_even_len123_compo8_206520_train.h5",
            "56d7e9b05ad7064b4a455f573ac1bf08",
            206520,
        ),
        test=ArtifactSpec(
            "arith_x2357_even_len123_compo8_20651_test.h5",
            "c9b751c289f62b7cccc311203c639a21",
            20651,
        ),
        composition_ood=ArtifactSpec(
            "arith_x2357_even_len123_compo8_80401_held_out.h5",
            "cb01ac95487ba56c55a5e8b9327a7fc6",
            80401,
        ),
        length_ood=_LENGTH_OOD,
    ),
    ArithmeticProfile(
        name="paper-alpha-0.33",
        alpha="0.33",
        held_out_compositions=16,
        train=ArtifactSpec(
            "arith_x2357_even_len123_compo16_146606_train_v2.h5",
            "37f4877a66b67c590b192e1aef579b0a",
            146606,
        ),
        test=ArtifactSpec(
            "arith_x2357_even_len123_compo16_14660_test_v2.h5",
            "a5a66d77e60f5d7dffcbd4dcec8e21a5",
            14660,
        ),
        composition_ood=ArtifactSpec(
            "arith_x2357_even_len123_compo16_146306_held_out_v2.h5",
            "5b4387fd132f5b4bcb5de51c7d05ad79",
            146306,
        ),
        length_ood=_LENGTH_OOD,
    ),
)

# The observation model is pretrained on single-step level-1 transitions.
PRETRAINING_PROFILE = ArithmeticProfile(
    name="paper-observation-pretraining",
    alpha="1.00",
    held_out_compositions=0,
    train=ArtifactSpec("arith_lv1_10000_samples_train.h5", "", 10000),
    test=ArtifactSpec("arith_lv1_1000_samples_test.h5", "", 1000),
    length_ood=_LENGTH_OOD,
)


def available_profiles() -> tuple[str, ...]:
    """Return the canonical paper profile names in alpha order."""

    return tuple(profile.name for profile in _PROFILES)


def get_profile(name: str) -> ArithmeticProfile:
    """Resolve one canonical profile by name."""

    for profile in _PROFILES:
        if profile.name == name:
            return profile
    raise ValueError(f"unknown arithmetic data profile: {name!r}")


def profile_for_alpha(alpha: str) -> ArithmeticProfile:
    """Resolve the canonical profile that backs one alpha setting."""

    for profile in _PROFILES:
        if profile.alpha == alpha:
            return profile
    raise ValueError(f"unknown arithmetic alpha: {alpha!r}")


__all__ = [
    "LENGTH_OOD_DIGEST",
    "LENGTH_OOD_FILENAME",
    "PAPER_LENGTH_OOD_LENGTHS",
    "PAPER_MULTIPLIERS",
    "PAPER_TRAIN_LENGTHS",
    "PRETRAINING_PROFILE",
    "ArithmeticProfile",
    "ArtifactSpec",
    "SplitName",
    "available_profiles",
    "get_profile",
    "profile_for_alpha",
]
