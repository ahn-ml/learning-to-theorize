"""Resolved paper data profiles for arithmetic factorization."""

from __future__ import annotations

from dataclasses import dataclass

# Every alpha reuses one held-out split of longer factorizations.
LENGTH_OOD_FILENAME = "arith_x2357_even_len456_15317_test.h5"

# Factorizations are chains over this multiplier set with even-length balance.
PAPER_MULTIPLIERS: tuple[int, ...] = (2, 3, 5, 7)
PAPER_TRAIN_LENGTHS: tuple[int, ...] = (1, 2, 3)
PAPER_LENGTH_OOD_LENGTHS: tuple[int, ...] = (4, 5, 6)


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """One HDF5 artifact and its episode count."""

    filename: str
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


_LENGTH_OOD = ArtifactSpec(LENGTH_OOD_FILENAME, 15317)

_PROFILES: tuple[ArithmeticProfile, ...] = (
    ArithmeticProfile(
        name="paper-alpha-1.00",
        alpha="1.00",
        held_out_compositions=0,
        train=ArtifactSpec("arith_x2357_even_len123_279611_train.h5", 279611),
        test=ArtifactSpec("arith_x2357_even_len123_27961_test.h5", 27961),
        length_ood=_LENGTH_OOD,
    ),
    ArithmeticProfile(
        name="paper-alpha-0.66",
        alpha="0.66",
        held_out_compositions=8,
        train=ArtifactSpec("arith_x2357_even_len123_compo8_206520_train.h5", 206520),
        test=ArtifactSpec("arith_x2357_even_len123_compo8_20651_test.h5", 20651),
        composition_ood=ArtifactSpec("arith_x2357_even_len123_compo8_80401_held_out.h5", 80401),
        length_ood=_LENGTH_OOD,
    ),
    ArithmeticProfile(
        name="paper-alpha-0.33",
        alpha="0.33",
        held_out_compositions=16,
        train=ArtifactSpec("arith_x2357_even_len123_compo16_146606_train_v2.h5", 146606),
        test=ArtifactSpec("arith_x2357_even_len123_compo16_14660_test_v2.h5", 14660),
        composition_ood=ArtifactSpec("arith_x2357_even_len123_compo16_146306_held_out_v2.h5", 146306),
        length_ood=_LENGTH_OOD,
    ),
)

# The observation model is pretrained on single-step level-1 transitions.
PRETRAINING_PROFILE = ArithmeticProfile(
    name="paper-observation-pretraining",
    alpha="1.00",
    held_out_compositions=0,
    train=ArtifactSpec("arith_lv1_10000_samples_train.h5", 10000),
    test=ArtifactSpec("arith_lv1_1000_samples_test.h5", 1000),
    length_ood=_LENGTH_OOD,
)


def available_profiles() -> tuple[str, ...]:
    """Return the paper profile names in alpha order."""

    return tuple(profile.name for profile in _PROFILES)


def get_profile(name: str) -> ArithmeticProfile:
    """Resolve one profile by name."""

    for profile in _PROFILES:
        if profile.name == name:
            return profile
    raise ValueError(f"unknown arithmetic data profile: {name!r}")


def profile_for_alpha(alpha: str) -> ArithmeticProfile:
    """Resolve the profile that backs one alpha setting."""

    for profile in _PROFILES:
        if profile.alpha == alpha:
            return profile
    raise ValueError(f"unknown arithmetic alpha: {alpha!r}")


__all__ = [
    "LENGTH_OOD_FILENAME",
    "PAPER_LENGTH_OOD_LENGTHS",
    "PAPER_MULTIPLIERS",
    "PAPER_TRAIN_LENGTHS",
    "PRETRAINING_PROFILE",
    "ArithmeticProfile",
    "ArtifactSpec",
    "available_profiles",
    "get_profile",
    "profile_for_alpha",
]
