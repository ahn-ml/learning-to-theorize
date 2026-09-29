"""Resolved data profiles used by the GridWorld paper experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SplitName = Literal["practice", "exam", "ood_test"]
SplitMode = Literal["alpha", "all"]


@dataclass(frozen=True, slots=True)
class SplitSpec:
    """One sequentially generated artifact within a profile."""

    name: SplitName
    num_episodes: int
    batch_size: int
    use_ood_programs: bool = False

    def __post_init__(self) -> None:
        if self.num_episodes < 0:
            raise ValueError("num_episodes must be non-negative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True, slots=True)
class GridWorldProfile:
    """A fully resolved, deterministic GridWorld generation profile."""

    name: str
    output_prefix: str
    min_program_length: int
    max_program_length: int
    split_mode: SplitMode
    alpha: float | None
    splits: tuple[SplitSpec, ...]
    seed: int = 1127
    split_seed: int = 1127
    num_supports: int = 1
    grid_size: int = 10
    num_colors: int = 8
    num_objects: int = 1

    def __post_init__(self) -> None:
        if self.min_program_length < 1:
            raise ValueError("min_program_length must be positive")
        if self.max_program_length < self.min_program_length:
            raise ValueError("max_program_length must not be smaller than min_program_length")
        if self.split_mode == "alpha" and (self.alpha is None or not 0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha profiles require alpha in [0, 1]")
        if self.split_mode == "all" and self.alpha is not None:
            raise ValueError("all-program profiles must not set alpha")
        if self.num_supports < 1:
            raise ValueError("num_supports must be positive")
        if self.grid_size < 1 or self.num_colors < 1 or self.num_objects < 1:
            raise ValueError("grid_size, num_colors, and num_objects must be positive")
        names = [split.name for split in self.splits]
        if len(names) != len(set(names)):
            raise ValueError("split names must be unique")

    def artifact_filename(self, split: SplitSpec, extension: str = "h5") -> str:
        """Return the dataset filename for a split."""

        num_pairs = split.num_episodes * (self.num_supports + 1)
        return f"{self.output_prefix}_n{num_pairs}_{split.name}.{extension}"


def _alpha_profile(alpha: float) -> GridWorldProfile:
    alpha_label = f"{alpha:.2f}"
    ood_count = 0 if alpha == 1.0 else 5_000
    splits: list[SplitSpec] = [
        SplitSpec("practice", 50_000, batch_size=32),
        SplitSpec("exam", 5_000, batch_size=1),
    ]
    if ood_count:
        splits.append(SplitSpec("ood_test", ood_count, batch_size=1, use_ood_programs=True))
    return GridWorldProfile(
        name=f"paper-alpha-{alpha_label}",
        output_prefix=(
            "GridWorld_0107_lvl1-3_8c_10x10_1sup_4prim_sp_"
            f"alpha{alpha_label}_seed1127"
        ),
        min_program_length=1,
        max_program_length=3,
        split_mode="alpha",
        alpha=alpha,
        splits=tuple(splits),
    )


_PROFILES: dict[str, GridWorldProfile] = {
    profile.name: profile
    for profile in (_alpha_profile(0.33), _alpha_profile(0.66), _alpha_profile(1.0))
}
_PROFILES["paper-length-4-8"] = GridWorldProfile(
    name="paper-length-4-8",
    output_prefix=(
        "GridWorld_0107_lvl4-8_8c_10x10_1sup_4prim_sp_"
        "alpha1.00_seed1127"
    ),
    min_program_length=4,
    max_program_length=8,
    split_mode="all",
    alpha=None,
    splits=(
        SplitSpec("practice", 100, batch_size=32),
        SplitSpec("exam", 10_000, batch_size=1),
    ),
)
_PROFILES["vae-pretraining"] = GridWorldProfile(
    name="vae-pretraining",
    output_prefix="New_lvl1-3_8c_10x10_1sup_4prim_sp",
    min_program_length=1,
    max_program_length=3,
    split_mode="all",
    alpha=None,
    splits=(
        SplitSpec("practice", 500_000, batch_size=32),
        SplitSpec("exam", 5_000, batch_size=1),
    ),
)


def available_profiles() -> tuple[str, ...]:
    """Return stable CLI profile names."""

    return tuple(_PROFILES)


def get_profile(name: str) -> GridWorldProfile:
    """Look up a resolved paper profile."""

    try:
        return _PROFILES[name]
    except KeyError as error:
        choices = ", ".join(available_profiles())
        raise ValueError(f"unknown GridWorld profile {name!r}; choose one of: {choices}") from error
