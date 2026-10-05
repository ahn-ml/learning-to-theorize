"""Generate the Image Editing paper datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from tasks.image_editing.data.generator import generate_episodes, load_cifar10
from tasks.image_editing.data.hdf5 import write_artifact
from tasks.image_editing.data.profiles import (
    ImageEditingProfile,
    SplitSpec,
    available_profiles,
    get_profile,
    resolve_programs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="generate paper-compatible HDF5 artifacts")
    generate.add_argument("--profile", choices=available_profiles(), required=True)
    generate.add_argument("--cifar-dir", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument(
        "--split",
        help="generate only this split; defaults to every split in the profile",
    )
    return parser


def _generate_split(
    profile: ImageEditingProfile,
    split: SplitSpec,
    cifar_dir: Path,
    output_dir: Path,
) -> dict:
    images = load_cifar10(cifar_dir, split.image_source)
    programs = resolve_programs(profile, split)
    episodes, shortfalls = generate_episodes(
        images,
        programs,
        split.episodes_per_program,
        seed=split.seed,
        pair_attempts=split.pair_attempts,
    )
    destination = output_dir / split.filename
    write_artifact(
        destination,
        episodes,
        batch_size=profile.batch_size,
        shuffle=split.shuffle,
    )
    return {
        "split": split.name,
        "output": str(destination),
        "seed": split.seed,
        "num_programs": len(programs),
        "episodes": len(episodes),
        "expected_episodes": split.num_episodes,
        "matches_released_count": len(episodes) == split.num_episodes,
        "programs_short_of_target": len(shortfalls),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    profile = get_profile(args.profile)
    splits = [profile.get_split(args.split)] if args.split else list(profile.splits)
    results = [
        _generate_split(profile, split, args.cifar_dir, args.output_dir)
        for split in splits
    ]
    print(json.dumps({"profile": profile.name, "results": results}, indent=2))
    return 0 if all(item["matches_released_count"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
