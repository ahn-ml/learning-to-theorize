"""Generate and verify the Image Editing paper datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from tasks.image_editing.data.generator import generate_episodes, load_cifar10
from tasks.image_editing.data.hdf5 import read_episode, write_artifact
from tasks.image_editing.data.profiles import (
    ImageEditingProfile,
    SplitSpec,
    available_profiles,
    get_profile,
    resolve_programs,
)


DEFAULT_VERIFY_EPISODES = 64


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    programs = commands.add_parser("programs", help="inspect a resolved program split")
    programs.add_argument("--profile", choices=available_profiles(), required=True)

    generate = commands.add_parser(
        "generate", help="generate paper-compatible HDF5 artifacts"
    )
    generate.add_argument("--profile", choices=available_profiles(), required=True)
    generate.add_argument("--cifar-dir", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument(
        "--split",
        help="generate only this split; defaults to every split in the profile",
    )

    verify = commands.add_parser(
        "verify", help="regenerate a prefix of each artifact and compare it byte for byte"
    )
    verify.add_argument("--profile", choices=available_profiles(), required=True)
    verify.add_argument("--cifar-dir", type=Path, required=True)
    verify.add_argument("--data-dir", type=Path, required=True)
    verify.add_argument("--episodes", type=int, default=DEFAULT_VERIFY_EPISODES)

    verify_artifacts = commands.add_parser(
        "verify-artifacts",
        help="verify filename, episode count, and root attributes without regenerating",
    )
    verify_artifacts.add_argument("--data-dir", type=Path, required=True)
    verify_artifacts.add_argument(
        "--profiles",
        nargs="+",
        choices=available_profiles(),
        default=available_profiles(),
    )
    verify_artifacts.add_argument(
        "--sha256",
        action="store_true",
        help="also hash every file; slow on the multi-gigabyte training artifacts",
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
    episodes, report = generate_episodes(
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
        "programs_short_of_target": len(report.shortfalls),
    }


def _verify_split(
    profile: ImageEditingProfile,
    split: SplitSpec,
    cifar_dir: Path,
    data_dir: Path,
    episodes_to_check: int,
) -> dict:
    path = data_dir / split.filename
    if not path.exists():
        return {"split": split.name, "status": "missing", "path": str(path)}

    # The seeded stream advances through every episode of program k before
    # program k+1 begins, so a prefix check is only sound while it stays inside
    # the first program.  Comparing further would require regenerating all
    # ``episodes_per_program`` episodes of each earlier program; use ``generate``
    # into a scratch directory and compare hashes for a whole-file check.
    checked = min(episodes_to_check, split.episodes_per_program)

    with h5py.File(path, "r") as handle:
        stored_episodes = int(handle.attrs["dataset_length"])
        stored_batch_size = int(handle.attrs["batch_size"])
        stored_shuffle = bool(handle.attrs["shuffle"])
        checked = min(checked, stored_episodes)
        reference = [read_episode(handle, index) for index in range(checked)]

    images = load_cifar10(cifar_dir, split.image_source)
    programs = resolve_programs(profile, split)
    regenerated, _ = generate_episodes(
        images,
        programs[:1],
        split.episodes_per_program,
        seed=split.seed,
        pair_attempts=split.pair_attempts,
        stop_after=checked,
    )

    comparable = min(checked, len(regenerated))
    identical = comparable == checked and all(
        np.array_equal(reference[i].grids, regenerated[i].grids)
        and reference[i].program == regenerated[i].program
        for i in range(comparable)
    )
    metadata_ok = (
        stored_episodes == split.num_episodes
        and stored_batch_size == profile.batch_size
        and stored_shuffle == split.shuffle
    )
    return {
        "split": split.name,
        "status": "ok" if identical and metadata_ok else "mismatch",
        "episodes_compared": comparable,
        "byte_identical_prefix": bool(identical),
        "stored_episodes": stored_episodes,
        "expected_episodes": split.num_episodes,
        "episode_count_matches": stored_episodes == split.num_episodes,
        "batch_size_matches": stored_batch_size == profile.batch_size,
        "shuffle_matches": stored_shuffle == split.shuffle,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "programs":
        profile = get_profile(args.profile)
        payload = {
            "profile": profile.name,
            "alpha": profile.alpha,
            "base_seed": profile.base_seed,
            "split_seed": profile.split_seed,
            "program_lengths": [profile.min_program_length, profile.max_program_length],
            "splits": [
                {
                    "split": split.name,
                    "filename": split.filename,
                    "program_source": split.program_source,
                    "image_source": split.image_source,
                    "seed": split.seed,
                    "num_programs": len(resolve_programs(profile, split)),
                    "episodes_per_program": split.episodes_per_program,
                    "episodes": split.num_episodes,
                    "pairs": split.num_pairs,
                }
                for split in profile.splits
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "generate":
        profile = get_profile(args.profile)
        splits = (
            [profile.get_split(args.split)] if args.split else list(profile.splits)
        )
        results = [
            _generate_split(profile, split, args.cifar_dir, args.output_dir)
            for split in splits
        ]
        print(json.dumps({"profile": profile.name, "results": results}, indent=2))
        return 0 if all(item["matches_released_count"] for item in results) else 1

    if args.command == "verify":
        profile = get_profile(args.profile)
        results = [
            _verify_split(profile, split, args.cifar_dir, args.data_dir, args.episodes)
            for split in profile.splits
        ]
        print(json.dumps({"profile": profile.name, "results": results}, indent=2))
        return 0 if all(item.get("status") == "ok" for item in results) else 1

    if args.command == "verify-artifacts":
        results = []
        ok = True
        for name in args.profiles:
            profile = get_profile(name)
            for split in profile.splits:
                path = args.data_dir / split.filename
                entry: dict = {"profile": name, "split": split.name, "filename": split.filename}
                if not path.exists():
                    entry["status"] = "missing"
                    ok = False
                else:
                    with h5py.File(path, "r") as handle:
                        stored = int(handle.attrs["dataset_length"])
                        entry["episodes"] = stored
                        entry["expected_episodes"] = split.num_episodes
                        entry["batch_size_matches"] = (
                            int(handle.attrs["batch_size"]) == profile.batch_size
                        )
                        entry["shuffle_matches"] = (
                            bool(handle.attrs["shuffle"]) == split.shuffle
                        )
                    entry["bytes"] = path.stat().st_size
                    if args.sha256:
                        entry["sha256"] = _sha256(path)
                    entry["status"] = (
                        "ok"
                        if stored == split.num_episodes
                        and entry["batch_size_matches"]
                        and entry["shuffle_matches"]
                        else "mismatch"
                    )
                    ok = ok and entry["status"] == "ok"
                results.append(entry)
        print(json.dumps({"results": results}, indent=2))
        return 0 if ok else 1

    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
