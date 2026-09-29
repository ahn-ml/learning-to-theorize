"""Generate and verify the GridWorld paper datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from tasks.gridworld.data.artifacts import (
    GridWorldArtifactEvidence,
    verify_artifact_directory,
    verify_profile_artifacts,
)
from tasks.gridworld.data.profiles import available_profiles, get_profile
from tasks.gridworld.data.hdf5 import verify_profile, write_profile
from tasks.gridworld.data.programs import alpha_split, canonical_programs
from tasks.gridworld.data.shape_pool import (
    CANONICAL_SHAPE_POOL_SHA256,
    generate_canonical_shape_pool,
    shape_pool_pickle_bytes,
    load_shape_pool,
    save_shape_pool,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    shape_pool = commands.add_parser("shape-pool", help="recreate the canonical shape pool")
    shape_pool.add_argument("--output", type=Path, required=True)

    programs = commands.add_parser("programs", help="inspect a resolved program split")
    programs.add_argument("--profile", choices=available_profiles(), required=True)

    generate = commands.add_parser("generate", help="generate paper-compatible HDF5 files")
    generate.add_argument("--profile", choices=available_profiles(), required=True)
    generate.add_argument("--shape-pool", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)

    verify = commands.add_parser("verify", help="regenerate and compare existing paper HDF5 files")
    verify.add_argument("--profile", choices=available_profiles(), required=True)
    verify.add_argument("--shape-pool", type=Path, required=True)
    verify.add_argument("--data-dir", type=Path, required=True)
    verify.add_argument("--max-total-episodes", type=int)

    generate_all = commands.add_parser(
        "generate-all",
        help="generate the shape pool and every canonical GridWorld profile",
    )
    generate_all.add_argument("--output-dir", type=Path, required=True)
    generate_all.add_argument("--shape-pool", type=Path)
    generate_all.add_argument(
        "--profiles",
        nargs="+",
        choices=available_profiles(),
        default=available_profiles(),
        help="profiles to generate; defaults to every canonical profile",
    )
    verify_artifacts = commands.add_parser(
        "verify-artifacts",
        help="verify exact filename, byte size, and SHA-256 for paper artifacts",
    )
    verify_artifacts.add_argument("--data-dir", type=Path, required=True)
    verify_artifacts.add_argument(
        "--profiles",
        nargs="+",
        choices=available_profiles(),
        default=available_profiles(),
        help="profiles to verify; defaults to every canonical profile",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "shape-pool":
        pool = generate_canonical_shape_pool()
        digest = hashlib.sha256(shape_pool_pickle_bytes(pool)).hexdigest()
        if digest != CANONICAL_SHAPE_POOL_SHA256:
            raise RuntimeError(f"generated shape pool has unexpected SHA-256: {digest}")
        save_shape_pool(args.output, pool)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "sha256": digest,
                    "num_shapes": len(pool.shapes),
                    "total_attempts": pool.total_attempts,
                    "filtered_candidates": pool.filtered_candidates,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "generate-all":
        output_directory = args.output_dir.expanduser().resolve()
        shape_pool_path = (
            args.shape_pool.expanduser().resolve()
            if args.shape_pool is not None
            else output_directory / "shape_pool.pkl"
        )
        if shape_pool_path.exists():
            pool = load_shape_pool(
                shape_pool_path,
                expected_sha256=CANONICAL_SHAPE_POOL_SHA256,
            )
        else:
            pool = generate_canonical_shape_pool()
            digest = hashlib.sha256(shape_pool_pickle_bytes(pool)).hexdigest()
            if digest != CANONICAL_SHAPE_POOL_SHA256:
                raise RuntimeError(
                    f"generated shape pool has unexpected SHA-256: {digest}"
                )
            save_shape_pool(shape_pool_path, pool)

        artifacts: list[Path] = []
        evidence: list[GridWorldArtifactEvidence] = []
        for profile_name in args.profiles:
            profile = get_profile(profile_name)
            artifacts.extend(
                write_profile(profile, pool, output_directory)
            )
            evidence.extend(verify_profile_artifacts(profile, output_directory))
        print(
            json.dumps(
                {
                    "shape_pool": str(shape_pool_path),
                    "profiles": list(args.profiles),
                    "artifacts": [str(path) for path in artifacts],
                    "artifact_evidence": [item.as_dict() for item in evidence],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "verify-artifacts":
        evidence = verify_artifact_directory(
            args.data_dir,
            profiles=args.profiles,
        )
        print(
            json.dumps(
                {
                    "data_root": str(args.data_dir.expanduser().resolve()),
                    "profiles": list(args.profiles),
                    "exact_paper_artifacts": True,
                    "artifact_evidence": [item.as_dict() for item in evidence],
                },
                indent=2,
            )
        )
        return 0

    profile = get_profile(args.profile)
    if args.command == "programs":
        all_programs = canonical_programs(profile.min_program_length, profile.max_program_length)
        if profile.split_mode == "alpha":
            assert profile.alpha is not None
            train, ood = alpha_split(all_programs, profile.alpha, profile.split_seed)
        else:
            train, ood = all_programs, ()
        print(
            json.dumps(
                {
                    "profile": profile.name,
                    "all": [list(program) for program in all_programs],
                    "train": [list(program) for program in train],
                    "ood": [list(program) for program in ood],
                },
                indent=2,
            )
        )
        return 0

    shape_pool = load_shape_pool(
        args.shape_pool,
        expected_sha256=CANONICAL_SHAPE_POOL_SHA256,
    )
    if args.command == "generate":
        paths = write_profile(profile, shape_pool, args.output_dir)
        evidence = verify_profile_artifacts(profile, args.output_dir)
        print(
            json.dumps(
                {
                    "profile": profile.name,
                    "artifacts": [str(path) for path in paths],
                    "artifact_evidence": [item.as_dict() for item in evidence],
                },
                indent=2,
            )
        )
        return 0
    if args.command == "verify":
        result = verify_profile(
            profile,
            shape_pool,
            args.data_dir,
            max_total_episodes=args.max_total_episodes,
        )
        print(
            json.dumps(
                {
                    "profile": result.profile,
                    "compared_by_split": result.compared_by_split,
                    "complete": result.complete,
                },
                indent=2,
            )
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
