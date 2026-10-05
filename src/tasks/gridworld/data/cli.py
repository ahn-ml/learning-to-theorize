"""Generate the GridWorld datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from tasks.gridworld.data.artifacts import (
    GridWorldArtifactEvidence,
    verify_profile_artifacts,
)
from tasks.gridworld.data.profiles import available_profiles, get_profile
from tasks.gridworld.data.hdf5 import write_profile
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
    generate_all = commands.add_parser(
        "generate-all",
        help="generate the shape pool and every GridWorld dataset",
    )
    generate_all.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_directory = args.output_dir.expanduser().resolve()
    shape_pool_path = output_directory / "shape_pool.pkl"
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

    profiles = available_profiles()
    artifacts: list[Path] = []
    evidence: list[GridWorldArtifactEvidence] = []
    for profile_name in profiles:
        profile = get_profile(profile_name)
        artifacts.extend(
            write_profile(profile, pool, output_directory)
        )
        evidence.extend(verify_profile_artifacts(profile, output_directory))
    print(
        json.dumps(
            {
                "shape_pool": str(shape_pool_path),
                "profiles": list(profiles),
                "artifacts": [str(path) for path in artifacts],
                "artifact_evidence": [item.as_dict() for item in evidence],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
