"""Generate and verify the released arithmetic factorization artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

from tasks.arithmetic_factorization.data.profiles import (
    PAPER_LENGTH_OOD_LENGTHS,
    PAPER_MULTIPLIERS,
    PAPER_TRAIN_LENGTHS,
    ArithmeticProfile,
    available_profiles,
    get_profile,
)

# The released artifacts were produced with these exact generator settings.
PAPER_MAX_DIGITS = 6
PAPER_MAX_INPUT_VALUE = 999999
PAPER_SAMPLES_PER_PROGRAM = 10000
PAPER_LENGTH_OOD_SAMPLES_PER_PROGRAM = 100
PAPER_TEST_RATIO = 1 / 11
PAPER_SEED = 42

# The alpha splits were drawn by a generator revision that no longer exists,
# so the held-out program index tuples are pinned from the artifact metadata.
PAPER_HELD_OUT_COMBINATIONS: dict[str, tuple[tuple[int, ...], ...]] = {
    "1.00": (),
    "0.66": (
        (1,),
        (1, 2),
        (0, 2),
        (1, 3, 3),
        (0, 0, 3),
        (1, 1, 2),
        (0, 1, 3),
        (0, 0, 2),
    ),
    "0.33": (
        (2,),
        (1,),
        (0, 3),
        (1, 2),
        (0, 2),
        (1, 3),
        (0, 3, 3),
        (0, 1, 3),
        (0, 1, 1),
        (1, 3, 3),
        (0, 0, 2),
        (0, 0, 3),
        (0, 2, 2),
        (1, 2, 3),
        (2, 3, 3),
        (1, 1, 2),
    ),
}


def _primitives() -> list[str]:
    return [f"*{value}" for value in PAPER_MULTIPLIERS]


def _digest(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - artifact identity, not security
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generate_profile(profile: ArithmeticProfile, output_dir: Path) -> None:
    from tasks.arithmetic_factorization.data.generator import (
        generate_evenly_distributed_dataset,
        save_dataset_to_h5,
    )

    held_out = PAPER_HELD_OUT_COMBINATIONS[profile.alpha]
    (train, train_programs), (test, test_programs), (kept_out, kept_out_programs) = (
        generate_evenly_distributed_dataset(
            primitives_list=_primitives(),
            allowed_lengths=list(PAPER_TRAIN_LENGTHS),
            samples_per_program=PAPER_SAMPLES_PER_PROGRAM,
            max_input_value=PAPER_MAX_INPUT_VALUE,
            max_digit_length=PAPER_MAX_DIGITS,
            test_ratio=PAPER_TEST_RATIO,
            held_out_combinations=list(held_out) or None,
            seed=PAPER_SEED,
        )
    )
    metadata = {
        "primitives": _primitives(),
        "allowed_lengths": list(PAPER_TRAIN_LENGTHS),
        "samples_per_program": PAPER_SAMPLES_PER_PROGRAM,
        "held_out_combinations": list(held_out) or None,
    }
    save_dataset_to_h5(
        train, train_programs, str(output_dir / profile.train.filename),
        PAPER_MAX_DIGITS, metadata,
    )
    save_dataset_to_h5(
        test, test_programs, str(output_dir / profile.test.filename),
        PAPER_MAX_DIGITS, metadata,
    )
    if profile.composition_ood is not None:
        save_dataset_to_h5(
            kept_out, kept_out_programs,
            str(output_dir / profile.composition_ood.filename),
            PAPER_MAX_DIGITS, metadata,
        )


def _generate_length_ood(output_dir: Path) -> None:
    from tasks.arithmetic_factorization.data.generator import (
        generate_evenly_distributed_dataset,
        save_dataset_to_h5,
    )
    from tasks.arithmetic_factorization.data.profiles import LENGTH_OOD_FILENAME

    (_, _), (test, test_programs), (_, _) = generate_evenly_distributed_dataset(
        primitives_list=_primitives(),
        allowed_lengths=list(PAPER_LENGTH_OOD_LENGTHS),
        samples_per_program=PAPER_LENGTH_OOD_SAMPLES_PER_PROGRAM,
        max_input_value=PAPER_MAX_INPUT_VALUE,
        max_digit_length=PAPER_MAX_DIGITS,
        test_ratio=PAPER_TEST_RATIO,
        held_out_combinations=None,
        seed=PAPER_SEED,
    )
    save_dataset_to_h5(
        test, test_programs, str(output_dir / LENGTH_OOD_FILENAME),
        PAPER_MAX_DIGITS,
        {
            "primitives": _primitives(),
            "allowed_lengths": list(PAPER_LENGTH_OOD_LENGTHS),
            "samples_per_program": PAPER_LENGTH_OOD_SAMPLES_PER_PROGRAM,
            "held_out_combinations": None,
        },
    )


def generate_observation_data(output_dir: Path) -> int:
    from tasks.arithmetic_factorization.data.observations import OBSERVATION_FILENAME, generate_observations

    path = output_dir / OBSERVATION_FILENAME
    if path.exists():
        print(f"skipping observations: {path.name} already exists")
    else:
        generate_observations(path)
    return 0


def generate_all(output_dir: Path) -> int:
    """Generate every canonical profile, refusing to overwrite existing files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    generate_observation_data(output_dir)
    for name in available_profiles():
        profile = get_profile(name)
        if (output_dir / profile.train.filename).exists():
            print(f"skipping {name}: {profile.train.filename} already exists")
            continue
        print(f"generating {name}")
        _generate_profile(profile, output_dir)
    from tasks.arithmetic_factorization.data.profiles import LENGTH_OOD_FILENAME

    if (output_dir / LENGTH_OOD_FILENAME).exists():
        print(f"skipping length OOD: {LENGTH_OOD_FILENAME} already exists")
    else:
        print("generating length OOD split")
        _generate_length_ood(output_dir)
    return 0


def verify(output_dir: Path) -> int:
    """Compare every released artifact against its recorded digest."""

    failures = 0
    seen: set[str] = set()
    for name in available_profiles():
        profile = get_profile(name)
        for split in profile.splits:
            artifact = profile.artifact(split)
            if artifact.filename in seen or not artifact.digest:
                continue
            seen.add(artifact.filename)
            path = output_dir / artifact.filename
            if not path.exists():
                print(f"MISSING  {artifact.filename}")
                failures += 1
                continue
            observed = _digest(path)
            status = "OK      " if observed == artifact.digest else "MISMATCH"
            if observed != artifact.digest:
                failures += 1
            print(f"{status} {artifact.filename}")
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify arithmetic factorization artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate-all", "generate-observations", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    output_dir = arguments.output_dir.expanduser().resolve()
    if arguments.command == "generate-all":
        return generate_all(output_dir)
    if arguments.command == "generate-observations":
        return generate_observation_data(output_dir)
    return verify(output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
