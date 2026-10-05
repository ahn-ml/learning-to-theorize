#!/usr/bin/env python3
"""Pretrain the task observation model with the release recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    PRETRAINING_PROCESSES,
    TASKS,
    parse_devices,
    add_tracking_arguments,
    module_command,
    run_commands,
    gridworld_artifact,
    tracking_argv,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--devices", type=parse_devices, required=True)
    parser.add_argument("--dry-run", action="store_true")
    add_tracking_arguments(parser)
    arguments = parser.parse_args()
    if arguments.task == "image_editing":
        parser.error("ImageEditing uses the provided observation checkpoint; see docs/reproduction.md")
    if len(arguments.devices.split(",")) != PRETRAINING_PROCESSES:
        parser.error(f"observation pretraining requires {PRETRAINING_PROCESSES} devices")
    root = arguments.data_root.expanduser().resolve()
    if arguments.task == "gridworld":
        train = gridworld_artifact(root, "vae-pretraining", "practice")
        test = gridworld_artifact(root, "vae-pretraining", "exam")
    else:
        from tasks.arithmetic_factorization.data.profiles import PRETRAINING_PROFILE
        train, test = root / PRETRAINING_PROFILE.train.filename, root / PRETRAINING_PROFILE.test.filename
    command = module_command(f"tasks.{arguments.task}.observation_runner",
                             "--train-h5", train, "--test-h5", test,
                             "--output-root", arguments.output_root.expanduser().resolve(),
                             *tracking_argv(arguments), processes=PRETRAINING_PROCESSES)
    return run_commands([command], devices=arguments.devices, dry_run=arguments.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
