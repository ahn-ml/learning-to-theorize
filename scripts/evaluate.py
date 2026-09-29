#!/usr/bin/env python3
"""Evaluate NEO or NEO-S with the canonical task protocol."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _common import (
    TASKS,
    parse_devices,
    add_tracking_arguments,
    load_task_config,
    tracking_argv,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--config", type=Path, help="default: configs/<task>/reproduction.yaml")
    parser.add_argument("--method", choices=("neo",), required=True)
    parser.add_argument(
        "--protocol",
        choices=("standard", "test-time-scaling"),
        default="standard",
    )
    parser.add_argument("--seed", choices=("42", "43", "44", "all"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--alpha", choices=("0.33", "0.66", "1.00", "all"), default="all")
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--selection-record", type=Path)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--devices", type=parse_devices, required=True)
    parser.add_argument("--dry-run", action="store_true")
    add_tracking_arguments(parser)
    arguments = parser.parse_args()
    load_task_config(arguments.config, arguments.task)
    if len(arguments.devices.split(",")) != 1:
        parser.error("evaluation requires exactly one device")
    os.environ["CUDA_VISIBLE_DEVICES"] = arguments.devices
    from tasks.neo_evaluation import main as evaluate_neo
    argv = ["--task", arguments.task, "--alpha", arguments.alpha,
            "--seed", arguments.seed, "--protocol", arguments.protocol,
            "--data-root", str(arguments.data_root),
            "--output-root", str(arguments.output_root), *tracking_argv(arguments)]
    for option, value in (("--training-root", arguments.training_root),
                          ("--checkpoint-directory", arguments.checkpoint_directory),
                          ("--selection-record", arguments.selection_record)):
        if value is not None:
            argv.extend((option, str(value)))
    if arguments.selection_only:
        argv.append("--selection-only")
    if arguments.dry_run:
        argv.append("--dry-run")
    return evaluate_neo(argv)


if __name__ == "__main__":
    raise SystemExit(main())
