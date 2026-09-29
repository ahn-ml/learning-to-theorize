#!/usr/bin/env python3
"""Train NEO for selected alpha values and seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    TASKS,
    parse_devices,
    add_tracking_arguments,
    load_task_config,
    module_command,
    run_commands,
    training_data,
    tracking_argv,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--config", type=Path, help="default: configs/<task>/reproduction.yaml")
    parser.add_argument("--method", choices=("neo",), required=True)
    parser.add_argument("--alpha", choices=("0.33", "0.66", "1.00", "all"), required=True)
    parser.add_argument("--seed", choices=("42", "43", "44", "all"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--observation-checkpoint", type=Path, required=True)
    parser.add_argument("--observation-selection", choices=("recorded-final", "best-reconstruction"),
                        help="GridWorld observation handoff policy; defaults to the selected recipe")
    parser.add_argument("--devices", type=parse_devices, required=True)
    parser.add_argument("--dry-run", action="store_true")
    add_tracking_arguments(parser)
    arguments = parser.parse_args()
    if arguments.observation_selection is not None and arguments.task != "gridworld":
        parser.error("--observation-selection applies only to GridWorld")
    config = load_task_config(arguments.config, arguments.task)
    expected = config["training"]["devices"]
    if len(arguments.devices.split(",")) != expected:
        parser.error(f"{arguments.task} reproduction training requires {expected} devices")
    if arguments.observation_selection is None:
        arguments.observation_selection = config["training"].get("observation_selection")
    alphas = ("1.00", "0.66", "0.33") if arguments.task == "image_editing" else ("0.33", "0.66", "1.00")
    alphas = alphas if arguments.alpha == "all" else (arguments.alpha,)
    seeds = ("42", "43", "44") if arguments.seed == "all" else (arguments.seed,)
    conditions = ([(alpha, seed) for seed in seeds for alpha in alphas]
                  if arguments.task == "gridworld" else
                  [(alpha, seed) for alpha in alphas for seed in seeds])
    root = arguments.data_root.expanduser().resolve()
    commands = []
    for alpha, seed in conditions:
        train, test = training_data(arguments.task, alpha, root)
        options = ["--method", arguments.method, "--seed", seed,
                   "--train-h5", train, "--test-h5", test,
                   "--observation-checkpoint", arguments.observation_checkpoint.expanduser().resolve(),
                   "--output-root", arguments.output_root.expanduser().resolve(),
                   *tracking_argv(arguments)]
        if arguments.task == "gridworld":
            options += ["--experiment", f"alpha-{alpha}"]
            if arguments.observation_selection != "recorded-final":
                options += ["--observation-selection", arguments.observation_selection]
        else:
            options += ["--alpha", alpha]
        commands.append(module_command(f"tasks.{arguments.task}.theorizer_runner", *options,
                                       processes=expected))
    return run_commands(commands, devices=arguments.devices, dry_run=arguments.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
