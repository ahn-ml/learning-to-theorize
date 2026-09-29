"""Shared argument handling and subprocess execution."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

TASKS = ("gridworld", "arithmetic_factorization", "image_editing")


def parse_devices(value: str) -> str:
    """Normalize a comma-separated list of distinct GPU indices."""
    parts = [part.strip() for part in value.split(",")]
    if not all(part.isdecimal() for part in parts):
        raise argparse.ArgumentTypeError("--devices requires comma-separated GPU indices, e.g. 0,1")
    devices = [str(int(part)) for part in parts]
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("--devices requires distinct GPU indices")
    return ",".join(devices)


def load_task_config(path: Path | None, task: str) -> Mapping[str, Any]:
    """Load the task execution recipe and resolve its paths."""

    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    path = path or Path(__file__).resolve().parents[1] / "configs" / task / "reproduction.yaml"
    path = path.expanduser().resolve(strict=True)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, Mapping):
        raise ValueError(f"{path} must contain a YAML mapping")
    configured_task = config.get("task")
    if configured_task != task:
        raise ValueError(
            f"config task is {configured_task!r}, but --task is {task!r}"
        )
    if config.get("schema_version") != 2:
        raise ValueError("use configs/<task>/reproduction.yaml (schema_version: 2)")
    if set(config) != {"schema_version", "task", "pretraining", "training"}:
        raise ValueError("reproduction config requires task, pretraining and training")
    pretraining, training = config["pretraining"], config["training"]
    if not isinstance(pretraining, Mapping) or not isinstance(training, Mapping):
        raise ValueError("pretraining and training must be mappings")
    allowed = {"profile", "overrides"} if task == "gridworld" else {"profile"}
    if set(pretraining) - allowed or pretraining.get("profile") not in (
        ("pretrained",) if task == "image_editing" else ("recovered", "appendix")
    ):
        raise ValueError("unsupported pretraining profile/options")
    allowed_training = {"preset", "devices"} | ({"observation_selection"} if task == "gridworld" else set())
    if set(training) != allowed_training or training["preset"] != "paper":
        raise ValueError("training uses the fixed paper preset; unsupported training options")
    if training["devices"] != (2 if task == "image_editing" else 1):
        raise ValueError("training device count must match the measured topology")
    if task == "gridworld" and training["observation_selection"] not in ("recorded-final", "best-reconstruction"):
        raise ValueError("unknown observation selection policy")
    config = dict(config)
    config["pretraining"] = dict(pretraining)
    if "overrides" in pretraining:
        config["pretraining"]["overrides"] = str((path.resolve().parent / pretraining["overrides"]).resolve(strict=True))
    return config


def add_tracking_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wandb-project", default="LearningToTheorize")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline"),
        default="online",
    )


def tracking_argv(arguments: argparse.Namespace) -> list[str]:
    result = [
        "--wandb-project",
        arguments.wandb_project,
        "--wandb-mode",
        arguments.wandb_mode,
    ]
    if arguments.wandb_entity:
        result.extend(("--wandb-entity", arguments.wandb_entity))
    if arguments.wandb_group:
        result.extend(("--wandb-group", arguments.wandb_group))
    return result


def module_command(module: str, *arguments: str, processes: int = 1) -> list[str]:
    command = [sys.executable]
    if processes > 1:
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={processes}"]
    return [*command, "-m", module, *map(str, arguments)]


def run_commands(commands: Sequence[Sequence[str]], *, devices: str | None = None,
                 dry_run: bool = False) -> int:
    environment = {"CUDA_VISIBLE_DEVICES": devices} if devices is not None else {}
    print(json.dumps({"command_count": len(commands), "commands": [
        {"argv": list(command), "environment": environment} for command in commands
    ]}, indent=2), flush=True)
    if dry_run:
        return 0
    for command in commands:
        result = subprocess.run(command, env={**os.environ, **environment}, check=False)
        if result.returncode:
            return result.returncode
    return 0


def gridworld_artifact(root: Path, profile_name: str, split: str) -> Path:
    from tasks.gridworld.data.profiles import get_profile
    profile = get_profile(profile_name)
    spec = next(item for item in profile.splits if item.name == split)
    return root / profile.artifact_filename(spec)


def training_data(task: str, alpha: str, root: Path) -> tuple[Path, Path]:
    if task == "gridworld":
        profile = f"paper-alpha-{alpha}"
        return gridworld_artifact(root, profile, "practice"), gridworld_artifact(root, profile, "exam")
    if task == "arithmetic_factorization":
        from tasks.arithmetic_factorization.data.profiles import profile_for_alpha
        profile = profile_for_alpha(alpha)
        return root / profile.train.filename, root / profile.test.filename
    from tasks.image_editing.data.profiles import get_profile
    profile = get_profile(f"paper-alpha-{alpha}")
    return root / profile.get_split("train").filename, root / profile.get_split("id_test").filename
