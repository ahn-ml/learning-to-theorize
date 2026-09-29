#!/usr/bin/env python3
"""Generate every canonical dataset for one paper task."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import TASKS, load_task_config, module_command, run_commands


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--config", type=Path, help="default: configs/<task>/reproduction.yaml")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    load_task_config(arguments.config, arguments.task)
    root = arguments.data_root.expanduser().resolve()
    module = f"tasks.{arguments.task}.data.cli"
    if arguments.task != "image_editing":
        commands = [module_command(module, "generate-all", "--output-dir", root)]
    else:
        from tasks.image_editing.data.profiles import available_profiles, get_profile
        cifar = root / "cifar-10-batches-py"
        if not arguments.dry_run and not cifar.is_dir():
            parser.error(f"place CIFAR-10 Python batches at {cifar}")
        commands = [module_command(module, "generate", "--profile", name,
                    "--cifar-dir", cifar, "--output-dir", root, "--split", split.name)
                    for name in available_profiles() for split in get_profile(name).splits
                    if not (root / split.filename).exists()]
    return run_commands(commands, dry_run=arguments.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
