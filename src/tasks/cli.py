"""Dispatch task-specific command-line entry points."""

import argparse
from importlib import import_module
from typing import Mapping, Sequence


def run(task: str, commands: Mapping[str, str], argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m tasks.{task}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in commands:
        subparsers.add_parser(name, add_help=False)
    args, remainder = parser.parse_known_args(argv)
    module = import_module(commands[args.command])
    result = module.main(remainder)
    return 0 if result is None else int(result)
