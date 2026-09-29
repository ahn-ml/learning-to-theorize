"""Task commands."""

from tasks.cli import run

COMMANDS = {
    "environment": "training.environment",
    "data": "tasks.gridworld.data.cli",
    "pretrain": "tasks.gridworld.observation_runner",
    "train": "tasks.gridworld.theorizer_runner",
    "evaluate": "tasks.gridworld.theorizer_evaluation",
}


def main(argv=None):
    return run("gridworld", COMMANDS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
