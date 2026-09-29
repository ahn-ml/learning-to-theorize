"""Task commands."""

from tasks.cli import run

COMMANDS = {
    "environment": "training.environment",
    "data": "tasks.image_editing.data.cli",
    "train": "tasks.image_editing.theorizer_runner",
    "evaluate": "tasks.image_editing.evaluation",
}


def main(argv=None):
    return run("image_editing", COMMANDS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
