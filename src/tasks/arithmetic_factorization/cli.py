"""Task commands."""

from tasks.cli import run

COMMANDS = {
    "data": "tasks.arithmetic_factorization.data.cli",
    "pretrain": "tasks.arithmetic_factorization.observation_runner",
    "train": "tasks.arithmetic_factorization.theorizer_runner",
    "evaluate": "tasks.arithmetic_factorization.evaluation",
    "environment": "training.environment",
}


def main(argv=None):
    return run("arithmetic_factorization", COMMANDS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
