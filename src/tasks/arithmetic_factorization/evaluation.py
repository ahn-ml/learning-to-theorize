"""Arithmetic NEO and NEO-S evaluation entry point."""

import sys
from tasks.neo_evaluation import main as evaluate_neo


def main(argv=None):
    return evaluate_neo(["--task", "arithmetic_factorization", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
