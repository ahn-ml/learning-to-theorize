"""Explicit hyperparameter experiments, separate from fixed paper profiles."""
from dataclasses import replace
import json
import math
from pathlib import Path


def apply_pretraining_overrides(config, path: Path):
    values = json.loads(path.read_text())
    return apply_pretraining_values(config, values)


def apply_pretraining_values(config, values):
    """Apply the supported numeric overrides without changing fixed data/topology."""
    allowed = {"epochs", "kl_weight", "learning_rate", "weight_decay",
               "minimum_learning_rate_ratio", "warmup_ratio"}
    if not isinstance(values, dict) or not values or set(values) - allowed:
        raise ValueError("expected a nonempty object of pretraining hyperparameters")
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"invalid numeric hyperparameter: {key}")
        if key == "epochs" and (not isinstance(value, int) or value < 1):
            raise ValueError("epochs must be a positive integer")
        if key in {"kl_weight", "weight_decay"} and value < 0:
            raise ValueError(f"{key} must be nonnegative")
        if key == "learning_rate" and value <= 0:
            raise ValueError("learning_rate must be positive")
        if key in {"minimum_learning_rate_ratio", "warmup_ratio"} and not 0 <= value <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    return replace(config, **values)
