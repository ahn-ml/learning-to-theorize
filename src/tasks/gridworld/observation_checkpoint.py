"""Hand pretrained GridWorld observation weights to NEO training.

The checkpoint passed by the user is used as given. It must be a release-v1
checkpoint written by the observation runner with exactly the encoder and
decoder parameters of the GridWorld VAE. When it sits in a run's
``best_reconstruction`` directory, that run's ``best_reconstruction.json``
must name it.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn


_RELEASE_FIELDS = {
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "training_state",
    "optimization",
    "metadata",
}
_OBSERVATION_PREFIXES = ("encoder.", "decoder.")


def inspect_observation_checkpoint(checkpoint_path: str | Path) -> str:
    """Validate an observation checkpoint and return its SHA-256 digest."""

    digest, _ = _read_observation_checkpoint(checkpoint_path)
    return digest


def load_gridworld_observation_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[str, ...]:
    """Load the encoder and decoder weights and return the loaded keys."""

    digest, state_dict = _read_observation_checkpoint(checkpoint_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            "observation checkpoint SHA-256 changed: "
            f"expected {expected_sha256}, got {digest}"
        )
    expected = {key for key in model.state_dict() if key.startswith(_OBSERVATION_PREFIXES)}
    missing = sorted(expected - set(state_dict))
    unexpected = sorted(set(state_dict) - expected)
    if missing or unexpected:
        raise ValueError(
            "observation checkpoint does not match the GridWorld VAE: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.load_state_dict(state_dict, strict=False)
    return tuple(sorted(state_dict))


def _read_observation_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[str, dict[str, Tensor]]:
    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    payload: Any = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("format_version") != 1:
        raise ValueError("unsupported observation checkpoint; expected release format v1")
    missing = sorted(_RELEASE_FIELDS - set(payload))
    if missing:
        raise ValueError(f"release-v1 observation checkpoint is missing fields: {missing}")
    state_dict = payload["model_state_dict"]
    if not isinstance(state_dict, Mapping) or not all(
        isinstance(key, str) and isinstance(value, Tensor)
        for key, value in state_dict.items()
    ):
        raise ValueError("observation checkpoint has no valid model_state_dict")
    _check_selection_record(path, digest)
    return digest, {
        key: value
        for key, value in state_dict.items()
        if key.startswith(_OBSERVATION_PREFIXES)
    }


def _check_selection_record(path: Path, digest: str) -> None:
    if path.parent.name != "best_reconstruction":
        return
    record_path = path.parent.parent / "best_reconstruction.json"
    if not record_path.is_file():
        return
    record = json.loads(record_path.read_text())
    recorded = Path(str(record.get("checkpoint", "")))
    if recorded.name != path.name or record.get("checkpoint_sha256") != digest:
        raise ValueError(
            f"{record_path} selects {recorded.name or 'no checkpoint'}, not {path.name}; "
            "pass the checkpoint it names"
        )


__all__ = [
    "inspect_observation_checkpoint",
    "load_gridworld_observation_checkpoint",
]
