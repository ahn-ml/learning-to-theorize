"""Paper-facing handoff from observation pretraining to the GridWorld theorizer.

The public runner accepts only the safe release envelope written by this
repository. The default handoff preserves the recorded final checkpoint
contract. Explicit best-reconstruction selection requires a completed run.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch

from tasks.gridworld.models.vae import VAE
from tasks.gridworld.data.artifacts import get_artifact_spec
from tasks.gridworld.observation_pretraining import (
    GridWorldObservationPretrainingConfig,
)
from training.optimization import (
    EpochTrainingState,
    OptimizationConfig,
)
from training.runtime import sha256_file


PAPER_OBSERVATION_CHECKPOINT_POSITION = "after_eval_before_train"
PAPER_OBSERVATION_CHECKPOINT_EPOCH = 499
PAPER_OBSERVATION_CHECKPOINT_GLOBAL_STEP = 122_255
PAPER_OBSERVATION_CHECKPOINT_NUMBER = 122_256
OBSERVATION_SELECTION_POLICIES = ("recorded-final", "best-reconstruction")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_RELEASE_FIELDS = {
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "training_state",
    "optimization",
    "metadata",
}


@dataclass(frozen=True, slots=True)
class ObservationCheckpointEvidence:
    """Identity and provenance recorded for a theorizer input checkpoint."""

    checkpoint_format: str
    sha256: str
    matches_paper_pretraining_record: bool
    historical_checkpoint_number: int | None
    source_commit: str | None

    pretraining_profile: str = "recovered"
    kl_weight: float = 0.0
    matches_appendix_pretraining_record: bool = False
    selection_policy: str = "recorded-final"
    selection_manifest_sha256: str | None = None
    selection_metrics_sha256: str | None = None
    pretraining_status_sha256: str | None = None

    def as_dict(self) -> dict[str, str | int | float | bool | None]:
        values = asdict(self)
        if self.selection_policy == "recorded-final":
            for key in ("selection_policy", "selection_manifest_sha256",
                        "selection_metrics_sha256", "pretraining_status_sha256"):
                values.pop(key)
        return values


@dataclass(frozen=True, slots=True)
class ObservationCheckpointLoadReport:
    """Selected observation weights and the evidence that authorized loading."""

    evidence: ObservationCheckpointEvidence
    loaded_keys: tuple[str, ...]
    ignored_keys: tuple[str, ...]


def inspect_observation_checkpoint(
    checkpoint_path: str | Path,
    *,
    require_canonical: bool,
    expected_sha256: str | None = None,
    selection_policy: str = "recorded-final",
) -> ObservationCheckpointEvidence:
    """Validate a release-v1 theorizer input and record its provenance."""

    if selection_policy not in OBSERVATION_SELECTION_POLICIES:
        raise ValueError("unknown observation selection policy")
    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            "observation checkpoint SHA-256 changed: "
            f"expected {expected_sha256}, got {digest}"
        )
    payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("observation checkpoint must contain a mapping")
    if (
        type(payload.get("format_version")) is int
        and payload["format_version"] == 1
    ):
        evidence = _inspect_release_v1(payload, digest)
        if selection_policy == "best-reconstruction":
            return _inspect_best_reconstruction(path, payload, evidence)
        if require_canonical and not (evidence.matches_paper_pretraining_record or evidence.matches_appendix_pretraining_record):
            raise ValueError(
                "canonical theorizer runs require a release-v1 checkpoint that "
                "matches a supported pretraining record"
            )
        return evidence

    raise ValueError(
        "unsupported observation checkpoint; expected release format v1"
    )


def load_gridworld_observation_checkpoint(
    model: VAE,
    checkpoint_path: str | Path,
    *,
    require_canonical: bool,
    map_location: str | torch.device = "cpu",
    expected_sha256: str | None = None,
    selection_policy: str = "recorded-final",
) -> ObservationCheckpointLoadReport:
    """Load encoder/decoder weights after validating the checkpoint handoff."""

    evidence = inspect_observation_checkpoint(
        checkpoint_path,
        require_canonical=require_canonical,
        expected_sha256=expected_sha256,
        selection_policy=selection_policy,
    )
    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    payload: Any = torch.load(path, map_location=map_location, weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or type(payload.get("format_version")) is not int
        or payload["format_version"] != 1
    ):
        raise ValueError("observation checkpoint changed while it was being loaded")
    _inspect_release_v1(payload, evidence.sha256)
    if sha256_file(path) != evidence.sha256:
        raise ValueError("observation checkpoint changed while it was being loaded")
    loaded_keys, ignored_keys = _load_release_observation_weights(
        model,
        payload["model_state_dict"],
    )
    return ObservationCheckpointLoadReport(
        evidence=evidence,
        loaded_keys=loaded_keys,
        ignored_keys=ignored_keys,
    )


def _inspect_best_reconstruction(
    path: Path,
    payload: Mapping[str, Any],
    evidence: ObservationCheckpointEvidence,
) -> ObservationCheckpointEvidence:
    """Validate the explicit best-selection handoff without relabeling history.

    This needs the completed pretraining run beside the snapshot. The recorded
    final-checkpoint flags remain false for an earlier selected snapshot.
    """
    if path.parent.name != "best_reconstruction":
        raise ValueError("best-reconstruction checkpoint must remain in its run's best_reconstruction directory")
    run = path.parent.parent
    try:
        config = json.loads((run / "config.json").read_text())
        status = json.loads((run / "status.json").read_text())
        selected = json.loads((run / "best_reconstruction.json").read_text())
        rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("best-reconstruction requires a complete pretraining run record") from error
    if not all(isinstance(record, dict) for record in (config, status, selected, *rows)):
        raise ValueError("best-reconstruction run records must be mappings")
    expected = GridWorldObservationPretrainingConfig(kl_weight=evidence.kl_weight)
    sweep = config.get("pretraining_sweep") is True
    if sweep:
        # Explicit non-paper profiles retain their provenance, but may be
        # transferred after a complete, verified reconstruction selection.
        values = config.get("pretraining", {})
        allowed = {"epochs", "kl_weight", "learning_rate", "weight_decay",
                   "minimum_learning_rate_ratio", "warmup_ratio"}
        baseline = asdict(expected)
        if set(values) != set(baseline) or any(
            values[k] != baseline[k] for k in baseline if k not in allowed
        ):
            raise ValueError("best-reconstruction pretraining configuration changed fixed fields")
        from tasks.gridworld.pretraining_sweep import apply_pretraining_values
        expected = apply_pretraining_values(expected, {k: values[k] for k in allowed})
    if (config.get("canonical") is not (not sweep)
            or payload["metadata"].get("canonical") is not (not sweep)
            or bool(payload["metadata"].get("pretraining_sweep", False)) != sweep
            or config.get("pretraining_profile") != evidence.pretraining_profile
            or config.get("pretraining") != asdict(expected)
            or payload["optimization"] != asdict(expected.optimization())):
        raise ValueError("best-reconstruction pretraining configuration does not match the selected profile")
    source = config.get("source", {})
    if (source.get("commit") != evidence.source_commit or evidence.source_commit is None
            or source.get("clean") is not True or source.get("pushed") is not True):
        raise ValueError("best-reconstruction requires matching clean, pushed pretraining provenance")
    for name, split in (("train", "practice"), ("test", "exam")):
        if config.get("artifacts", {}).get(name, {}).get("sha256") != get_artifact_spec("vae-pretraining", split).sha256:
            raise ValueError("best-reconstruction requires the recorded pretraining data identities")
    if (status.get("status") != "completed" or status.get("completed_epochs") != expected.epochs
            or status.get("global_step") != expected.total_steps or status.get("stopped_early") is not False):
        raise ValueError("best-reconstruction requires completed pretraining for the recorded configuration")
    required = {
        "policy": "global-validation-reconstruction-first-maximum-v1",
        "metric": "grid_accuracy", "direction": "max", "aggregation": "global_sample_weighted",
        "tie_break": "earliest", "selection_uses_downstream_scores": False,
        "selection_uses_ood_scores": False, "training_continues_after_selection": True,
    }
    if any(selected.get(key) != value for key, value in required.items()):
        raise ValueError("best-reconstruction selection policy does not match")
    evaluations = [row for row in rows if "eval/grid_accuracy" in row]
    if len(evaluations) != expected.epochs:
        raise ValueError("best-reconstruction requires every pretraining validation epoch")
    for epoch, row in enumerate(evaluations, 1):
        score = row["eval/grid_accuracy"]
        if (row.get("epoch") != epoch
                or row.get("global_step") != (epoch - 1) * expected.train_steps_per_epoch
                or type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("best-reconstruction validation trajectory is invalid")
    winner = max(evaluations, key=lambda row: row["eval/grid_accuracy"])
    state = {"completed_epochs": winner["epoch"] - 1, "global_step": winner["global_step"]}
    recorded_checkpoint = Path(selected.get("checkpoint", "")).expanduser()
    if not recorded_checkpoint.is_absolute():
        recorded_checkpoint = run / recorded_checkpoint
    if (selected.get("score") != winner["eval/grid_accuracy"] or selected.get("training_state") != state
            or payload["training_state"] != state or selected.get("checkpoint_sha256") != evidence.sha256
            or recorded_checkpoint.resolve() != path):
        raise ValueError("checkpoint is not the first maximum of the full reconstruction validation trajectory")
    return replace(evidence, selection_policy="best-reconstruction",
                   selection_manifest_sha256=sha256_file(run / "best_reconstruction.json"),
                   selection_metrics_sha256=sha256_file(run / "metrics.jsonl"),
                   pretraining_status_sha256=sha256_file(run / "status.json"))


def _inspect_release_v1(
    payload: Mapping[str, Any],
    digest: str,
) -> ObservationCheckpointEvidence:
    missing = sorted(_RELEASE_FIELDS - set(payload))
    if missing:
        raise ValueError(f"release-v1 observation checkpoint is missing fields: {missing}")
    _validate_model_state_mapping(payload["model_state_dict"])
    for field in ("optimizer_state_dict", "scheduler_state_dict"):
        if not isinstance(payload[field], Mapping):
            raise ValueError(f"release-v1 {field} must be a mapping")

    try:
        state = EpochTrainingState(**payload["training_state"])
    except (TypeError, ValueError) as error:
        raise ValueError("release-v1 training_state is invalid") from error
    try:
        optimization = asdict(
            OptimizationConfig(**payload["optimization"])
        )
    except (TypeError, ValueError) as error:
        raise ValueError("release-v1 optimization is invalid") from error
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping) or not all(
        isinstance(key, str) for key in metadata
    ):
        raise ValueError("release-v1 metadata must be a string-keyed mapping")
    if not all(
        isinstance(value, (str, int, float, bool, type(None)))
        for value in metadata.values()
    ):
        raise ValueError("release-v1 metadata must contain primitive values")

    expected_metadata = {
        "domain": "gridworld",
        "stage": "observation_pretraining",
        "position": PAPER_OBSERVATION_CHECKPOINT_POSITION,
        "evaluated_epoch": state.completed_epochs,
        "historical_checkpoint_number": state.global_step + 1,
    }
    mismatches = {
        key: (expected, metadata.get(key))
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"release-v1 observation metadata mismatch: {mismatches}")
    if not isinstance(metadata.get("canonical"), bool):
        raise ValueError("release-v1 metadata must declare canonical")

    source_commit = metadata.get("source_commit")
    if source_commit is not None and (
        not isinstance(source_commit, str) or _COMMIT.fullmatch(source_commit) is None
    ):
        raise ValueError("release-v1 source_commit must be a full Git commit or null")
    profile = metadata.get("pretraining_profile", "recovered")
    kl_weight = metadata.get("kl_weight", 0.0)
    sweep = metadata.get("pretraining_sweep", False)
    if sweep is not False:
        import math
        if (sweep is not True or metadata["canonical"] is not False
                or isinstance(kl_weight, bool) or not isinstance(kl_weight, (int, float))
                or not math.isfinite(kl_weight) or kl_weight < 0):
            raise ValueError("invalid non-paper pretraining sweep metadata")
    if profile not in ("recovered", "appendix") or (not sweep and kl_weight != (1e-5 if profile == "appendix" else 0.0)):
        raise ValueError("observation pretraining profile and KL weight disagree")
    matches_training_record = (
        metadata.get("domain") == "gridworld"
        and metadata.get("stage") == "observation_pretraining"
        and metadata.get("position") == PAPER_OBSERVATION_CHECKPOINT_POSITION
        and metadata.get("evaluated_epoch") == PAPER_OBSERVATION_CHECKPOINT_EPOCH
        and metadata.get("historical_checkpoint_number")
        == PAPER_OBSERVATION_CHECKPOINT_NUMBER
        and metadata.get("canonical") is True
        and isinstance(source_commit, str)
        and state
        == EpochTrainingState(
            completed_epochs=PAPER_OBSERVATION_CHECKPOINT_EPOCH,
            global_step=PAPER_OBSERVATION_CHECKPOINT_GLOBAL_STEP,
        )
        and optimization
        == asdict(GridWorldObservationPretrainingConfig().optimization())
    )
    checkpoint_number = metadata.get("historical_checkpoint_number")
    return ObservationCheckpointEvidence(
        checkpoint_format="release-v1",
        sha256=digest,
        matches_paper_pretraining_record=matches_training_record and profile == "recovered",
        pretraining_profile=profile,
        kl_weight=kl_weight,
        matches_appendix_pretraining_record=matches_training_record and profile == "appendix",
        historical_checkpoint_number=(
            checkpoint_number if isinstance(checkpoint_number, int) else None
        ),
        source_commit=source_commit if isinstance(source_commit, str) else None,
    )


def _validate_model_state_mapping(value: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError("observation checkpoint has no valid model_state_dict")
    if not all(isinstance(tensor, torch.Tensor) for tensor in value.values()):
        raise ValueError("observation model_state_dict must contain only tensors")
    return value  # type: ignore[return-value]


def _load_release_observation_weights(
    model: VAE,
    value: Any,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    state_dict = _validate_model_state_mapping(value)
    selected = {
        key: tensor
        for key, tensor in state_dict.items()
        if key.startswith(("encoder.", "decoder."))
    }
    expected = {
        key
        for key in model.state_dict()
        if key.startswith(("encoder.", "decoder."))
    }
    missing = sorted(expected - set(selected))
    unexpected = sorted(set(selected) - expected)
    if missing or unexpected:
        raise ValueError(
            "release-v1 observation checkpoint is incompatible: "
            f"missing={missing}, unexpected={unexpected}"
        )
    incompatible = model.load_state_dict(selected, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            "release-v1 observation load produced unexpected keys: "
            f"{sorted(incompatible.unexpected_keys)}"
        )
    return (
        tuple(sorted(selected)),
        tuple(sorted(set(state_dict) - set(selected))),
    )


__all__ = [
    "ObservationCheckpointEvidence",
    "ObservationCheckpointLoadReport",
    "PAPER_OBSERVATION_CHECKPOINT_EPOCH",
    "PAPER_OBSERVATION_CHECKPOINT_GLOBAL_STEP",
    "PAPER_OBSERVATION_CHECKPOINT_NUMBER",
    "PAPER_OBSERVATION_CHECKPOINT_POSITION",
    "inspect_observation_checkpoint",
    "load_gridworld_observation_checkpoint",
]
