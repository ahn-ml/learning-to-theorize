"""Evaluate one Image Editing NEO checkpoint on one paper protocol.

Evaluation protocols:

``id``
    The in-distribution test split, standard support/query forward with the
    MDL coefficient the training schedule had reached at the checkpoint.
``comp-ood``
    The compositional-OOD split, same forward on held-out programs.
``length-ood``
    The length-OOD split (programs of length 3-4).  NEO rolls out to six
    transitions with the MDL coefficient disabled, selecting purely by
    reconstruction loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from tasks.image_editing.data.dataset import ImageEditingDataset, collate_episodes
from tasks.image_editing.data.profiles import get_profile
from tasks.image_editing.experiment_config import get_experiment
from tasks.image_editing.task import build_neo
from training.runtime import bfloat16_autocast, seed_everything


PROTOCOLS = ("id", "comp-ood", "length-ood")

# NEO uses six transitions for length OOD.
LENGTH_OOD_TRANSITIONS = 6

EVALUATION_BATCH_SIZE = 64

_PROTOCOL_SPLIT = {
    "id": "id_test",
    "comp-ood": "comp_ood_test",
    "length-ood": "length_ood_test",
}


def load_model_checkpoint(model: nn.Module, path: Path, *, metadata: dict | None = None) -> int:
    """Load checkpoint weights into a composed model."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if metadata is not None:
        metadata.update(global_step=payload.get("global_step"), config=payload.get("config"))
    state = payload["model_state_dict"]
    model.load_state_dict(state, strict=True)
    return len(state)


def checkpoint_length_control(metadata: dict) -> float:
    """Return the MDL coefficient the training schedule had reached at the checkpoint."""
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no training configuration for schedule restoration")
    start = float(config["training.length_control_start"])
    if not config["training.length_control_scheduling"]:
        return start
    step, total = metadata.get("global_step"), config.get("total_steps")
    if not isinstance(step, int) or not isinstance(total, int) or total <= 0 or not 0 <= step <= total:
        raise ValueError("checkpoint has no valid optimizer progress for schedule restoration")
    end = float(config["training.length_control_end"])
    return start + (end - start) * (step / total)


@dataclass(frozen=True, slots=True)
class EvaluationRunConfig:
    """One resolved evaluation invocation."""

    alpha: str
    protocol: str
    checkpoint: Path
    data_h5: Path
    device: str
    seed: int
    hard_grounding: bool = False


def evaluate_checkpoint(config: EvaluationRunConfig) -> dict[str, float | int | str]:
    """Run one protocol and return aggregated metrics."""

    experiment = get_experiment(config.alpha)
    seed_everything(config.seed)
    device = torch.device(config.device)

    model = build_neo(experiment)
    metadata: dict = {}
    load_model_checkpoint(model, config.checkpoint, metadata=metadata)
    stored = metadata["config"]
    actual = (stored["method"], str(stored.get("alpha")), stored.get("seed"))
    expected = ("neo", config.alpha, config.seed)
    if actual != expected:
        raise ValueError(f"checkpoint condition {actual} does not match requested {expected}")
    model.to(device).eval()

    if config.protocol == "length-ood":
        forward_kwargs: dict = {
            "num_transitions": LENGTH_OOD_TRANSITIONS,
            "length_control_coefficient": 1.0,
        }
    else:
        forward_kwargs = {"length_control_coefficient": checkpoint_length_control(metadata)}

    dataset = ImageEditingDataset(config.data_h5)
    loader = DataLoader(
        dataset,
        batch_size=EVALUATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_episodes,
        pin_memory=True,
    )

    forward_kwargs["hard_grounding"] = config.hard_grounding
    sums: dict[str, float] = {}
    batches = 0
    episodes = 0
    for batch in loader:
        moved = batch.to(device)
        with torch.no_grad(), bfloat16_autocast(device):
            output = model(moved.grids, is_eval=True, **forward_kwargs)
        record = {
            "support_l1": float(output.metrics.l1),
            "query_l1": float(output.query_metrics.l1),
            "mean_explanation_length": float(output.mean_explanation_length),
        }
        for key, value in record.items():
            sums[key] = sums.get(key, 0.0) + value * len(batch)
        batches += 1
        episodes += len(batch)

    if episodes == 0:
        raise ValueError("empty ImageEditing evaluation")
    if not all(math.isfinite(value) for value in sums.values()):
        raise ValueError("non-finite ImageEditing evaluation metrics")
    results: dict[str, float | int | str] = {
        key: value / episodes for key, value in sums.items()
    }
    results.update(
        alpha=config.alpha,
        protocol=config.protocol,
        checkpoint=str(config.checkpoint),
        data_h5=str(config.data_h5),
        batches=batches,
        episodes=episodes,
        seed=config.seed,
        hard_grounding=config.hard_grounding,
        length_control_coefficient=forward_kwargs["length_control_coefficient"],
    )
    return results


def resolve_protocol_h5(alpha: str, protocol: str, data_root: Path) -> Path:
    """Map a protocol to its released artifact under ``data_root``."""

    if protocol == "length-ood":
        profile = get_profile("paper-length-ood")
    else:
        profile = get_profile(f"paper-alpha-{alpha}")
    split = profile.get_split(_PROTOCOL_SPLIT[protocol])
    return data_root / split.filename
