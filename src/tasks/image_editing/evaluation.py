"""Evaluate one Image Editing checkpoint on one paper protocol.

Evaluation protocols:

``id``
    The in-distribution test split, standard support/query forward.
``comp-ood``
    The compositional-OOD split, same forward on held-out programs.
``length-ood``
    The length-OOD split (programs of length 3-4).  NEO rolls out to six
    transitions with the MDL coefficient disabled, selecting purely by
    reconstruction loss.

Checkpoints support both ``theory_programmer.*``/``program_executor.*`` and
``policy.*``/``transition.*`` parameter names. Frozen ``lpips_fn.*`` metric
tensors are ignored when loading model weights.

LPIPS is reported when the optional ``lpips`` package is installed and
``--lpips`` is passed; it never enters a gradient.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from tasks.image_editing.data.dataset import ImageEditingDataset, collate_episodes
from tasks.image_editing.data.profiles import get_profile
from tasks.image_editing.experiment_config import (
    PAPER_ALPHAS,
    PAPER_EVALUATION,
    get_experiment,
)
from tasks.image_editing.task import build_neo
from training.runtime import (
    ExperimentTrackingConfig,
    MetricLog,
    bfloat16_autocast,
    seed_everything,
    sha256_file,
)


PROTOCOLS = ("id", "comp-ood", "length-ood")
EVALUATED_METHODS = ("neo",)

# NEO uses six transitions for length OOD.
LENGTH_OOD_TRANSITIONS = 6

_PROTOCOL_SPLIT = {
    "id": "id_test",
    "comp-ood": "comp_ood_test",
    "length-ood": "length_ood_test",
}

_HISTORICAL_RENAMES = (
    ("policy.", "theory_programmer."),
    ("transition.", "program_executor."),
)


class LpipsMetric(nn.Module):
    """NHWC ``[0, 1]`` adapter over the reference ``lpips`` implementation."""

    def __init__(self) -> None:
        super().__init__()
        import lpips as lpips_library

        self.distance = lpips_library.LPIPS(net="alex", verbose=False)
        self.distance.eval()
        for parameter in self.distance.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(self, prediction, target, *, reduction: str = "mean"):
        prediction = prediction.permute(0, 3, 1, 2) * 2.0 - 1.0
        target = target.permute(0, 3, 1, 2) * 2.0 - 1.0
        values = self.distance(prediction.float(), target.float()).view(-1)
        return values.mean() if reduction == "mean" else values


def load_model_checkpoint(model: nn.Module, path: Path, *, metadata: dict | None = None) -> int:
    """Load checkpoint weights into a composed model."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if metadata is not None:
        metadata.update(global_step=payload.get("global_step"), config=payload.get("config"))
    state = payload.get("model_state_dict", payload)
    mapped = {}
    for key, value in state.items():
        if key.startswith("lpips_fn."):
            continue
        for old, new in _HISTORICAL_RENAMES:
            if key.startswith(old):
                key = new + key[len(old):]
                break
        mapped[key] = value
    model.load_state_dict(mapped, strict=True)
    return len(mapped)


def checkpoint_length_control(metadata: dict) -> float:
    """Restore the saved run's schedule, without guessing progress for old weights."""
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

    method: str
    alpha: str
    protocol: str
    checkpoint: Path
    data_h5: Path
    device: str
    seed: int
    batch_size: int = PAPER_EVALUATION.batch_size
    num_workers: int = PAPER_EVALUATION.num_workers
    use_lpips: bool = False
    max_batches: int | None = None
    restore_checkpoint_schedule: bool = False
    hard_grounding: bool = False


def evaluate_checkpoint(config: EvaluationRunConfig) -> dict[str, float | int | str]:
    """Run one protocol and return aggregated metrics."""

    experiment = get_experiment(config.method, config.alpha)
    seed_everything(config.seed)
    device = torch.device(config.device)

    model = build_neo(experiment)
    metadata: dict = {}
    tensors = load_model_checkpoint(model, config.checkpoint, metadata=metadata)
    stored = metadata.get("config")
    # Raw state dictionaries have no training condition to verify.
    if stored is not None and "method" in stored:
        actual = (stored["method"], str(stored.get("alpha")), stored.get("seed"))
        expected = (config.method, config.alpha, config.seed)
        if actual != expected:
            raise ValueError(f"checkpoint condition {actual} does not match requested {expected}")
    model.to(device).eval()

    lpips = LpipsMetric().to(device) if config.use_lpips else None
    if lpips is not None:
        model.objective.lpips = lpips

    forward_kwargs: dict = {}
    coefficient_source = "historical-default"
    if config.restore_checkpoint_schedule:
        forward_kwargs["length_control_coefficient"] = checkpoint_length_control(metadata)
        coefficient_source = "checkpoint-progress"
    if config.protocol == "length-ood":
        coefficient_source = "length-ood-protocol"
        forward_kwargs = {
            "num_transitions": LENGTH_OOD_TRANSITIONS,
            "length_control_coefficient": 1.0,
        }

    dataset = ImageEditingDataset(config.data_h5)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_episodes,
        pin_memory=True,
    )

    forward_kwargs["hard_grounding"] = config.hard_grounding
    sums: dict[str, float] = {}
    batches = 0
    episodes = 0
    for index, batch in enumerate(loader):
        if config.max_batches is not None and index >= config.max_batches:
            break
        moved = batch.to(device)
        with torch.no_grad(), bfloat16_autocast(device):
            output = model(moved.grids, is_eval=True, **forward_kwargs)
        record = {
            "support_l1": float(output.metrics.l1),
            "support_lpips": float(output.metrics.lpips),
            "query_l1": float(output.query_metrics.l1),
            "query_lpips": float(output.query_metrics.lpips),
        }
        if hasattr(output, "mean_explanation_length"):
            record["mean_explanation_length"] = float(output.mean_explanation_length)
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
        method=config.method,
        alpha=config.alpha,
        protocol=config.protocol,
        checkpoint=str(config.checkpoint),
        data_h5=str(config.data_h5),
        batches=batches,
        episodes=episodes,
        checkpoint_tensors=tensors,
        seed=config.seed,
    )
    results["hard_grounding"] = config.hard_grounding
    results["length_control_coefficient"] = forward_kwargs.get(
        "length_control_coefficient", experiment.length_control_coefficient
    )
    results["length_control_source"] = coefficient_source
    if not config.use_lpips:
        results.pop("support_lpips", None)
        results.pop("query_lpips", None)
    return results


def resolve_protocol_h5(alpha: str, protocol: str, data_root: Path) -> Path:
    """Map a protocol to its released artifact under ``data_root``."""

    if protocol == "length-ood":
        profile = get_profile("paper-length-ood")
    else:
        profile = get_profile(f"paper-alpha-{alpha}")
    split = profile.get_split(_PROTOCOL_SPLIT[protocol])
    return data_root / split.filename


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=EVALUATED_METHODS, required=True)
    parser.add_argument("--alpha", choices=PAPER_ALPHAS, required=True)
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=PAPER_EVALUATION.seeds[0])
    parser.add_argument("--batch-size", type=int, default=PAPER_EVALUATION.batch_size)
    parser.add_argument(
        "--num-workers", type=int, default=PAPER_EVALUATION.num_workers
    )
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--restore-checkpoint-schedule", action="store_true",
                        help="Use the saved training schedule at checkpoint progress for ID/compositional evaluation")
    parser.add_argument("--development-max-batches", type=int)
    parser.add_argument("--hard-grounding", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default="LearningToTheorize")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    arguments = parser.parse_args(argv)
    if arguments.wandb_mode != "disabled" and arguments.output is None:
        parser.error("--output is required when W&B logging is enabled")
    if arguments.output:
        arguments.output = arguments.output.expanduser().resolve()
        if arguments.output.exists():
            raise FileExistsError(arguments.output)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)

    if arguments.protocol == "comp-ood" and arguments.alpha == "1.00":
        raise SystemExit("alpha 1.00 has no compositional-OOD split")

    config = EvaluationRunConfig(
        method=arguments.method,
        alpha=arguments.alpha,
        protocol=arguments.protocol,
        checkpoint=arguments.checkpoint,
        data_h5=resolve_protocol_h5(
            arguments.alpha, arguments.protocol, arguments.data_root
        ),
        device=arguments.device,
        seed=arguments.seed,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        use_lpips=arguments.lpips,
        max_batches=arguments.development_max_batches,
        restore_checkpoint_schedule=arguments.restore_checkpoint_schedule,
        hard_grounding=arguments.hard_grounding,
    )
    log = None
    if arguments.wandb_mode != "disabled":
        resolved = {key: str(value) if isinstance(value, Path) else value
                    for key, value in asdict(config).items()}
        resolved["checkpoint_sha256"] = sha256_file(config.checkpoint)
        log = MetricLog(
            arguments.output.with_suffix(".metrics.jsonl"),
            ExperimentTrackingConfig(
                project=arguments.wandb_project,
                name=f"{config.method}-{config.alpha}-{config.seed}-{config.protocol}",
                entity=arguments.wandb_entity,
                group=arguments.wandb_group,
                mode=arguments.wandb_mode,
            ),
            resolved,
        )
    success = False
    try:
        results = evaluate_checkpoint(config)
        rendered = json.dumps(results, indent=2, sort_keys=True)
        print(rendered)
        if arguments.output:
            arguments.output.write_text(rendered + "\n")
        if log is not None:
            log.log({key: value for key, value in results.items()
                     if isinstance(value, (int, float))})
        success = True
    finally:
        if log is not None:
            log.finish(exit_code=0 if success else 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
