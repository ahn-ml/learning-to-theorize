"""Paper-exact deterministic GridWorld rollout evaluation and CLI runner."""

from __future__ import annotations

import argparse
import math
import platform
import socket
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor

from tasks.gridworld.data.profiles import get_profile
from models.neo import NEO
from tasks.gridworld.theorizer_config import (
    GRIDWORLD_LENGTH_OOD_ARTIFACT_SHA256,
    GRIDWORLD_THEORIZER_SEEDS,
    GridWorldAlphaExperiment,
    available_theorizer_experiments,
    get_theorizer_experiment,
)
from tasks.gridworld.theorizer_runner import (
    build_theorizer_loader,
)
from tasks.gridworld.theorizer_scaling import (
    GridWorldTestTimeScalingResult,
    evaluate_gridworld_test_time_scaling,
)
from tasks.gridworld.task import build_neo
from training.runtime import (
    DistributedContext,
    ExperimentTrackingConfig,
    GitSourceState,
    MetricLog,
    WandbMode,
    capture_git_source,
    create_run_directory,
    initialize_distributed,
    seed_everything,
    sha256_file,
    write_json_atomic,
    write_json_exclusive,
)


EvaluationSplit = Literal["id", "compositional-ood", "length-ood"]
EvaluationProtocol = Literal["standard", "test-time-scaling"]
EvaluationMethod = Literal[
    "neo",
]
PAPER_EVALUATION_SEED = 42
PAPER_SCALING_BUDGETS = (1, 4, 16, 64)
PAPER_SCALING_TEMPERATURE = 0.3


@dataclass(frozen=True, slots=True)
class GridWorldRolloutResult:
    """Final prediction and stopping information for one episode side."""

    final_prediction: Tensor
    action_indices: tuple[tuple[int, ...], ...]
    num_steps: int
    pixel_accuracy: float
    solved: bool


@dataclass(frozen=True, slots=True)
class GridWorldTransferResult:
    """Support-extracted program applied to one query input."""

    final_prediction: Tensor
    action_indices: tuple[tuple[int, ...], ...]
    num_extracted_steps: int
    num_applied_steps: int
    pixel_accuracy: float
    solved: bool


@dataclass(frozen=True, slots=True)
class GridWorldEvaluationMetrics:
    total: int
    solved: int
    grid_accuracy: float
    pixel_accuracy: float
    per_step_solved: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GridWorldEvaluationResult:
    """The standard self-explanation and transfer result for one split."""

    max_steps: int
    self_explanation: GridWorldEvaluationMetrics
    transfer: GridWorldEvaluationMetrics

    def legacy_dict(self) -> dict[str, Any]:
        """Return the serialized evaluation metrics."""

        result: dict[str, Any] = {"max_steps": self.max_steps, "eval_mode": "both"}
        for prefix, metrics in (
            ("self_expl", self.self_explanation),
            ("transfer", self.transfer),
        ):
            result.update(
                {
                    f"{prefix}_total": metrics.total,
                    f"{prefix}_solved": metrics.solved,
                    f"{prefix}_grid_acc": metrics.grid_accuracy,
                    f"{prefix}_pixel_acc": metrics.pixel_accuracy,
                    f"{prefix}_per_step_solved": {
                        str(step): count
                        for step, count in enumerate(metrics.per_step_solved, start=1)
                    },
                }
            )
        return result


@dataclass(frozen=True, slots=True)
class GridWorldEvaluationArtifact:
    filename: str
    sha256: str
    episodes: int
    max_steps: int


@dataclass(frozen=True, slots=True)
class GridWorldEvaluationRunConfig:
    experiment: str
    model_seed: int
    split: EvaluationSplit
    checkpoint: Path
    data_h5: Path
    output_root: Path
    method: EvaluationMethod = "neo"
    protocol: EvaluationProtocol = "standard"
    scaling_budgets: tuple[int, ...] = PAPER_SCALING_BUDGETS
    sample_temperature: float = PAPER_SCALING_TEMPERATURE
    run_name: str | None = None
    batch_size: int = 128
    num_workers: int = 0
    num_samples: int | None = None
    wandb_project: str = "LearningToTheorize"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_mode: WandbMode = "online"
    canonical: bool = True
    force_cpu: bool = False
    hard_grounding: bool = True

    def __post_init__(self) -> None:
        if self.method == "neo":
            get_theorizer_experiment(self.experiment)
        else:
            raise ValueError(f"unknown GridWorld evaluation method: {self.method}")
        if self.protocol not in ("standard", "test-time-scaling"):
            raise ValueError(f"unknown evaluation protocol: {self.protocol}")
        if self.model_seed < 0:
            raise ValueError("model_seed must be non-negative")
        if self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")
        if self.num_samples is not None and self.num_samples < 1:
            raise ValueError("num_samples must be positive when provided")
        _validate_scaling_settings(
            self.scaling_budgets,
            sample_temperature=self.sample_temperature,
        )
        if self.protocol == "standard" and (
            self.scaling_budgets != PAPER_SCALING_BUDGETS
            or self.sample_temperature != PAPER_SCALING_TEMPERATURE
        ):
            raise ValueError(
                "scaling settings require protocol test-time-scaling"
            )
        if self.canonical and self.protocol == "test-time-scaling":
            if self.scaling_budgets != PAPER_SCALING_BUDGETS:
                raise ValueError(
                    "canonical test-time scaling requires budgets 1,4,16,64"
                )
            if self.sample_temperature != PAPER_SCALING_TEMPERATURE:
                raise ValueError(
                    "canonical test-time scaling requires sample temperature 0.3"
                )
        if self.split == "compositional-ood" and self.experiment == "alpha-1.00":
            raise ValueError("alpha-1.00 has no held-out compositional split")

    @property
    def paper_experiment(
        self,
    ) -> GridWorldAlphaExperiment:
        return get_theorizer_experiment(self.experiment)

    @property
    def resolved_run_name(self) -> str:
        if self.run_name is not None:
            return self.run_name
        suffix = "" if self.protocol == "standard" else f"-{self.protocol}"
        return (
            f"gridworld-{self.experiment}-seed-{self.model_seed}-{self.split}"
            f"{suffix}"
        )


@dataclass(slots=True)
class _MetricAccumulator:
    max_steps: int
    total: int = 0
    solved: int = 0
    pixel_sum: float = 0.0
    per_step: list[int] | None = None

    def __post_init__(self) -> None:
        self.per_step = [0 for _ in range(self.max_steps)]

    def add(self, *, solved: bool, pixel_accuracy: float, num_steps: int) -> None:
        self.total += 1
        self.pixel_sum += pixel_accuracy
        if solved:
            self.solved += 1
            assert self.per_step is not None
            self.per_step[min(num_steps, self.max_steps) - 1] += 1

    def finish(self) -> GridWorldEvaluationMetrics:
        denominator = max(1, self.total)
        assert self.per_step is not None
        return GridWorldEvaluationMetrics(
            total=self.total,
            solved=self.solved,
            grid_accuracy=self.solved / denominator,
            pixel_accuracy=self.pixel_sum / denominator,
            per_step_solved=tuple(self.per_step),
        )


def evaluation_artifact(
    experiment: str | GridWorldAlphaExperiment,
    split: EvaluationSplit,
) -> GridWorldEvaluationArtifact:
    """Resolve the exact paper artifact and rollout budget for one split."""

    paper = (
        get_theorizer_experiment(experiment)
        if isinstance(experiment, str)
        else experiment
    )
    max_steps = paper.training.max_transition_length
    if split == "id":
        profile = paper.profile
        spec = next(item for item in profile.splits if item.name == "exam")
        return GridWorldEvaluationArtifact(
            profile.artifact_filename(spec),
            paper.test_artifact_sha256,
            spec.num_episodes,
            max_steps,
        )
    if split == "compositional-ood":
        if paper.compositional_ood_artifact_sha256 is None:
            raise ValueError(f"{paper.name} has no held-out compositional split")
        profile = paper.profile
        spec = next(item for item in profile.splits if item.name == "ood_test")
        return GridWorldEvaluationArtifact(
            profile.artifact_filename(spec),
            paper.compositional_ood_artifact_sha256,
            spec.num_episodes,
            max_steps,
        )
    if split == "length-ood":
        profile = get_profile("paper-length-4-8")
        spec = next(item for item in profile.splits if item.name == "exam")
        return GridWorldEvaluationArtifact(
            profile.artifact_filename(spec),
            GRIDWORLD_LENGTH_OOD_ARTIFACT_SHA256,
            spec.num_episodes,
            10,
        )
    raise ValueError(f"unknown evaluation split: {split}")


def self_explanation_rollout(
    model: NEO,
    input_grid: Tensor,
    target_grid: Tensor,
    *,
    max_steps: int,
    device: torch.device,
    hard_grounding: bool = True,
) -> GridWorldRolloutResult:
    """Infer actions toward a target, optionally re-encoding every prediction."""

    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    with torch.no_grad():
        current_grid = input_grid.unsqueeze(0).to(device).long()
        target = target_grid.unsqueeze(0).to(device).long()
        current, _ = model.encoder(current_grid)
        target_state, _ = model.encoder(target)
        action_indices: list[tuple[int, ...]] = []
        prediction = current_grid.squeeze(0)

        for _ in range(max_steps):
            action = model.theory_programmer(current, target_state)
            values, indices = _quantize_for_evaluation(model, action)
            if indices is not None:
                action_indices.append(_indices_tuple(indices))
            current = model.program_executor(current, values)
            prediction = model.decoder(current).argmax(dim=-1).squeeze(0)
            if hard_grounding:
                current, _ = model.encoder(prediction.unsqueeze(0))
            if torch.equal(prediction, target.squeeze(0)):
                break

        solved = torch.equal(prediction, target.squeeze(0))
        pixel_accuracy = float(prediction.eq(target.squeeze(0)).float().mean().item())
        return GridWorldRolloutResult(
            final_prediction=prediction.cpu(),
            action_indices=tuple(action_indices),
            num_steps=len(action_indices),
            pixel_accuracy=pixel_accuracy,
            solved=solved,
        )


def transfer_rollout(
    model: NEO,
    support_input: Tensor,
    support_output: Tensor,
    query_input: Tensor,
    query_target: Tensor,
    *,
    max_steps: int,
    device: torch.device,
    hard_grounding: bool = True,
) -> GridWorldTransferResult:
    """Extract support actions and replay them with the same grounding setting."""

    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    with torch.no_grad():
        support_in = support_input.unsqueeze(0).to(device).long()
        support_out = support_output.unsqueeze(0).to(device).long()
        query_in = query_input.unsqueeze(0).to(device).long()
        query_out = query_target.unsqueeze(0).to(device).long()

        current, _ = model.encoder(support_in)
        support_target, _ = model.encoder(support_out)
        extracted_actions: list[Tensor] = []
        action_indices: list[tuple[int, ...]] = []
        for _ in range(max_steps):
            action = model.theory_programmer(current, support_target)
            values, indices = _quantize_for_evaluation(model, action)
            extracted_actions.append(values.clone())
            if indices is not None:
                action_indices.append(_indices_tuple(indices))
            current = model.program_executor(current, extracted_actions[-1])
            support_prediction = model.decoder(current).argmax(dim=-1)
            if hard_grounding:
                current, _ = model.encoder(support_prediction)
            if torch.equal(support_prediction.squeeze(0), support_out.squeeze(0)):
                break

        current, _ = model.encoder(query_in)
        prediction = query_in.squeeze(0)
        applied_steps = 0
        for action in extracted_actions:
            current = model.program_executor(current, action)
            prediction = model.decoder(current).argmax(dim=-1).squeeze(0)
            applied_steps += 1
            if hard_grounding:
                current, _ = model.encoder(prediction.unsqueeze(0))

        solved = torch.equal(prediction, query_out.squeeze(0))
        pixel_accuracy = float(prediction.eq(query_out.squeeze(0)).float().mean().item())
        return GridWorldTransferResult(
            final_prediction=prediction.cpu(),
            action_indices=tuple(action_indices),
            num_extracted_steps=len(extracted_actions),
            num_applied_steps=applied_steps,
            pixel_accuracy=pixel_accuracy,
            solved=solved,
        )


def evaluate_theorizer(
    model: NEO,
    batches: Iterable[Tensor],
    *,
    max_steps: int,
    device: torch.device,
    hard_grounding: bool = True,
    num_samples: int | None = None,
) -> GridWorldEvaluationResult:
    """Evaluate sequential episodes and aggregate support/query metrics."""

    model.eval()
    self_metrics = _MetricAccumulator(max_steps)
    transfer_metrics = _MetricAccumulator(max_steps)
    for batch in batches:
        if batch.ndim != 4 or batch.shape[1] < 4:
            raise ValueError("evaluation batches must have shape (batch, >=4, H, W)")
        for episode in batch:
            self_result = self_explanation_rollout(
                model,
                episode[0],
                episode[1],
                max_steps=max_steps,
                device=device,
                hard_grounding=hard_grounding,
            )
            self_metrics.add(
                solved=self_result.solved,
                pixel_accuracy=self_result.pixel_accuracy,
                num_steps=self_result.num_steps,
            )
            transfer_result = transfer_rollout(
                model,
                episode[0],
                episode[1],
                episode[2],
                episode[3],
                max_steps=max_steps,
                device=device,
                hard_grounding=hard_grounding,
            )
            transfer_metrics.add(
                solved=transfer_result.solved,
                pixel_accuracy=transfer_result.pixel_accuracy,
                num_steps=transfer_result.num_applied_steps,
            )
            if num_samples is not None and self_metrics.total >= num_samples:
                return GridWorldEvaluationResult(
                    max_steps,
                    self_metrics.finish(),
                    transfer_metrics.finish(),
                )
    return GridWorldEvaluationResult(
        max_steps,
        self_metrics.finish(),
        transfer_metrics.finish(),
    )


def load_theorizer_checkpoint(
    model: NEO,
    path: str | Path,
    *,
    expected_experiment: str | None = None,
    expected_method: str | None = None,
    expected_seed: int | None = None,
) -> Mapping[str, Any]:
    """Load a release checkpoint written by the public training runner."""

    checkpoint_path = Path(path).expanduser().resolve(strict=True)
    payload: Any = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("theorizer checkpoint must contain a mapping")
    release_format = payload.get("format_version") == 1
    if not release_format:
        raise ValueError("evaluation requires a release-format checkpoint")
    state_dict = payload.get("model_state_dict", payload)
    if not isinstance(state_dict, Mapping):
        raise ValueError("theorizer checkpoint has no model_state_dict")
    arguments = payload.get("args")
    if not isinstance(arguments, dict):
        raise ValueError("release checkpoint is missing resolved arguments")
    if expected_experiment is not None and arguments.get("experiment") != expected_experiment:
        raise ValueError("checkpoint experiment does not match evaluation")
    checkpoint_method = arguments.get("method", "neo")
    if expected_method is not None and checkpoint_method != expected_method:
        raise ValueError("checkpoint method does not match evaluation")
    if expected_seed is not None and arguments.get("seed") != expected_seed:
        raise ValueError("checkpoint seed does not match evaluation")
    model.load_state_dict(state_dict, strict=True)
    return payload


def run_theorizer_evaluation(
    config: GridWorldEvaluationRunConfig,
) -> tuple[Path, GridWorldEvaluationResult | GridWorldTestTimeScalingResult]:
    """Run one auditable paper evaluation and write local/W&B evidence."""

    context = initialize_distributed(force_cpu=config.force_cpu)
    if context.world_size != 1:
        raise ValueError("GridWorld evaluation requires exactly one process")
    # Capture the repository that supplied this imported module.  Using the
    # caller's current working directory can mislabel a run when an editable
    # installation still points at a different checkout.
    source = capture_git_source(Path(__file__).resolve().parent)
    _validate_run_mode(config, context, source)
    artifact = evaluation_artifact(config.paper_experiment, config.split)
    evidence = _artifact_evidence(config, artifact)
    output = create_run_directory(
        config.output_root,
        config.resolved_run_name,
        source,
        context,
    )
    resolved = _resolved_config(config, context, source, artifact, evidence)
    write_json_exclusive(output / "config.json", resolved)
    write_json_exclusive(output / "source.json", asdict(source))
    write_json_atomic(
        output / "status.json",
        _status("running", config.canonical, protocol=config.protocol),
    )

    metric_log: MetricLog | None = None
    loader = None
    succeeded = False
    try:
        metric_log = MetricLog(
            output / "metrics.jsonl",
            ExperimentTrackingConfig(
                project=config.wandb_project,
                entity=config.wandb_entity,
                group=config.wandb_group or f"gridworld-{config.method}-{config.experiment}",
                name=config.resolved_run_name,
                mode=config.wandb_mode,
                tags=(
                    "gridworld",
                    "paper-evaluation",
                    config.method,
                    config.split,
                    config.protocol,
                ),
            ),
            resolved,
        )
        seed_everything(PAPER_EVALUATION_SEED)
        model = build_neo(config.paper_experiment)
        load_theorizer_checkpoint(
            model,
            config.checkpoint,
            expected_experiment=config.experiment,
            expected_method=config.method,
            expected_seed=config.model_seed,
        )
        model.to(context.device).eval()
        loader = build_theorizer_loader(
            config.data_h5,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=PAPER_EVALUATION_SEED,
            shuffle=False,
            pin_memory=context.device.type == "cuda",
            episode_limit=config.num_samples,
        )
        batches = loader.batches()
        _consume_historical_dataloader_iterator_seed()
        if config.protocol == "standard":
            result: GridWorldEvaluationResult | GridWorldTestTimeScalingResult = (
                evaluate_theorizer(
                    model,
                    batches,
                    max_steps=artifact.max_steps,
                    device=context.device,
                    num_samples=config.num_samples,
                    hard_grounding=config.hard_grounding,
                )
            )
        else:
            result = evaluate_gridworld_test_time_scaling(
                model,
                batches,
                max_steps=artifact.max_steps,
                budgets=config.scaling_budgets,
                sample_temperature=config.sample_temperature,
                device=context.device,
                num_samples=config.num_samples,
                hard_grounding=config.hard_grounding,
            )
        legacy = result.legacy_dict()
        write_json_exclusive(output / "result.json", legacy)
        metrics = _flat_metrics(legacy)
        metric_log.log(metrics)
        write_json_atomic(
            output / "status.json",
            _status(
                "completed",
                config.canonical,
                protocol=config.protocol,
                result="result.json",
            ),
        )
        succeeded = True
        return output, result
    except BaseException as error:
        write_json_atomic(
            output / "status.json",
            _status(
                "failed",
                config.canonical,
                protocol=config.protocol,
                error=f"{type(error).__name__}: {error}",
            ),
        )
        raise
    finally:
        if loader is not None:
            loader.close()
        if metric_log is not None:
            metric_log.finish(exit_code=0 if succeeded else 1)


def _indices_tuple(indices: Tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in indices.detach().reshape(-1).cpu().tolist())


def _consume_historical_dataloader_iterator_seed() -> int:
    """Match the final evaluator's generator-less DataLoader iteration.

    PyTorch draws one int64 base seed from the global generator whenever a
    DataLoader iterator is constructed, even with zero workers.  The pinned
    evaluator left ``generator=None``.  The shared release loader uses a
    dedicated generator because theorizer training checkpoints its state, so
    evaluation must preserve this one global draw explicitly before NEO-S
    sampling.
    """

    return int(torch.empty((), dtype=torch.int64).random_().item())


def _quantize_for_evaluation(
    model: NEO,
    action: Tensor,
) -> tuple[Tensor, Tensor]:
    """Accept a result dataclass or a tuple of evaluation metrics."""

    quantized = model.quantizer(action, training_mode=False)
    if hasattr(quantized, "values") and hasattr(quantized, "indices"):
        return quantized.values, quantized.indices
    if isinstance(quantized, tuple) and len(quantized) == 2:
        values, metadata = quantized
        if isinstance(metadata, Mapping) and "min_encoding_indices" in metadata:
            return values, metadata["min_encoding_indices"]
    raise TypeError("unsupported action quantizer evaluation result")


def _validate_run_mode(
    config: GridWorldEvaluationRunConfig,
    context: DistributedContext,
    source: GitSourceState,
) -> None:
    if not config.canonical:
        return
    if config.model_seed not in GRIDWORLD_THEORIZER_SEEDS:
        raise ValueError("canonical evaluation requires model seed 42, 43, or 44")
    if config.num_samples is not None or config.force_cpu:
        raise ValueError("canonical evaluation cannot limit samples or force CPU")
    if context.device.type != "cuda" or torch.cuda.device_count() != 1:
        raise ValueError("canonical evaluation requires exactly one visible CUDA GPU")
    if config.wandb_mode == "disabled":
        raise ValueError("canonical evaluation requires online or offline W&B")


def _artifact_evidence(
    config: GridWorldEvaluationRunConfig,
    expected: GridWorldEvaluationArtifact,
) -> dict[str, Any]:
    data = Path(config.data_h5).expanduser().resolve(strict=True)
    checkpoint = Path(config.checkpoint).expanduser().resolve(strict=True)
    evidence = {
        "data": {
            "path": str(data),
            "bytes": data.stat().st_size,
            "sha256": sha256_file(data),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
    }
    if config.canonical:
        if data.name != expected.filename:
            raise ValueError(f"canonical data filename must be {expected.filename}")
        if evidence["data"]["sha256"] != expected.sha256:
            raise ValueError("evaluation data does not match the paper SHA-256")
        with h5py.File(data, "r") as file:
            if _paper_hdf5_episode_count(file) != expected.episodes:
                raise ValueError("evaluation artifact episode count does not match the paper")
    return evidence


def _paper_hdf5_episode_count(file: h5py.File) -> int:
    """Validate and count the paper artifact's root-level sample groups."""

    raw_length = file.attrs.get("dataset_length")
    if raw_length is None:
        raise ValueError("evaluation artifact is missing dataset_length")
    try:
        num_episodes = int(raw_length)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "evaluation artifact dataset_length must be an integer"
        ) from error
    if num_episodes < 1 or raw_length != num_episodes:
        raise ValueError(
            "evaluation artifact dataset_length must be a positive integer"
        )

    sample_indices: set[int] = set()
    for name in file:
        prefix = "sample_"
        if not name.startswith(prefix) or not name[len(prefix) :].isdigit():
            raise ValueError(
                f"evaluation artifact has unexpected root object {name!r}"
            )
        index = int(name[len(prefix) :])
        if name != f"{prefix}{index}":
            raise ValueError(
                f"evaluation artifact has non-canonical sample name {name!r}"
            )
        sample_indices.add(index)

    if sample_indices != set(range(num_episodes)):
        raise ValueError(
            "evaluation artifact sample groups do not match dataset_length"
        )
    return num_episodes


def _resolved_config(
    config: GridWorldEvaluationRunConfig,
    context: DistributedContext,
    source: GitSourceState,
    artifact: GridWorldEvaluationArtifact,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "domain": "gridworld",
        "stage": "theorizer_evaluation",
        "canonical": config.canonical,
        "method": config.method,
        "experiment": config.experiment,
        "model_seed": config.model_seed,
        "evaluation_seed": PAPER_EVALUATION_SEED,
        "rng": {
            "seed_before_model_construction": PAPER_EVALUATION_SEED,
            "dataloader_iterator_base_seed": "one_global_int64_draw",
        },
        "protocol": config.protocol,
        "split": config.split,
        "max_steps": artifact.max_steps,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "num_samples": config.num_samples,
        "rollout": {
            "action_selection": (
                "nearest_vq_code"
                if config.protocol == "standard"
                else "sample_vq_code"
            ),
            "ground_after_each_step": config.hard_grounding,
            "support_early_stop": config.protocol == "standard",
            "query_early_stop": False,
            "aggregation": "per_episode",
        },
        "test_time_scaling": (
            {
                "budgets": list(config.scaling_budgets),
                "sample_temperature": config.sample_temperature,
                "sample_max_budget_once": True,
                "reuse_nested_prefixes": True,
                "selection": "majority_action_sequence",
                "tie_break": "first_seen",
            }
            if config.protocol == "test-time-scaling"
            else None
        ),
        "artifacts": dict(evidence),
        "source": asdict(source),
        "tracking": {
            "project": config.wandb_project,
            "entity": config.wandb_entity,
            "group": config.wandb_group or f"gridworld-{config.method}-{config.experiment}",
            "mode": config.wandb_mode,
        },
        "environment": {
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "hdf5": h5py.version.hdf5_version,
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "device": str(context.device),
        },
    }


def _flat_metrics(legacy: Mapping[str, Any]) -> dict[str, int | float]:
    return {
        key: value
        for key, value in legacy.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _status(status: str, canonical: bool, **extra: Any) -> dict[str, Any]:
    return {
        "status": status,
        "canonical": canonical,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }


def _validate_scaling_settings(
    budgets: Sequence[int],
    *,
    sample_temperature: float,
) -> None:
    resolved = tuple(budgets)
    if not resolved or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in resolved
    ):
        raise ValueError("scaling budgets must contain positive integers")
    if tuple(sorted(set(resolved))) != resolved:
        raise ValueError("scaling budgets must be unique and strictly increasing")
    if not math.isfinite(sample_temperature) or sample_temperature <= 0:
        raise ValueError("sample temperature must be positive")


def parse_scaling_budgets(value: str) -> tuple[int, ...]:
    parts = tuple(part.strip() for part in value.split(","))
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "scaling budgets must be comma-separated integers without blanks"
        )
    try:
        budgets = tuple(int(part) for part in parts)
        _validate_scaling_settings(
            budgets,
            sample_temperature=PAPER_SCALING_TEMPERATURE,
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return budgets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one GridWorld model")
    parser.add_argument(
        "--method",
        choices=("neo",),
        default="neo",
        help="paper method to evaluate (default: neo)",
    )
    parser.add_argument("--experiment", choices=available_theorizer_experiments(), required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument(
        "--protocol",
        choices=("standard", "test-time-scaling"),
        default="standard",
    )
    parser.add_argument(
        "--split",
        choices=("id", "compositional-ood", "length-ood"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-h5", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="evaluation batch size (paper default: 128)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--scaling-budgets",
        type=parse_scaling_budgets,
        default=PAPER_SCALING_BUDGETS,
        metavar="K1,K2,...",
        help="nested NEO-S sample budgets (paper: 1,4,16,64)",
    )
    parser.add_argument(
        "--sample-temperature",
        type=float,
        default=PAPER_SCALING_TEMPERATURE,
        help="NEO-S VQ sampling temperature (paper: 0.3)",
    )
    parser.add_argument("--wandb-project", default="LearningToTheorize")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--hard-grounding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--development-cpu", action="store_true")
    parser.add_argument("--development-num-samples", type=int)
    return parser


def _config_from_arguments(arguments: argparse.Namespace) -> GridWorldEvaluationRunConfig:
    if not arguments.development and (
        arguments.development_cpu
        or arguments.development_num_samples is not None
    ):
        raise ValueError("development overrides require --development")
    return GridWorldEvaluationRunConfig(
        method=arguments.method,
        experiment=arguments.experiment,
        model_seed=arguments.model_seed,
        split=arguments.split,
        checkpoint=arguments.checkpoint,
        data_h5=arguments.data_h5,
        output_root=arguments.output_root,
        protocol=arguments.protocol,
        scaling_budgets=arguments.scaling_budgets,
        sample_temperature=arguments.sample_temperature,
        run_name=arguments.run_name,
        batch_size=(
            arguments.batch_size
            if arguments.batch_size is not None
            else 128
        ),
        num_workers=arguments.num_workers,
        num_samples=arguments.development_num_samples,
        wandb_project=arguments.wandb_project,
        wandb_entity=arguments.wandb_entity,
        wandb_group=arguments.wandb_group,
        wandb_mode=arguments.wandb_mode,
        canonical=not arguments.development,
        force_cpu=arguments.development_cpu,
        hard_grounding=arguments.hard_grounding,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = _config_from_arguments(build_parser().parse_args(argv))
    output, result = run_theorizer_evaluation(config)
    print(f"output_directory={output}")
    if isinstance(result, GridWorldEvaluationResult):
        print(f"transfer_grid_accuracy={result.transfer.grid_accuracy:.6f}")
    else:
        maximum = result.budgets[-1]
        print(f"select@{maximum.budget}_transfer={maximum.select_transfer:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
