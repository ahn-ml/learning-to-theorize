"""Train ImageEditing NEO for one alpha and seed with two GPU ranks.

Each rank uses batch size 64. The programmer and executor learning rates are
0.25x and 0.5x the base rate, and the MDL coefficient follows a per-step linear
schedule. The pretrained observation model stays frozen. DDP uses
``find_unused_parameters=True`` because some programmer parameters are not
used by the forward pass.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler, Subset

from tasks.image_editing.data.dataset import (
    EpisodeBatch,
    ImageEditingDataset,
    collate_episodes,
)
from tasks.image_editing.experiment_config import (
    Experiment,
    PAPER_ALPHAS,
    PAPER_SEEDS,
    get_experiment,
)
from tasks.image_editing.task import build_neo
from training.optimization import (
    OptimizationConfig,
    build_cosine_warmup_scheduler,
    optimization_step,
)
from training.runtime import (
    DistributedContext,
    ExperimentTrackingConfig,
    MetricLog,
    barrier,
    bfloat16_autocast,
    capture_git_source,
    create_run_directory,
    initialize_distributed,
    seed_everything,
)


CHECKPOINT_INTERVAL_EPOCHS = 5
LOG_INTERVAL_STEPS = 50


@dataclass(frozen=True, slots=True)
class TheorizerRunConfig:
    """One resolved training invocation."""

    alpha: str
    seed: int
    train_h5: Path
    test_h5: Path
    observation_checkpoint: Path
    output_root: Path
    wandb_project: str = "LearningToTheorize"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_mode: str = "online"

    @property
    def experiment(self) -> Experiment:
        return get_experiment(self.alpha)

    @property
    def run_name(self) -> str:
        return f"image-editing-neo-alpha{self.alpha}-seed{self.seed}"


def load_observation_checkpoint(model: nn.Module, path: Path) -> int:
    """Load the pretrained encoder/decoder into a composed model."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    observation = {
        key: value
        for key, value in state.items()
        if key.startswith(("encoder.", "decoder."))
    }
    if not observation:
        raise ValueError(f"no encoder/decoder tensors found in {path}")
    missing, unexpected = model.load_state_dict(observation, strict=False)
    if unexpected:
        raise ValueError(f"observation checkpoint has unexpected keys: {unexpected[:5]}")
    still_missing = [
        k for k in missing if k.startswith(("encoder.", "decoder."))
    ]
    if still_missing:
        raise ValueError(f"observation checkpoint is missing keys: {still_missing[:5]}")
    return len(observation)


def build_optimizer(model: nn.Module, experiment: Experiment) -> AdamW:
    """AdamW with the paper's two-timescale parameter groups."""

    training = experiment.training
    programmer, executor, remainder = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("theory_programmer."):
            programmer.append(parameter)
        elif name.startswith("program_executor."):
            executor.append(parameter)
        else:
            remainder.append(parameter)
    base = training.learning_rate
    groups = [
        {"params": programmer, "lr": base * training.policy_learning_rate_scale},
        {"params": executor, "lr": base * training.transition_learning_rate_scale},
        {"params": remainder, "lr": base},
    ]
    return AdamW(
        [group for group in groups if group["params"]],
        lr=base,
        weight_decay=training.weight_decay,
    )


def _loader(
    dataset: ImageEditingDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    context: DistributedContext,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Build a loader whose per-rank batch is the contract's batch size.

    Under data parallelism the sampler splits the dataset across ranks, so each
    rank sees ``batch_size`` episodes and the effective batch is
    ``batch_size * world_size`` -- 128 for the paper's two ranks.
    """

    sampler: DistributedSampler | None = None
    if context.is_distributed and not shuffle:
        # Evaluate every episode exactly once, without DistributedSampler padding.
        dataset = Subset(dataset, range(context.rank, len(dataset), context.world_size))
    elif context.is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=shuffle,
            seed=seed,
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return (
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_episodes,
            pin_memory=True,
            generator=generator,
        ),
        sampler,
    )


def _forward(
    model: nn.Module,
    batch: EpisodeBatch,
    *,
    is_eval: bool,
    coefficient: float,
):
    # bf16 mixed precision on CUDA; the quantizer opts out of autocast internally.
    with bfloat16_autocast(batch.grids.device):
        return model(
            batch.grids,
            is_eval=is_eval,
            length_control_coefficient=coefficient,
        )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    coefficient: float,
) -> dict[str, float]:
    model.eval()
    sums = dict.fromkeys(
        ("support_l1", "query_l1", "query_reconstruction_loss", "mean_explanation_length"), 0.0
    )
    count = 0
    for batch in loader:
        output = _forward(model, batch.to(device), is_eval=True, coefficient=coefficient)
        record = {
            "support_l1": float(output.metrics.l1),
            "query_l1": float(output.query_metrics.l1),
            "query_reconstruction_loss": float(output.query_reconstruction_loss),
            "mean_explanation_length": float(output.mean_explanation_length),
        }
        for key, value in record.items():
            sums[key] += value * len(batch)
        count += len(batch)
    if torch.distributed.is_initialized():
        totals = torch.tensor([*sums.values(), count], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(totals)
        sums = dict(zip(sums, totals[:-1].tolist()))
        count = totals[-1].item()
    return {key: value / max(count, 1) for key, value in sums.items()}


def run_theorizer(config: TheorizerRunConfig) -> Path:
    """Train one condition and return the run directory."""

    experiment = config.experiment
    training = experiment.training
    context = initialize_distributed()
    if context.world_size != training.world_size:
        raise ValueError(
            f"this condition trained on {training.world_size} ranks with a "
            f"per-rank batch of {training.batch_size}; torchrun reports "
            f"{context.world_size}. Launch with "
            f"--nproc-per-node={training.world_size}, or the effective batch "
            "and optimizer step count will not match the paper."
        )
    seed_everything(config.seed)
    device = context.device

    model = build_neo(experiment)
    loaded = load_observation_checkpoint(model, config.observation_checkpoint)
    model.to(device)
    core = model
    if context.is_distributed:
        # The theory programmer constructs FiLM and length-embedding parameters
        # it never uses, so DDP must tolerate unused parameters, as the paper's
        # Fabric DDPStrategy did.
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    epochs = training.epochs
    batch_size = training.batch_size
    num_workers = training.num_workers

    train_dataset = ImageEditingDataset(config.train_h5)
    test_dataset = ImageEditingDataset(config.test_h5)
    train_loader, train_sampler = _loader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=config.seed,
        context=context,
    )
    test_loader, _ = _loader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        seed=config.seed + 1,
        context=context,
    )

    total_steps = epochs * len(train_loader)
    optimization = OptimizationConfig(
        total_steps=total_steps,
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        max_gradient_norm=training.gradient_clip_norm,
        warmup_ratio=training.warmup_ratio,
        minimum_learning_rate_ratio=training.minimum_learning_rate_ratio,
    )
    optimizer = build_optimizer(core, experiment)
    scheduler = build_cosine_warmup_scheduler(optimizer, optimization)

    run_directory = create_run_directory(
        config.output_root,
        config.run_name,
        capture_git_source(Path(__file__).resolve().parents[3]),
        context,
    )
    if context.is_global_zero:
        (run_directory / "checkpoints").mkdir(exist_ok=True)
    resolved = {
        "task": "image_editing",
        "method": "neo",
        "alpha": config.alpha,
        "seed": config.seed,
        "total_steps": total_steps,
        "world_size": context.world_size,
        "per_rank_batch_size": batch_size,
        "effective_batch_size": batch_size * context.world_size,
        "observation_tensors_loaded": loaded,
        **{f"training.{k}": v for k, v in asdict(training).items()},
    }
    log: MetricLog | None = None
    if context.is_global_zero:
        (run_directory / "config.json").write_text(json.dumps(resolved, indent=2))
        log = MetricLog(
            run_directory / "metrics.jsonl",
            ExperimentTrackingConfig(
                project=config.wandb_project,
                name=config.run_name,
                entity=config.wandb_entity,
                group=config.wandb_group,
                mode=config.wandb_mode,
            ),
            resolved,
        )

    def coefficient_at(step: int) -> float:
        return experiment.scheduled_length_control(step / max(total_steps, 1))

    exit_code = 1
    try:
        global_step = 0
        start = time.monotonic()
        for epoch in range(epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for batch in train_loader:
                moved = batch.to(device)
                coefficient = coefficient_at(global_step)
                step_result = optimization_step(
                    model,
                    moved,
                    lambda m, b: _forward(m, b, is_eval=False, coefficient=coefficient),
                    optimizer,
                    scheduler,
                    global_step=global_step,
                    max_gradient_norm=training.gradient_clip_norm,
                )
                core.quantizer.step()
                global_step = step_result.global_step
                if log is not None and (
                    global_step % LOG_INTERVAL_STEPS == 0 or global_step == total_steps
                ):
                    log.log(
                        {
                            "train/step": global_step,
                            "train/epoch": epoch,
                            "train/loss": float(step_result.objective.loss.detach()),
                            "train/reconstruction_loss": float(
                                step_result.objective.reconstruction_loss.detach()
                            ),
                            "train/gradient_norm": float(step_result.gradient_norm),
                            "train/learning_rate": step_result.learning_rates[-1],
                            "train/length_control_coefficient": coefficient,
                        }
                    )
            evaluation = evaluate(
                core,
                test_loader,
                device,
                coefficient=coefficient_at(global_step),
            )
            if log is not None:
                log.log(
                {
                    "eval/step": global_step,
                    "eval/epoch": epoch,
                    **{f"eval/{k}": v for k, v in evaluation.items()},
                }
                )
            if context.is_global_zero and (
                (epoch + 1) % CHECKPOINT_INTERVAL_EPOCHS == 0 or epoch == epochs - 1
            ):
                torch.save(
                    {
                        "model_state_dict": core.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "config": resolved,
                    },
                    run_directory / "checkpoints" / f"checkpoint_{global_step}.pt",
                )
        if context.is_global_zero:
            (run_directory / "result.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "global_step": global_step,
                    "seconds": time.monotonic() - start,
                    "final_evaluation": evaluation,
                },
                indent=2,
                )
            )
        exit_code = 0
    finally:
        if log is not None:
            log.finish(exit_code=exit_code)
        barrier(context)
        if context.owns_process_group:
            torch.distributed.destroy_process_group()
    return run_directory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", choices=PAPER_ALPHAS, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=PAPER_SEEDS)
    parser.add_argument("--train-h5", type=Path, required=True)
    parser.add_argument("--test-h5", type=Path, required=True)
    parser.add_argument("--observation-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wandb-project", default="LearningToTheorize")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    arguments = parser.parse_args(argv)

    run_directory = run_theorizer(
        TheorizerRunConfig(
            alpha=arguments.alpha,
            seed=arguments.seed,
            train_h5=arguments.train_h5,
            test_h5=arguments.test_h5,
            observation_checkpoint=arguments.observation_checkpoint,
            output_root=arguments.output_root,
            wandb_project=arguments.wandb_project,
            wandb_entity=arguments.wandb_entity,
            wandb_group=arguments.wandb_group,
            wandb_mode=arguments.wandb_mode,
        )
    )
    print(json.dumps({"run_directory": str(run_directory)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
