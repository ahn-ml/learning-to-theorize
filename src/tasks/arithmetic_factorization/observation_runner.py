"""Pretrain the digit autoencoder on individual, unpaired numbers."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import torch

from tasks.arithmetic_factorization.observation_pretraining import observation_pretraining_config
from training.optimization import OptimizationConfig, build_cosine_warmup_scheduler

RUN_NAME = "arithmetic-observation-pretraining"


def select_reconstruction_checkpoint(run_directory: Path, final_step: int) -> dict:
    """Select the earliest saved maximum of number reconstruction accuracy."""
    from training.runtime import sha256_file

    candidates = []
    for line in (run_directory / "metrics.jsonl").read_text().splitlines():
        metrics = json.loads(line)
        if "reconstruction/number_accuracy" not in metrics:
            continue
        step = metrics["global_step"]
        checkpoint = run_directory / "checkpoints" / f"checkpoint_{step}.pth"
        if not checkpoint.is_file() and step == final_step:
            checkpoint = run_directory / "checkpoints" / "checkpoint_final.pth"
        if checkpoint.is_file():
            candidates.append((metrics["reconstruction/number_accuracy"], step,
                               metrics["completed_epochs"], checkpoint))
    if not candidates:
        raise ValueError("no saved reconstruction checkpoints")
    score, step, epoch, checkpoint = max(candidates, key=lambda item: (item[0], -item[1]))
    return {
        "checkpoint": str(checkpoint.relative_to(run_directory)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "metric": "number_accuracy", "score": score,
        "global_step": step, "completed_epochs": epoch,
        "tie_break": "earliest_step",
    }


@torch.no_grad()
def evaluate_reconstruction(model, loader, fabric) -> dict[str, float]:
    """Measure reconstruction on the observation vocabulary, not task transfer."""
    model.eval()
    totals = torch.zeros(5, device=fabric.device, dtype=torch.float64)
    for observations in loader:
        output = model(observations)
        count = observations.shape[0]
        totals[0] += output["loss"].double() * count
        totals[1] += output["digit_correct"].double()
        totals[2] += output["number_correct"].double()
        totals[3] += count
        totals[4] += observations.numel()
    totals = fabric.all_reduce(totals, reduce_op="sum")
    return {
        "loss": float(totals[0] / totals[3]),
        "digit_accuracy": float(totals[1] / totals[4]),
        "number_accuracy": float(totals[2] / totals[3]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretraining-profile", choices=("appendix",), default="appendix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--observations-h5", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--wandb-project", default="LearningToTheorize")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    arguments = parser.parse_args(argv)

    from lightning.fabric import Fabric
    from lightning.fabric.strategies import DDPStrategy
    from tasks.arithmetic_factorization.data.observations import build_observation_loader
    from tasks.arithmetic_factorization.observation import ArithmeticObservationModel
    from training.runtime import ExperimentTrackingConfig, MetricLog, capture_git_source, seed_everything, sha256_file

    config = observation_pretraining_config(arguments.pretraining_profile, seed=arguments.seed)
    source = capture_git_source(Path(__file__).resolve().parent)
    fabric = Fabric(
        accelerator="auto", devices=config.effective_world_size, precision="bf16-mixed",
        strategy=DDPStrategy(timeout=timedelta(minutes=30), process_group_backend="nccl"),
    )
    fabric.launch()
    seed_everything(config.seed)
    fabric.seed_everything(config.seed)
    train_loader = fabric.setup_dataloaders(build_observation_loader(
        arguments.observations_h5, batch_size=config.per_rank_batch_size,
        shuffle=True, seed=config.seed,
    ))
    reconstruction_loader = fabric.setup_dataloaders(build_observation_loader(
        arguments.observations_h5, batch_size=config.per_rank_batch_size,
        shuffle=False, seed=config.seed,
    ))
    total_steps = config.epochs * len(train_loader)
    model = ArithmeticObservationModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    model, optimizer = fabric.setup(model, optimizer)
    scheduler = build_cosine_warmup_scheduler(optimizer, OptimizationConfig(
        total_steps=total_steps, learning_rate=config.learning_rate,
        weight_decay=config.weight_decay, max_gradient_norm=config.max_gradient_norm,
        warmup_ratio=config.warmup_ratio, minimum_learning_rate_ratio=config.minimum_learning_rate_ratio,
    ))

    stamp = fabric.broadcast(datetime.utcnow().strftime("%Y%m%dT%H%M%S.%fZ"), src=0)
    run_directory = arguments.output_root.expanduser().resolve() / RUN_NAME / stamp
    checkpoints = run_directory / "checkpoints"
    resolved = {
        "schema_version": 2, "domain": "arithmetic_factorization", "stage": "pretraining",
        "canonical": True, "pretraining_profile": arguments.pretraining_profile,
        "observation_architecture": "digit-embedding-linear", "objective": "auto-reconstruction",
        "use_vae": False, "parameters": asdict(config),
        "artifacts": {"observations_h5": str(arguments.observations_h5.resolve()),
                      "observations_sha256": sha256_file(arguments.observations_h5)},
        "source": asdict(source),
    }
    metric_log = None
    if fabric.global_rank == 0:
        checkpoints.mkdir(parents=True, exist_ok=False)
        (run_directory / "config.json").write_text(json.dumps(resolved, indent=2) + "\n")
        metric_log = MetricLog(run_directory / "metrics.jsonl", ExperimentTrackingConfig(
            project=arguments.wandb_project, name=f"{RUN_NAME}-{stamp}", entity=arguments.wandb_entity,
            group=arguments.wandb_group or RUN_NAME, mode=arguments.wandb_mode,
            tags=("arithmetic_factorization", "observation-pretraining"),
        ), resolved)
    fabric.barrier()

    def save_checkpoint(name: str, completed_epochs: int, step: int) -> Path:
        path = checkpoints / name
        if fabric.global_rank == 0:
            core = model.module if hasattr(model, "module") else model
            torch.save({
                "format_version": 2, "epoch": completed_epochs, "global_step": step,
                "model_state_dict": core.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "config": resolved,
            }, path)
        return path

    exit_code = 0
    global_step = 0
    try:
        initial = evaluate_reconstruction(model, reconstruction_loader, fabric)
        if metric_log is not None:
            metric_log.log({"completed_epochs": 0, "global_step": 0,
                            **{f"reconstruction/{key}": value for key, value in initial.items()}})
        for epoch in range(config.epochs):
            model.train()
            for observations in train_loader:
                output = model(observations)
                if not torch.isfinite(output["loss"]):
                    raise FloatingPointError("nonfinite arithmetic pretraining loss")
                fabric.backward(output["loss"])
                fabric.clip_gradients(model, optimizer, max_norm=config.max_gradient_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                if metric_log is not None and global_step % config.log_interval_steps == 0:
                    metric_log.log({"epoch": epoch, "global_step": global_step,
                                    "train/loss": float(output["loss"]),
                                    "learning_rate": scheduler.get_last_lr()[0]})
            if (epoch + 1) % config.save_interval_epochs == 0:
                metrics = evaluate_reconstruction(model, reconstruction_loader, fabric)
                if metric_log is not None:
                    metric_log.log({"completed_epochs": epoch + 1, "global_step": global_step,
                                    **{f"reconstruction/{key}": value for key, value in metrics.items()}})
                save_checkpoint(f"checkpoint_{global_step}.pth", epoch + 1, global_step)
            if fabric.global_rank == 0:
                (run_directory / "status.json").write_text(json.dumps({
                    "status": "running", "completed_epochs": epoch + 1,
                    "global_step": global_step, "expected_steps": total_steps,
                }))
        final_metrics = evaluate_reconstruction(model, reconstruction_loader, fabric)
        final_checkpoint = save_checkpoint("checkpoint_final.pth", config.epochs, global_step)
        if fabric.global_rank == 0:
            metric_log.log({"completed_epochs": config.epochs, "global_step": global_step,
                            **{f"reconstruction/{key}": value for key, value in final_metrics.items()}})
            selected = select_reconstruction_checkpoint(run_directory, global_step)
            (run_directory / "best_reconstruction.json").write_text(json.dumps(selected, indent=2) + "\n")
            (run_directory / "status.json").write_text(json.dumps({
                "status": "completed", "completed_epochs": config.epochs,
                "global_step": global_step, "expected_steps": total_steps,
                "final_checkpoint": str(final_checkpoint), "reconstruction": final_metrics,
                "best_checkpoint": str(run_directory / selected["checkpoint"]),
            }))
        fabric.barrier()
    except Exception:
        exit_code = 1
        if fabric.global_rank == 0:
            (run_directory / "status.json").write_text(json.dumps({"status": "failed", "global_step": global_step}))
        raise
    finally:
        if metric_log is not None:
            metric_log.finish(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
