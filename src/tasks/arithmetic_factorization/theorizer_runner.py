"""Train one arithmetic factorization model for a method, alpha, and seed."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import torch

from tasks.arithmetic_factorization.experiment import (
    ARITHMETIC_ALPHAS,
    ARITHMETIC_SEEDS,
    get_experiment,
    trainable_method,
)
from tasks.arithmetic_factorization.observation_pretraining import (
    TRANSFERRED_PREFIXES,
)
from training.optimization import (
    OptimizationConfig,
    build_cosine_warmup_scheduler,
    build_program_optimizer,
)


def run_name(method: str, alpha: str, seed: int) -> str:
    """Return the canonical run name for one training cell."""

    return f"arithmetic-{method}-alpha-{alpha}-seed-{seed}"


def load_observation_model(model: torch.nn.Module, checkpoint: Path) -> dict[str, Any]:
    """Transfer and freeze the pretrained digit autoencoder."""

    from tasks.arithmetic_factorization.task import released_state_dict

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    transferred = released_state_dict(
        {
            key: value
            for key, value in state.items()
            if key.startswith(TRANSFERRED_PREFIXES)
        }
    )
    if not transferred:
        raise ValueError(f"no observation parameters found in {checkpoint}")
    model.load_state_dict(transferred, strict=False)
    for name, parameter in model.named_parameters():
        if name in transferred:
            parameter.requires_grad = False
    model.freeze_observation_model()
    return transferred


def build_optimizer(model: torch.nn.Module, training) -> torch.optim.AdamW:
    """Build the paper's two-timescale optimizer over the shared modules."""

    optimization = training.optimization
    return build_program_optimizer(
        model,
        learning_rate=optimization.learning_rate,
        programmer_learning_rate=(
            optimization.learning_rate * optimization.theory_programmer_learning_rate_scale
        ),
        executor_learning_rate=(
            optimization.learning_rate * optimization.program_executor_learning_rate_scale
        ),
        weight_decay=optimization.weight_decay,
    )


def _evaluate(model, loader, fabric) -> dict[str, float]:
    from tasks.arithmetic_factorization.data.dataset import unpack_batch
    from tasks.arithmetic_factorization.validation import released_transfer_batch

    totals: dict[str, float] = {}
    episodes = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            data, _ = unpack_batch(batch, fabric.device)
            output = model(data, is_eval=True)
            observed = {
                "loss": float(output.loss),
                "reconstruction_loss": float(output.reconstruction_loss),
                "self_explanation_accuracy": output.metrics.number_accuracy,
                "digit_accuracy": output.metrics.digit_accuracy,
                "mean_explanation_length": float(output.mean_explanation_length),
            }
            if output.query_metrics is not None:
                observed["latent_transfer_accuracy"] = output.query_metrics.number_accuracy
                observed["latent_self_explanation_accuracy"] = observed["self_explanation_accuracy"]
                observed["latent_mean_explanation_length"] = observed["mean_explanation_length"]
                core = model.module if hasattr(model, "module") else model
                with torch.autocast(device_type=data.device.type, enabled=False):
                    paper_output = core(data, is_eval=True)
                observed["paper_transfer_accuracy"] = paper_output.query_metrics.number_accuracy
                observed["paper_self_explanation_accuracy"] = paper_output.metrics.number_accuracy
                observed["paper_mean_explanation_length"] = float(paper_output.mean_explanation_length)
                transferred = released_transfer_batch(
                    core, data, max_steps=core.experiment.training.max_transition_length
                )
                observed["transfer_accuracy"] = float(transferred["query_solved"].float().mean())
                observed["self_explanation_accuracy"] = float(transferred["support_solved"].float().mean())
                observed["digit_accuracy"] = float(transferred["support_digit_correct"].mean())
                observed["mean_explanation_length"] = float(transferred["selected_lengths"].float().mean())
            for key, value in observed.items():
                totals[key] = totals.get(key, 0.0) + value * len(data)
            episodes += len(data)
    return {key: value / max(episodes, 1) for key, value in totals.items()}


def setup_training(fabric, model, training):
    """Group original parameter names before Fabric adds wrapper prefixes."""
    from tasks.arithmetic_factorization.fabric_output import fabric_output_hook

    model.register_forward_hook(fabric_output_hook)
    optimizer = build_optimizer(model, training)
    return fabric.setup(model, optimizer)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("neo",), required=True)
    parser.add_argument("--alpha", required=True, choices=ARITHMETIC_ALPHAS)
    parser.add_argument("--seed", required=True, type=int, choices=ARITHMETIC_SEEDS)
    parser.add_argument("--train-h5", required=True, type=Path)
    parser.add_argument("--test-h5", required=True, type=Path)
    parser.add_argument("--observation-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--wandb-project", default="LearningToTheorize")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    arguments = parser.parse_args(argv)

    from lightning.fabric import Fabric
    from lightning.fabric.strategies import DDPStrategy

    from tasks.arithmetic_factorization.data.dataset import build_dataloader, unpack_batch
    from tasks.arithmetic_factorization.task import build_neo
    from training.runtime import (
        ExperimentTrackingConfig,
        MetricLog,
        capture_git_source,
        seed_everything,
    )

    method = trainable_method(arguments.method)
    training = get_experiment(method, arguments.alpha)
    optimization = training.optimization

    fabric = Fabric(
        accelerator="auto",
        devices=1,
        precision=optimization.precision,
        strategy=DDPStrategy(
            find_unused_parameters=True,
            timeout=timedelta(minutes=30),
            process_group_backend="nccl",
        ),
    )
    fabric.launch()
    seed_everything(arguments.seed)
    # Preserve the requested seed if Fabric injects a distributed sampler.
    fabric.seed_everything(arguments.seed)

    train_loader = fabric.setup_dataloaders(
        build_dataloader(
            arguments.train_h5,
            batch_size=optimization.batch_size,
            shuffle=True,
            seed=arguments.seed,
        )
    )
    test_loader = fabric.setup_dataloaders(
        build_dataloader(
            arguments.test_h5,
            batch_size=optimization.batch_size,
            shuffle=False,
            seed=arguments.seed,
        )
    )

    total_steps = optimization.epochs * len(train_loader)
    if total_steps != training.total_steps:
        raise ValueError(
            "resolved schedule does not match the paper contract; expected "
            f"{training.total_steps} steps, got {total_steps}"
        )

    model = build_neo(method, arguments.alpha)
    load_observation_model(model, arguments.observation_checkpoint)
    model, optimizer = setup_training(fabric, model, training)
    schedule = OptimizationConfig(
        total_steps=total_steps,
        learning_rate=optimization.learning_rate,
        weight_decay=optimization.weight_decay,
        max_gradient_norm=optimization.max_gradient_norm,
        warmup_ratio=optimization.warmup_ratio,
        minimum_learning_rate_ratio=optimization.minimum_learning_rate_ratio,
    )
    scheduler = build_cosine_warmup_scheduler(optimizer, schedule)

    name = run_name(method, arguments.alpha, arguments.seed)
    source = capture_git_source(Path(__file__).resolve().parent)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S.%fZ")
    run_directory = arguments.output_root.expanduser().resolve() / name / stamp
    checkpoints = run_directory / "checkpoints"
    resolved = {
        "schema_version": 1,
        "domain": "arithmetic_factorization",
        "stage": "training",
        "canonical": True,
        "validation_protocol": "released reencoded and paper latent-MDL transfer in FP32; bf16 latent metrics separately",
        "checkpoint_policy": "save every improved ID under either FP32 protocol, every fifth epoch, and final state",
        "action_sampling": {
            "training": "categorical softmax(-squared_distance/tau)" if training.quantizer.stochastic else "nearest code",
            "gradient_estimator": "VQ identity straight-through; not relaxed Gumbel-Softmax",
            "tau_steps": training.action_tau_steps,
            "clock": "completed optimizer steps; forward/evaluation never advances it",
            "orthogonal_regularization": (
                "active: normalized code Gram loss; optimizer correction added to EMA centroid and carried into EMA sums"
                if training.quantizer.orthogonal_regularization_weight > 0
                else "inactive"
            ),
        },
        "method": method,
        "alpha": arguments.alpha,
        "seed": arguments.seed,
        "training": asdict(training),
        "artifacts": {
            "train_h5": str(arguments.train_h5),
            "test_h5": str(arguments.test_h5),
            "observation_checkpoint": str(arguments.observation_checkpoint),
        },
        "source": asdict(source),
    }
    metric_log = None
    if fabric.global_rank == 0:
        checkpoints.mkdir(parents=True, exist_ok=False)
        (run_directory / "config.json").write_text(
            json.dumps(resolved, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        metric_log = MetricLog(
            run_directory / "metrics.jsonl",
            ExperimentTrackingConfig(
                project=arguments.wandb_project,
                name=f"{name}-{stamp}",
                entity=arguments.wandb_entity,
                group=arguments.wandb_group or f"arithmetic-{method}-{arguments.alpha}",
                mode=arguments.wandb_mode,
                tags=("arithmetic_factorization", method, arguments.alpha),
            ),
            resolved,
        )
    fabric.barrier()

    exit_code = 0
    global_step = 0
    best_transfer = {key: float("-inf") for key in ("transfer_accuracy", "paper_transfer_accuracy")}
    try:
        for epoch in range(optimization.epochs):
            metrics = _evaluate(model, test_loader, fabric)
            if metric_log is not None:
                metric_log.log(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        **{f"test/{k}": v for k, v in metrics.items()},
                    }
                )
            improved = any(metrics[key] > best_transfer[key] for key in best_transfer)
            for key in best_transfer:
                best_transfer[key] = max(best_transfer[key], metrics[key])
            if fabric.global_rank == 0 and (improved or (epoch + 1) % 5 == 0):
                torch.save(
                    {
                        "format_version": 1,
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": (
                            model.module.state_dict()
                            if hasattr(model, "module")
                            else model.state_dict()
                        ),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "config": resolved,
                    },
                    checkpoints / f"checkpoint_{global_step + 1}.pth",
                )

            model.train()
            for batch in train_loader:
                data, _ = unpack_batch(batch, fabric.device)
                output = model(data)
                fabric.backward(output.loss)
                fabric.clip_gradients(
                    model, optimizer, max_norm=optimization.max_gradient_norm
                )
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                model.quantizer.step()
                model.update_length_control(global_step, total_steps)
                if metric_log is not None and global_step % 50 == 0:
                    metric_log.log(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "train/loss": float(output.loss),
                            "train/reconstruction_loss": float(
                                output.reconstruction_loss
                            ),
                            "train/self_explanation_accuracy": (
                                output.metrics.number_accuracy
                            ),
                            "train/mean_explanation_length": float(
                                output.mean_explanation_length
                            ),
                            # These are the temperatures actually used by this
                            # forward, before quantizer.step() advanced its clock.
                            "train/action_temperature": float(output.action_vq_by_length[0].temperature),
                            "train/action_vq_loss": float(output.action_vq_loss),
                            "train/orthogonal_loss": (
                                float(output.action_vq_by_length[0].orthogonal_loss)
                                if output.action_vq_by_length[0].orthogonal_loss is not None else 0.0
                            ),
                            "length_control_coefficient": (
                                model.length_control_coefficient
                            ),
                            "learning_rate": scheduler.get_last_lr()[0],
                        }
                    )
                global_step += 1
        # Validate and preserve the state after the final optimizer update too.
        metrics = _evaluate(model, test_loader, fabric)
        if metric_log is not None:
            metric_log.log({
                "epoch": optimization.epochs,
                "global_step": global_step,
                **{f"test/{k}": v for k, v in metrics.items()},
            })
        if fabric.global_rank == 0:
            torch.save(
                {
                    "format_version": 1,
                    "epoch": optimization.epochs,
                    "global_step": global_step,
                    "model_state_dict": (
                        model.module.state_dict()
                        if hasattr(model, "module")
                        else model.state_dict()
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": resolved,
                },
                checkpoints / f"checkpoint_{global_step + 1}.pth",
            )
    except BaseException:
        exit_code = 1
        raise
    finally:
        if fabric.global_rank == 0:
            (run_directory / "status.json").write_text(
                json.dumps({
                    "status": "completed" if exit_code == 0 and global_step == total_steps else "failed",
                    "global_step": global_step,
                    "expected_steps": total_steps,
                    "completed_epochs": global_step // len(train_loader),
                    "exit_code": exit_code,
                }, indent=2) + "\n",
                encoding="utf-8",
            )
        if metric_log is not None:
            metric_log.finish(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
