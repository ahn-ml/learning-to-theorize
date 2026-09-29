"""Pretrain the shared digit autoencoder consumed by every arithmetic method."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import torch
from omegaconf import DictConfig, OmegaConf

from tasks.arithmetic_factorization.observation_pretraining import (
    ArithmeticObservationPretrainingConfig,
    observation_pretraining_config,
)
from training.optimization import (
    OptimizationConfig,
    build_cosine_warmup_scheduler,
    build_program_optimizer,
)

RUN_NAME = "arithmetic-observation-pretraining"


def pretraining_parameters(
    config: ArithmeticObservationPretrainingConfig,
    *,
    steps_per_epoch: int,
) -> DictConfig:
    """Build the released parameter tree for the pretraining stage."""

    total_steps = config.epochs * steps_per_epoch
    return OmegaConf.create(
        {
            "max_transition_length": config.max_transition_length,
            "codebook_size": config.action_codebook_size,
            "use_one_codebook": True,
            "use_vae": False,
            "vae_deterministic": True,
            "grid_dim": config.num_digits,
            "num_colors": config.num_symbols,
            "num_train_pairs": 1,
            "state_dim": config.state_dim,
            "action_dim": config.action_dim,
            "num_state_tokens": config.num_state_tokens,
            "num_action_tokens": config.num_action_tokens,
            "do_auto_recon": True,
            "length_control_coeff": 1.0,
            "encoder": {"type": "digit", "dropout": 0.0, "d_model": 32},
            "decoder": {"type": "digit", "dropout": 0.0, "d_model": 32},
            "policy": {
                "type": "transformer", "d_model": 32, "d_ff": 32,
                "num_heads": 2, "num_layers": 4, "dropout": 0.0,
                "use_sinusoidal_state_pos": False,
            },
            "transition": {
                "type": "transformer", "d_model": 32, "d_ff": 32,
                "num_heads": 2, "num_layers": 4, "dropout": 0.0,
                "use_residual": False,
                "use_sinusoidal_state_pos": False,
            },
            "vq": {
                "action": {
                    "codebook_size": 6,
                    "commitment_cost": 0.25,
                    "entropy_loss_weight": 0.0,
                    "entropy_temperature": 1.0,
                    "diversity_loss_weight": 0.0,
                    "use_ema": False,
                    "ema_decay": 0.99,
                    "orthogonal_reg_weight": 0.0,
                    "orthogonal_reg_max_codes": None,
                    "stochastic": True,
                    "tau_start": 0.3,
                    "tau_end": 0.05,
                    "scheduling_ratio": 0.25,
                    "tau_steps": int(total_steps * 0.25),
                }
            },
            "loss_weights": {
                "recon_loss": config.reconstruction_weight,
                "auto_recon_loss": config.auto_reconstruction_weight,
                "action_vq_loss": config.action_vq_weight,
                "grounding_loss": config.grounding_weight,
                "action_vae_beta": 1.0,
            },
            "total_steps": total_steps,
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretraining-profile", choices=("recovered", "appendix"), default="recovered")
    parser.add_argument("--train-h5", required=True, type=Path)
    parser.add_argument("--test-h5", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--wandb-project", default="LearningToTheorize")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    arguments = parser.parse_args(argv)

    from lightning.fabric import Fabric
    from lightning.fabric.strategies import DDPStrategy

    from tasks.arithmetic_factorization.data.dataset import build_dataloader, unpack_batch
    from tasks.arithmetic_factorization.task import ArithmeticObservationModel
    from tasks.arithmetic_factorization.theorizer_runner import _evaluate
    from training.runtime import (
        ExperimentTrackingConfig,
        MetricLog,
        capture_git_source,
        seed_everything,
    )

    config = observation_pretraining_config(arguments.pretraining_profile)
    source = capture_git_source(Path(__file__).resolve().parent)
    fabric = Fabric(
        accelerator="auto",
        devices=config.effective_world_size,
        precision="bf16-mixed",
        strategy=DDPStrategy(
            find_unused_parameters=True,
            timeout=timedelta(minutes=30),
            process_group_backend="nccl",
        ),
    )
    fabric.launch()
    seed_everything(config.seed)
    # Fabric's injected DistributedSampler reads PL_GLOBAL_SEED, not the
    # DataLoader generator. Set it before constructing the loaders.
    fabric.seed_everything(config.seed)

    train_loader = fabric.setup_dataloaders(
        build_dataloader(
            arguments.train_h5, batch_size=config.per_rank_batch_size,
            shuffle=True, seed=config.seed,
        )
    )
    test_loader = fabric.setup_dataloaders(
        build_dataloader(
            arguments.test_h5, batch_size=config.per_rank_batch_size,
            shuffle=False, seed=config.seed,
        )
    )

    parameters = pretraining_parameters(config, steps_per_epoch=len(train_loader))
    total_steps = int(parameters.total_steps)
    from tasks.arithmetic_factorization.fabric_output import fabric_output_hook
    model = ArithmeticObservationModel(parameters, config=config)
    model.register_forward_hook(fabric_output_hook)

    optimizer = build_program_optimizer(
        model,
        learning_rate=config.learning_rate,
        programmer_learning_rate=(
            config.learning_rate * config.theory_programmer_learning_rate_scale
        ),
        executor_learning_rate=(
            config.learning_rate * config.program_executor_learning_rate_scale
        ),
        weight_decay=config.weight_decay,
    )
    model, optimizer = fabric.setup(model, optimizer)
    scheduler = build_cosine_warmup_scheduler(
        optimizer,
        OptimizationConfig(
            total_steps=total_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            max_gradient_norm=config.max_gradient_norm,
            warmup_ratio=config.warmup_ratio,
            minimum_learning_rate_ratio=config.minimum_learning_rate_ratio,
        ),
    )

    source = capture_git_source(Path(__file__).resolve().parent)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S.%fZ")
    run_directory = arguments.output_root.expanduser().resolve() / RUN_NAME / stamp
    checkpoints = run_directory / "checkpoints"
    resolved: dict[str, Any] = {
        "schema_version": 1,
        "domain": "arithmetic_factorization",
        "stage": "pretraining",
        "canonical": True,
        "pretraining_profile": arguments.pretraining_profile,
        "observation_architecture": "digit-embedding-linear",
        "use_vae": False,
        "parameters": asdict(config),
        "artifacts": {
            "train_h5": str(arguments.train_h5),
            "test_h5": str(arguments.test_h5),
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
        OmegaConf.save(parameters, run_directory / "config.yaml")
        metric_log = MetricLog(
            run_directory / "metrics.jsonl",
            ExperimentTrackingConfig(
                project=arguments.wandb_project,
                name=f"{RUN_NAME}-{stamp}",
                entity=arguments.wandb_entity,
                group=arguments.wandb_group or RUN_NAME,
                mode=arguments.wandb_mode,
                tags=("arithmetic_factorization", "observation-pretraining"),
            ),
            resolved,
        )
    fabric.barrier()

    exit_code = 0
    global_step = 0
    try:
        for epoch in range(config.epochs):
            metrics = _evaluate(model, test_loader, fabric)
            if metric_log is not None:
                metric_log.log(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        **{f"test/{k}": v for k, v in metrics.items()},
                    }
                )
            if (
                fabric.global_rank == 0
                and (epoch + 1) % config.save_interval_epochs == 0
            ):
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
                if not torch.isfinite(output.loss):
                    raise FloatingPointError("nonfinite arithmetic pretraining loss")
                fabric.backward(output.loss)
                fabric.clip_gradients(
                    model, optimizer, max_norm=config.max_gradient_norm
                )
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                model.quantizer.step()
                if metric_log is not None and global_step % config.log_interval_steps == 0:
                    metric_log.log(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "train/loss": float(output.loss),
                            "train/auto_reconstruction_loss": float(
                                output.auto_reconstruction_loss
                            ),
                            "learning_rate": scheduler.get_last_lr()[0],
                        }
                    )
                global_step += 1
            if fabric.global_rank == 0:
                (run_directory / "status.json").write_text(json.dumps({
                    "status": "running", "completed_epochs": epoch + 1,
                    "global_step": global_step, "expected_steps": total_steps,
                }))
        if fabric.global_rank == 0:
            final_checkpoint = checkpoints / "checkpoint_final.pth"
            torch.save({
                "format_version": 1, "epoch": config.epochs,
                "global_step": global_step,
                "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "config": resolved,
            }, final_checkpoint)
            (run_directory / "status.json").write_text(json.dumps({
                "status": "completed", "completed_epochs": config.epochs,
                "global_step": global_step, "expected_steps": total_steps,
                "final_checkpoint": str(final_checkpoint),
            }))
        fabric.barrier()
    except Exception:
        exit_code = 1
        raise
    finally:
        if metric_log is not None:
            metric_log.finish(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
