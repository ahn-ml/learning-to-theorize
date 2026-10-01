"""Arithmetic model composition for shared Learning-to-Theorize methods."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from models.neo import NEO
from tasks.arithmetic_factorization.experiment import (
    ArithmeticExperiment,
    ArithmeticTrainingConfig,
    get_experiment,
)
from tasks.arithmetic_factorization.models.adapters import (
    ArithmeticActionQuantizer,
    ObservationDecoder,
    ObservationEncoder,
)
from tasks.arithmetic_factorization.models.policy_network import IPDPolicyNetwork
from tasks.arithmetic_factorization.models.quantizer import VectorQuantizer
from tasks.arithmetic_factorization.models.transition_network import (
    CrossAttentionTransition,
    TransitionNetwork,
)
from tasks.arithmetic_factorization.objective import (
    ArithmeticAccuracy,
    ArithmeticObjective,
)

# Released checkpoints use the pre-integration module names.
_RELEASED_PREFIX_MAP: tuple[tuple[str, str], ...] = (
    ("encoder.embedding.", "encoder.embedding_encoder.embedding."),
    ("decoder.output_layer.", "decoder.digit_decoder.output_layer."),
    ("policy.", "theory_programmer."),
    ("transition.", "program_executor."),
    ("quantizer.", "quantizer.quantizer."),
)


def module_parameters(
    training: ArithmeticTrainingConfig,
) -> DictConfig:
    """Build the parameter tree consumed by the Arithmetic model components."""

    quantizer = training.quantizer
    action: dict[str, Any] = {
        "commitment_cost": quantizer.commitment_weight,
        "entropy_loss_weight": quantizer.entropy_loss_weight,
        "entropy_temperature": quantizer.entropy_temperature,
        "use_ema": quantizer.use_exponential_moving_average,
        "ema_decay": quantizer.exponential_moving_average_decay,
        "stochastic": quantizer.stochastic,
        "tau_start": quantizer.tau_start,
        "tau_end": quantizer.tau_end,
        "orthogonal_reg_weight": quantizer.orthogonal_regularization_weight,
        "orthogonal_reg_max_codes": quantizer.orthogonal_regularization_max_codes,
    }
    # Use optimizer steps, not rollout calls, for the published 25% schedule.
    if quantizer.stochastic:
        action["tau_steps"] = training.action_tau_steps

    programmer = training.theory_programmer
    executor = training.program_executor

    parameters: dict[str, Any] = {
        "codebook_size": training.action_codebook_size,
        "grid_dim": training.num_digits,
        "num_colors": training.num_symbols,
        "state_dim": training.state_dim,
        "action_dim": training.action_dim,
        "num_state_tokens": training.num_state_tokens,
        "num_action_tokens": training.num_action_tokens,
        "length_control_coeff": training.length_control_coefficient,
        "length_control_coeff_end": training.length_control_coefficient_end,
        "length_control_scheduling_ratio": training.length_control_scheduling_ratio,
        "policy": {
            "d_model": programmer.model_dim,
            "d_ff": programmer.feedforward_dim,
            "num_heads": programmer.num_heads,
            "num_layers": programmer.num_layers,
            "dropout": programmer.dropout,
            "use_sinusoidal_state_pos": programmer.sinusoidal_state_positions,
        },
        "transition": {
            "type": "cross",
            "d_model": executor.model_dim,
            "d_ff": executor.feedforward_dim,
            "num_heads": executor.num_heads,
            "num_layers": executor.num_layers,
            "dropout": executor.dropout,
            "use_residual": executor.residual,
            "use_sinusoidal_state_pos": executor.sinusoidal_state_positions,
        },
        "vq": {"action": action},
    }
    return OmegaConf.create(parameters)


def build_released_modules(parameters: DictConfig) -> dict[str, Any]:
    """Construct the released modules in the paper's initialization order.

    Order determines RNG consumption, so encoder, decoder, programmer,
    executor, and bottleneck are built in exactly that sequence.
    """

    action = parameters.vq.action
    modules: dict[str, Any] = {
        "encoder": ObservationEncoder(parameters),
        "decoder": ObservationDecoder(parameters),
        "theory_programmer": IPDPolicyNetwork(parameters),
    }
    modules["program_executor"] = (
        CrossAttentionTransition(parameters)
        if parameters.transition.type == "cross"
        else TransitionNetwork(parameters)
    )
    modules["quantizer"] = ArithmeticActionQuantizer(
        VectorQuantizer(
            codebook_size=parameters.codebook_size,
            embedding_dim=parameters.action_dim,
            commitment_cost=action.commitment_cost,
            stochastic=action.stochastic,
            tau_start=action.tau_start,
            tau_end=action.tau_end,
            tau_steps=action.get("tau_steps", 10000),
            entropy_loss_weight=action.entropy_loss_weight,
            entropy_temperature=action.entropy_temperature,
            use_ema=action.use_ema,
            ema_decay=action.ema_decay,
            orthogonal_reg_weight=action.orthogonal_reg_weight,
            orthogonal_reg_max_codes=action.orthogonal_reg_max_codes,
        )
    )
    return modules


class ArithmeticRollout(NEO[ArithmeticAccuracy]):
    """Shared NEO with the arithmetic grounding and length-penalty schedule."""

    def __init__(self, experiment: Any, parameters: DictConfig) -> None:
        super().__init__(
            experiment,
            objective=ArithmeticObjective(),
            **build_released_modules(parameters),
        )
        self._length_control_start = float(parameters.length_control_coeff)
        self._length_control_end = float(
            parameters.get("length_control_coeff_end", parameters.length_control_coeff)
        )
        self._length_control_ratio = float(
            parameters.get("length_control_scheduling_ratio", 1.0)
        )
        self._length_control = self._length_control_start

    @property
    def length_control_coefficient(self) -> float:
        return self._length_control

    def update_length_control(self, global_step: int, total_steps: int) -> float:
        """Anneal the length penalty over the opening fraction of training."""

        if self._length_control_start == self._length_control_end:
            return self._length_control
        end_step = int(total_steps * self._length_control_ratio)
        if end_step <= 0 or global_step >= end_step:
            self._length_control = self._length_control_end
        else:
            progress = global_step / end_step
            self._length_control = self._length_control_start + progress * (
                self._length_control_end - self._length_control_start
            )
        return self._length_control

    def reduce_consistency_loss(
        self,
        per_step_distances: Tensor,
        selected_lengths: Tensor,
    ) -> Tensor:
        """Sum grounding across the rollout, independent of selected length.

        Average over the batch, then sum the per-step grounding distances.
        """

        return per_step_distances.mean(dim=1).sum()


class ArithmeticNEO(ArithmeticRollout):
    """The paper's theory model for one method and alpha setting."""

    def __init__(
        self,
        experiment: str | ArithmeticTrainingConfig,
        alpha: str | None = None,
    ) -> None:
        training = (
            get_experiment(experiment, alpha if alpha is not None else "1.00")
            if isinstance(experiment, str)
            else experiment
        )
        super().__init__(ArithmeticExperiment(training), module_parameters(training))


def released_state_dict(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Rename a released checkpoint onto the integrated module names.

    Already-integrated checkpoints are returned unchanged, so the mapping is
    safe to apply to either format.
    """

    if any(key.startswith("theory_programmer.") for key in state):
        return dict(state)
    renamed: dict[str, Tensor] = {}
    for key, value in state.items():
        for released, integrated in _RELEASED_PREFIX_MAP:
            if key.startswith(released):
                key = integrated + key[len(released) :]
                break
        renamed[key] = value
    return renamed


def build_neo(method: str, alpha: str) -> ArithmeticNEO:
    """Build the paper model for one method and alpha setting."""

    return ArithmeticNEO(get_experiment(method, alpha))


__all__ = [
    "ArithmeticNEO",
    "ArithmeticRollout",
    "build_released_modules",
    "build_neo",
    "module_parameters",
    "released_state_dict",
]
