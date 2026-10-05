"""Arithmetic model composition for shared Learning-to-Theorize methods."""

from __future__ import annotations

from typing import Any

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
from tasks.arithmetic_factorization.models.policy_network import PolicyNetwork
from tasks.arithmetic_factorization.models.quantizer import VectorQuantizer
from tasks.arithmetic_factorization.models.transition_network import (
    CrossAttentionTransition,
    TransitionNetwork,
)
from tasks.arithmetic_factorization.objective import (
    ArithmeticAccuracy,
    ArithmeticObjective,
)
from tasks.arithmetic_factorization.observation_pretraining import (
    ArithmeticObservationExperiment,
    ArithmeticObservationPretrainingConfig,
)


def module_parameters(
    training: ArithmeticTrainingConfig,
) -> DictConfig:
    """Build the parameter tree consumed by the Arithmetic model components."""

    quantizer = training.quantizer
    action: dict[str, Any] = {
        "commitment_cost": quantizer.commitment_weight,
        "use_ema": quantizer.use_exponential_moving_average,
        "ema_decay": quantizer.exponential_moving_average_decay,
        "tau_start": quantizer.tau_start,
        "tau_end": quantizer.tau_end,
        # Counted in optimizer steps, not rollout calls.
        "tau_steps": training.action_tau_steps,
    }

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
        },
        "transition": {
            "type": "cross",
            "d_model": executor.model_dim,
            "d_ff": executor.feedforward_dim,
            "num_heads": executor.num_heads,
            "num_layers": executor.num_layers,
            "dropout": executor.dropout,
        },
        "vq": {"action": action},
    }
    return OmegaConf.create(parameters)


def build_modules(parameters: DictConfig) -> dict[str, Any]:
    """Construct the model components in a fixed order.

    Order determines RNG consumption, so encoder, decoder, programmer,
    executor, and bottleneck are built in exactly that sequence. NEO uses the
    cross-attention executor; observation pretraining uses the self-attention
    one.
    """

    action = parameters.vq.action
    modules: dict[str, Any] = {
        "encoder": ObservationEncoder(parameters),
        "decoder": ObservationDecoder(parameters),
        "theory_programmer": PolicyNetwork(parameters),
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
            tau_start=action.tau_start,
            tau_end=action.tau_end,
            tau_steps=action.tau_steps,
            use_ema=action.use_ema,
            ema_decay=action.ema_decay,
        )
    )
    return modules


class ArithmeticRollout(NEO[ArithmeticAccuracy]):
    """Shared NEO with the arithmetic grounding and length-penalty schedule."""

    def __init__(self, experiment: Any, parameters: DictConfig) -> None:
        super().__init__(
            experiment,
            objective=ArithmeticObjective(),
            **build_modules(parameters),
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
        """Sum each episode's grounding distance over all rollout steps.

        Each episode's sum is divided by its selected program length before
        averaging over the batch, as in the paper's objective.
        """

        weighted = per_step_distances / selected_lengths.unsqueeze(0)
        return weighted.sum(dim=0).mean()


class ArithmeticNEO(ArithmeticRollout):
    """The NEO theory model for one alpha setting."""

    def __init__(self, training: ArithmeticTrainingConfig) -> None:
        super().__init__(ArithmeticExperiment(training), module_parameters(training))


class ArithmeticObservationModel(ArithmeticRollout):
    """Rollout model whose digit autoencoder is trained during pretraining.

    The objective is digit reconstruction only. The programmer, executor and
    quantizer still run because they fix initialization and random-number
    consumption; only ``encoder.*`` and ``decoder.*`` transfer to NEO.
    """

    def __init__(
        self,
        parameters: DictConfig,
        config: ArithmeticObservationPretrainingConfig,
    ) -> None:
        super().__init__(ArithmeticObservationExperiment(config), parameters)


def build_neo(method: str, alpha: str) -> ArithmeticNEO:
    """Build the NEO model for one alpha setting."""

    return ArithmeticNEO(get_experiment(method, alpha))


__all__ = [
    "ArithmeticNEO",
    "ArithmeticObservationModel",
    "ArithmeticRollout",
    "build_modules",
    "build_neo",
    "module_parameters",
]
