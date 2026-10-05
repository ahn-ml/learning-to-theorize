"""Settings for the arithmetic observation-pretraining stage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArithmeticObservationPretrainingConfig:
    """Arithmetic observation pretraining settings.

    Pretraining fits a digit autoencoder by reconstruction only. Only
    ``encoder.*`` and ``decoder.*`` transfer downstream, so the observation
    model is a ten-symbol embedding and its linear readout.
    """

    seed: int = 42
    epochs: int = 500
    per_rank_batch_size: int = 512
    effective_world_size: int = 4
    log_interval_steps: int = 50
    learning_rate: float = 0.003
    weight_decay: float = 0.01
    max_gradient_norm: float = 1.0
    warmup_ratio: float = 0.05
    minimum_learning_rate_ratio: float = 0.005
    theory_programmer_learning_rate_scale: float = 0.3
    program_executor_learning_rate_scale: float = 1.0

    num_digits: int = 4
    num_symbols: int = 10
    state_dim: int = 8
    action_dim: int = 2
    num_state_tokens: int = 4
    num_action_tokens: int = 1
    action_codebook_size: int = 8
    max_transition_length: int = 1

    auto_reconstruction_weight: float = 1.0
    grounding_weight: float = 0.0
    reconstruction_weight: float = 0.0
    action_vq_weight: float = 0.0

    # The shared rollout reads the objective weights and freeze flags under
    # these names. Pretraining trains the observation model, so nothing is
    # frozen.

    @property
    def freeze_observation_encoder(self) -> bool:
        return False

    @property
    def freeze_observation_decoder(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class ArithmeticObservationExperiment:
    """Pretraining settings as the shared rollout reads them."""

    training: ArithmeticObservationPretrainingConfig

    @property
    def length_control_coefficient(self) -> float:
        return 1.0


# Only these prefixes are transferred and frozen for NEO training.
TRANSFERRED_PREFIXES: tuple[str, ...] = ("encoder.", "decoder.")


__all__ = [
    "TRANSFERRED_PREFIXES",
    "ArithmeticObservationExperiment",
    "ArithmeticObservationPretrainingConfig",
]
