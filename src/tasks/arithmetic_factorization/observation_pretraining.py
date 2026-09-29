"""Frozen contract for the arithmetic observation-pretraining stage."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class ArithmeticObservationPretrainingConfig:
    """Arithmetic observation pretraining settings.

    Pretraining fits a digit autoencoder on single-step level-1 transitions.
    Only ``encoder.*`` and ``decoder.*`` transfer downstream, so the released
    artifact is a ten-symbol embedding and its linear readout.
    """

    seed: int = 42
    epochs: int = 50
    per_rank_batch_size: int = 512
    effective_world_size: int = 4
    save_interval_epochs: int = 5
    log_interval_steps: int = 50
    learning_rate: float = 0.003
    weight_decay: float = 0.01
    max_gradient_norm: float = 1.0
    warmup_ratio: float = 0.05
    minimum_learning_rate_ratio: float = 0.1
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
    grounding_weight: float = 1.0
    reconstruction_weight: float = 0.0
    action_vq_weight: float = 0.0

    def __post_init__(self) -> None:
        positive_integers = (
            self.epochs,
            self.per_rank_batch_size,
            self.effective_world_size,
            self.save_interval_epochs,
            self.log_interval_steps,
            self.num_digits,
            self.num_symbols,
            self.state_dim,
            self.action_dim,
            self.num_state_tokens,
            self.num_action_tokens,
            self.action_codebook_size,
            self.max_transition_length,
        )
        if any(value < 1 for value in positive_integers):
            raise ValueError("arithmetic pretraining counts must be positive")
        if self.num_state_tokens != self.num_digits:
            raise ValueError("the digit encoder needs one state token per digit")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError("warmup_ratio must fall in [0, 1]")

    # The shared rollout reads the objective weights and freeze flags under
    # these names. Pretraining trains the observation model, so nothing is
    # frozen and the reconstruction term is carried by auto-reconstruction.


    @property
    def freeze_observation_encoder(self) -> bool:
        return False

    @property
    def freeze_observation_decoder(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class ArithmeticObservationExperiment:
    """Pretraining contract as the shared rollout reads it."""

    training: ArithmeticObservationPretrainingConfig

    @property
    def length_control_coefficient(self) -> float:
        return 1.0


# The downstream stage consumes this checkpoint from the pretraining run.
PAPER_OBSERVATION_CHECKPOINT = "checkpoint_171.pth"

# Only these prefixes are transferred and frozen for every downstream method.
TRANSFERRED_PREFIXES: tuple[str, ...] = ("encoder.", "decoder.")


__all__ = [
    "PAPER_OBSERVATION_CHECKPOINT",
    "TRANSFERRED_PREFIXES",
    "ArithmeticObservationExperiment",
    "ArithmeticObservationPretrainingConfig",
]


def observation_pretraining_config(profile: str = "recovered") -> ArithmeticObservationPretrainingConfig:
    """Appendix F uses direct embeddings; Table 22 VAE beta is inapplicable."""
    config = ArithmeticObservationPretrainingConfig()
    if profile == "appendix":
        return replace(config, epochs=500, minimum_learning_rate_ratio=0.005)
    if profile != "recovered":
        raise ValueError("unknown pretraining profile")
    return config
