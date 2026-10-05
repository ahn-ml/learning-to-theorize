"""ImageEditing model and training settings."""

from __future__ import annotations

from dataclasses import dataclass


#: Model seeds trained for every alpha.
PAPER_SEEDS: tuple[int, ...] = (42, 43, 44)

PAPER_ALPHAS: tuple[str, ...] = ("1.00", "0.66", "0.33")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """One resolved training condition."""

    # topology
    world_size: int = 2
    epochs: int = 50
    batch_size: int = 64
    num_workers: int = 4

    # latent program shape
    max_transition_length: int = 3
    state_dim: int = 256
    action_dim: int = 16
    num_state_tokens: int = 1
    num_action_tokens: int = 1
    policy_hidden_dim: int = 128
    policy_film_layers: int = 4
    transition_film_layers: int = 4
    dropout: float = 0.0

    # action space
    action_codebook_size: int = 16
    action_commitment_weight: float = 0.25
    action_entropy_weight: float = 0.1
    action_entropy_temperature: float = 1.0
    action_ema_decay: float = 0.99
    action_ema_epsilon: float = 1e-5

    # optimisation
    learning_rate: float = 1e-3
    policy_learning_rate_scale: float = 0.25
    transition_learning_rate_scale: float = 0.5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    minimum_learning_rate_ratio: float = 0.1
    gradient_clip_norm: float = 1.0

    # objective
    reconstruction_weight: float = 1.0
    grounding_weight: float = 0.5
    state_prediction_weight: float = 0.5
    action_vq_weight: float = 0.5
    auto_reconstruction_weight: float = 0.0
    adaptive_grounding: bool = True

    # MDL length control
    length_control_coefficient: float = 1.01
    length_control_scheduling: bool = False
    length_control_start: float = 1.01
    length_control_end: float = 1.01

    # frozen observation model
    freeze_observation_encoder: bool = True
    freeze_observation_decoder: bool = True


@dataclass(frozen=True, slots=True)
class Experiment:
    """One alpha condition."""

    alpha: float
    training: TrainingConfig

    @property
    def length_control_coefficient(self) -> float:
        """Satisfy ``models.config.NEOExperimentConfig``.

        The shared rollout reads the coefficient from the experiment; tasks that
        anneal it pass the current value to ``NEO.forward`` instead.
        """

        return self.training.length_control_coefficient

    def scheduled_length_control(self, progress: float) -> float:
        """Linearly annealed coefficient at ``progress`` in ``[0, 1]``.

        Returns the fixed coefficient when scheduling is off, so callers can
        pass the result unconditionally.
        """

        if not 0.0 <= progress <= 1.0:
            raise ValueError("progress must lie in [0, 1]")
        if not self.training.length_control_scheduling:
            return self.training.length_control_start
        start = self.training.length_control_start
        end = self.training.length_control_end
        return start + (end - start) * progress


def _neo(*, start: float, end: float, scheduling: bool, ema_decay: float) -> TrainingConfig:
    return TrainingConfig(
        max_transition_length=3,
        grounding_weight=0.1,
        action_codebook_size=16,
        action_ema_decay=ema_decay,
        length_control_coefficient=1.01,
        length_control_scheduling=scheduling,
        length_control_start=start,
        length_control_end=end,
    )


# alpha: (start, end, scheduling, ema_decay)
_NEO_LENGTH_CONTROL = {
    "1.00": (1.01, 1.01, False, 0.99),
    "0.66": (1.05, 1.00, True, 0.99),
    "0.33": (1.01, 1.00, True, 0.99),
}

_EXPERIMENTS: dict[str, Experiment] = {}
for _alpha in PAPER_ALPHAS:
    _start, _end, _sched, _decay = _NEO_LENGTH_CONTROL[_alpha]
    _EXPERIMENTS[_alpha] = Experiment(
        alpha=float(_alpha),
        training=_neo(start=_start, end=_end, scheduling=_sched, ema_decay=_decay),
    )


def get_experiment(alpha: str) -> Experiment:
    """Look up one resolved training condition."""

    try:
        return _EXPERIMENTS[alpha]
    except KeyError as error:
        raise ValueError(
            f"unknown Image Editing alpha {alpha!r}; choose one of: {', '.join(PAPER_ALPHAS)}"
        ) from error


__all__ = [
    "Experiment",
    "PAPER_ALPHAS",
    "PAPER_SEEDS",
    "TrainingConfig",
    "get_experiment",
]
