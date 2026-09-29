"""GridWorld NEO model and training settings."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from tasks.gridworld.data.artifacts import get_artifact_spec
from tasks.gridworld.data.profiles import SplitName, get_profile


GRIDWORLD_LENGTH_OOD_ARTIFACT_SHA256 = get_artifact_spec(
    "paper-length-4-8", "exam"
).sha256
GRIDWORLD_THEORIZER_SEEDS = (42, 43, 44)


@dataclass(frozen=True, slots=True)
class GridWorldTheorizerTrainingConfig:
    """Shared settings of one paper GridWorld theorizer run."""

    world_size: int = 1
    epochs: int = 50
    train_episodes: int = 50_000
    test_episodes: int = 5_000
    batch_size: int = 128
    num_workers: int = 4
    max_transition_length: int = 4

    state_dim: int = 32
    action_dim: int = 16
    num_state_tokens: int = 1
    num_action_tokens: int = 1
    policy_hidden_dim: int = 128
    transition_hidden_dim: int = 128
    transition_residual: bool = True
    dropout: float = 0.0

    num_action_codebooks: int = 1
    action_codebook_size: int = 6
    action_commitment_weight: float = 0.25
    stochastic_action_vq: bool = True
    action_tau_start: float = 0.3
    action_tau_end: float = 0.1
    action_tau_scheduling_ratio: float = 0.8

    learning_rate: float = 5e-4
    policy_learning_rate_scale: float = 0.25
    transition_learning_rate_scale: float = 1.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    minimum_learning_rate_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    precision: str = "bf16-mixed"

    reconstruction_weight: float = 1.0
    action_vq_weight: float = 1.0
    grounding_weight: float = 0.1
    auto_reconstruction_weight: float = 0.0

    evaluate_before_training: bool = True
    checkpoint_interval_epochs: int = 5
    log_interval_steps: int = 50
    wandb_project: str = "NeuralProgrammer"
    freeze_observation_encoder: bool = True
    freeze_observation_decoder: bool = True
    deterministic_observation_vae: bool = True

    def __post_init__(self) -> None:
        if self.world_size != 1:
            raise ValueError("the paper alpha sweep used one visible GPU per run")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.train_episodes < 1 or self.test_episodes < 1:
            raise ValueError("episode counts must be positive")
        if self.checkpoint_interval_epochs < 1:
            raise ValueError("checkpoint_interval_epochs must be positive")

    @property
    def train_batches_per_epoch(self) -> int:
        """Number of optimizer steps per epoch."""

        return ceil(self.train_episodes / self.batch_size)

    @property
    def test_batches_per_epoch(self) -> int:
        """Number of evaluation batches per epoch."""

        return ceil(self.test_episodes / self.batch_size)

    @property
    def total_steps(self) -> int:
        return self.epochs * self.train_batches_per_epoch

    @property
    def warmup_steps(self) -> int:
        return int(self.total_steps * self.warmup_ratio)

    @property
    def action_tau_steps(self) -> int:
        return int(self.total_steps * self.action_tau_scheduling_ratio)

    @property
    def policy_learning_rate(self) -> float:
        return self.learning_rate * self.policy_learning_rate_scale

    @property
    def transition_learning_rate(self) -> float:
        return self.learning_rate * self.transition_learning_rate_scale

    def evaluation_global_step(self, epoch_index: int) -> int:
        """Global step at the pre-training evaluation for a zero-based epoch."""

        if not 0 <= epoch_index < self.epochs:
            raise ValueError(f"epoch_index must be in [0, {self.epochs})")
        return epoch_index * self.train_batches_per_epoch

    def checkpoint_step(self, epoch_number: int) -> int:
        """Periodic-checkpoint suffix for a one-based epoch number."""

        if not 1 <= epoch_number <= self.epochs:
            raise ValueError(f"epoch_number must be in [1, {self.epochs}]")
        if epoch_number % self.checkpoint_interval_epochs:
            raise ValueError(
                f"epoch_number must be divisible by {self.checkpoint_interval_epochs}"
            )
        return self.evaluation_global_step(epoch_number - 1) + 1


@dataclass(frozen=True, slots=True)
class GridWorldAlphaExperiment:
    """One alpha setting and its three model-seed runs."""

    name: str
    alpha: float
    length_control_coefficient: float
    data_profile: str
    train_artifact_sha256: str
    test_artifact_sha256: str
    compositional_ood_artifact_sha256: str | None
    training: GridWorldTheorizerTrainingConfig

    def __post_init__(self) -> None:
        profile = get_profile(self.data_profile)
        if profile.alpha != self.alpha:
            raise ValueError("experiment alpha does not match its data profile")
        digests = (
            self.train_artifact_sha256,
            self.test_artifact_sha256,
            *(
                (self.compositional_ood_artifact_sha256,)
                if self.compositional_ood_artifact_sha256 is not None
                else ()
            ),
        )
        if any(len(digest) != 64 for digest in digests):
            raise ValueError("artifact SHA-256 digests must contain 64 hexadecimal characters")
        try:
            for digest in digests:
                int(digest, 16)
        except ValueError as error:
            raise ValueError("artifact SHA-256 digests must be hexadecimal") from error

    @property
    def profile(self):
        """Resolved deterministic data profile for this experiment."""

        return get_profile(self.data_profile)


_TRAINING_50_EPOCHS = GridWorldTheorizerTrainingConfig()
_TRAINING_ALPHA033 = GridWorldTheorizerTrainingConfig(
    epochs=100,
    warmup_ratio=0.10,
    action_tau_scheduling_ratio=0.5,
)


def _artifact_sha256(profile: str, split: SplitName) -> str:
    return get_artifact_spec(profile, split).sha256


_EXPERIMENTS = {
    "alpha-0.33": GridWorldAlphaExperiment(
        name="alpha-0.33",
        alpha=0.33,
        length_control_coefficient=0.95,
        data_profile="paper-alpha-0.33",
        train_artifact_sha256=_artifact_sha256("paper-alpha-0.33", "practice"),
        test_artifact_sha256=_artifact_sha256("paper-alpha-0.33", "exam"),
        compositional_ood_artifact_sha256=_artifact_sha256(
            "paper-alpha-0.33", "ood_test"
        ),
        training=_TRAINING_ALPHA033,
    ),
    "alpha-0.66": GridWorldAlphaExperiment(
        name="alpha-0.66",
        alpha=0.66,
        length_control_coefficient=0.95,
        data_profile="paper-alpha-0.66",
        train_artifact_sha256=_artifact_sha256("paper-alpha-0.66", "practice"),
        test_artifact_sha256=_artifact_sha256("paper-alpha-0.66", "exam"),
        compositional_ood_artifact_sha256=_artifact_sha256(
            "paper-alpha-0.66", "ood_test"
        ),
        training=_TRAINING_50_EPOCHS,
    ),
    "alpha-1.00": GridWorldAlphaExperiment(
        name="alpha-1.00",
        alpha=1.0,
        length_control_coefficient=1.0,
        data_profile="paper-alpha-1.00",
        train_artifact_sha256=_artifact_sha256("paper-alpha-1.00", "practice"),
        test_artifact_sha256=_artifact_sha256("paper-alpha-1.00", "exam"),
        compositional_ood_artifact_sha256=None,
        training=_TRAINING_50_EPOCHS,
    ),
}


def available_theorizer_experiments() -> tuple[str, ...]:
    """Return stable names for the three paper alpha settings."""

    return tuple(_EXPERIMENTS)


def get_theorizer_experiment(name: str) -> GridWorldAlphaExperiment:
    """Return a fully resolved paper GridWorld theorizer experiment."""

    try:
        return _EXPERIMENTS[name]
    except KeyError as error:
        choices = ", ".join(available_theorizer_experiments())
        raise ValueError(f"unknown GridWorld experiment {name!r}; choose one of: {choices}") from error
