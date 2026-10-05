"""Frozen paper contracts for the arithmetic factorization experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

ARITHMETIC_SEEDS: tuple[int, ...] = (42, 43, 44)
ARITHMETIC_ALPHAS: tuple[str, ...] = ("0.33", "0.66", "1.00")

# Alpha selects how much of the composition space stays in the training split.
_ALPHA_TOTAL_STEPS: dict[str, int] = {"0.33": 57400, "0.66": 80800, "1.00": 109400}


@dataclass(frozen=True, slots=True)
class TheoryProgrammerConfig:
    """Transformer that proposes one latent operation per rollout step."""

    model_dim: int = 64
    feedforward_dim: int = 256
    num_heads: int = 4
    num_layers: int = 6
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class ProgramExecutorConfig:
    """Cross-attention interpreter that applies one operation to a state."""

    model_dim: int = 32
    feedforward_dim: int = 128
    num_heads: int = 2
    num_layers: int = 4
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class ActionQuantizerConfig:
    """Discrete operation bottleneck for NEO.

    Training samples codes from softmax(-distance / tau), with tau annealed
    from ``tau_start`` to ``tau_end`` over the first ``scheduling_ratio`` of
    optimizer steps; evaluation uses the nearest code. The EMA codebook is
    updated in every quantizer call from that call's assignments, so later
    rollout steps see it.
    """

    commitment_weight: float = 0.25
    use_exponential_moving_average: bool = True
    exponential_moving_average_decay: float = 0.99
    tau_start: float = 0.3
    tau_end: float = 0.05
    scheduling_ratio: float = 0.25


@dataclass(frozen=True, slots=True)
class LossWeights:
    """Objective term weights held fixed across every paper run."""

    reconstruction: float = 1.0
    auto_reconstruction: float = 0.0
    action_vq: float = 1.0
    grounding: float = 0.5


@dataclass(frozen=True, slots=True)
class OptimizationSettings:
    """Optimizer and schedule values held fixed across every paper run."""

    learning_rate: float = 0.0015
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    minimum_learning_rate_ratio: float = 0.1
    max_gradient_norm: float = 1.0
    batch_size: int = 512
    epochs: int = 200
    theory_programmer_learning_rate_scale: float = 0.5
    program_executor_learning_rate_scale: float = 1.0
    precision: str = "bf16-mixed"


@dataclass(frozen=True, slots=True)
class ArithmeticTrainingConfig:
    """Executable contract for one arithmetic factorization training run."""

    alpha: str
    total_steps: int
    max_transition_length: int
    action_codebook_size: int

    num_digits: int = 6
    num_symbols: int = 10
    state_dim: int = 8
    action_dim: int = 4
    num_state_tokens: int = 6
    num_action_tokens: int = 1

    length_control_coefficient: float = 1.01
    length_control_coefficient_end: float = 0.99
    length_control_scheduling_ratio: float = 0.1

    theory_programmer: TheoryProgrammerConfig = field(
        default_factory=TheoryProgrammerConfig
    )
    program_executor: ProgramExecutorConfig = field(
        default_factory=ProgramExecutorConfig
    )
    quantizer: ActionQuantizerConfig = field(default_factory=ActionQuantizerConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    optimization: OptimizationSettings = field(default_factory=OptimizationSettings)

    def __post_init__(self) -> None:
        if self.alpha not in ARITHMETIC_ALPHAS:
            raise ValueError(f"unknown arithmetic alpha: {self.alpha!r}")
        if self.max_transition_length < 1:
            raise ValueError("max_transition_length must be positive")
        if self.action_codebook_size < 1:
            raise ValueError("action_codebook_size must be positive")

    @property
    def action_tau_steps(self) -> int:
        """Number of steps over which the quantizer temperature anneals."""

        return int(self.total_steps * self.quantizer.scheduling_ratio)

    # The shared rollout reads the objective weights and freeze flags under
    # these names. The pretrained observation model is always frozen.

    @property
    def reconstruction_weight(self) -> float:
        return self.loss_weights.reconstruction

    @property
    def auto_reconstruction_weight(self) -> float:
        return self.loss_weights.auto_reconstruction

    @property
    def grounding_weight(self) -> float:
        return self.loss_weights.grounding

    @property
    def action_vq_weight(self) -> float:
        """Weight on the discrete action bottleneck loss."""

        return self.loss_weights.action_vq

    @property
    def freeze_observation_encoder(self) -> bool:
        return True

    @property
    def freeze_observation_decoder(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class ArithmeticExperiment:
    """One resolved arithmetic experiment as the shared rollout reads it."""

    training: ArithmeticTrainingConfig

    @property
    def length_control_coefficient(self) -> float:
        return self.training.length_control_coefficient


def get_experiment(method: str, alpha: str) -> ArithmeticTrainingConfig:
    """Resolve the NEO training contract for one alpha setting."""

    if method != "neo":
        raise ValueError(f"unknown arithmetic method: {method!r}")
    if alpha not in ARITHMETIC_ALPHAS:
        raise ValueError(f"unknown arithmetic alpha: {alpha!r}")
    return ArithmeticTrainingConfig(
        alpha=alpha,
        total_steps=_ALPHA_TOTAL_STEPS[alpha],
        max_transition_length=3,
        action_codebook_size=16,
    )


__all__ = [
    "ARITHMETIC_ALPHAS",
    "ARITHMETIC_SEEDS",
    "ActionQuantizerConfig",
    "ArithmeticExperiment",
    "ArithmeticTrainingConfig",
    "LossWeights",
    "OptimizationSettings",
    "ProgramExecutorConfig",
    "TheoryProgrammerConfig",
    "get_experiment",
]
