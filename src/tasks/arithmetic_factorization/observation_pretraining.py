"""Settings for learning digit embeddings from individual numbers."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArithmeticObservationPretrainingConfig:
    seed: int = 42
    epochs: int = 500
    per_rank_batch_size: int = 512
    effective_world_size: int = 4
    save_interval_epochs: int = 5
    log_interval_steps: int = 50
    learning_rate: float = 0.003
    weight_decay: float = 0.01
    max_gradient_norm: float = 1.0
    warmup_ratio: float = 0.05
    minimum_learning_rate_ratio: float = 0.005
    num_digits: int = 4
    num_symbols: int = 10
    state_dim: int = 8


TRANSFERRED_PREFIXES: tuple[str, ...] = ("encoder.", "decoder.")


def observation_pretraining_config(profile: str = "appendix", *, seed: int = 42) -> ArithmeticObservationPretrainingConfig:
    if profile != "appendix":
        raise ValueError("unknown pretraining profile")
    return ArithmeticObservationPretrainingConfig(seed=seed)
