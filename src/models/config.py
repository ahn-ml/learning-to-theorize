"""Structural configuration contracts for shared latent-program models."""

from __future__ import annotations

from typing import Protocol


class LatentProgramConfig(Protocol):
    """Values required by the shared theory-programming components.

    Task experiment configs satisfy this protocol structurally.  The shared
    model package therefore has no dependency on a task-specific config class.
    """

    state_dim: int
    action_dim: int
    num_state_tokens: int
    num_action_tokens: int
    policy_hidden_dim: int
    transition_hidden_dim: int
    transition_residual: bool
    dropout: float
    max_transition_length: int

    action_codebook_size: int
    action_commitment_weight: float
    stochastic_action_vq: bool
    action_tau_start: float
    action_tau_end: float

    @property
    def action_tau_steps(self) -> int: ...


class NEOTrainingConfig(LatentProgramConfig, Protocol):
    """Additional settings consumed by the task-independent NEO rollout."""

    reconstruction_weight: float
    grounding_weight: float
    auto_reconstruction_weight: float
    action_vq_weight: float
    freeze_observation_encoder: bool
    freeze_observation_decoder: bool


class NEOExperimentConfig(Protocol):
    """Resolved experiment contract consumed by :class:`models.neo.NEO`."""

    training: NEOTrainingConfig
    length_control_coefficient: float


__all__ = [
    "LatentProgramConfig",
    "NEOExperimentConfig",
    "NEOTrainingConfig",
]
