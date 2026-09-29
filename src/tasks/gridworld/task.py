"""GridWorld model composition for shared Learning-to-Theorize methods."""

from __future__ import annotations

from models.config import NEOExperimentConfig
from models.neo import NEO
from tasks.gridworld.models.vae import VAE, VAEConfig
from tasks.gridworld.objective import GridWorldAccuracy, GridWorldObjective
from tasks.gridworld.theorizer_config import (
    GridWorldAlphaExperiment,
    get_theorizer_experiment,
)


class GridWorldNEO(NEO[GridWorldAccuracy]):
    """Shared NEO composed with the GridWorld VAE and objective."""

    def __init__(self, experiment: str | NEOExperimentConfig) -> None:
        resolved = (
            get_theorizer_experiment(experiment)
            if isinstance(experiment, str)
            else experiment
        )
        training = resolved.training
        vae = VAE(
            VAEConfig(
                grid_size=10,
                num_colors=9,
                state_dim=training.state_dim,
                num_state_tokens=training.num_state_tokens,
                dropout=training.dropout,
                variational=True,
                sample_posterior=not training.deterministic_observation_vae,
            )
        )
        # Preserve the paper initialization order: encoder, decoder,
        # programmer, executor, then quantizer. NEO constructs the last three.
        super().__init__(
            resolved,
            encoder=vae.encoder,
            decoder=vae.decoder,
            objective=GridWorldObjective(),
        )


def build_neo(experiment: str | GridWorldAlphaExperiment) -> GridWorldNEO:
    """Build the paper NEO model for one resolved GridWorld experiment."""

    return GridWorldNEO(experiment)


__all__ = ["GridWorldNEO", "build_neo"]
