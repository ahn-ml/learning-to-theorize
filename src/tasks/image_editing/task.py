"""Image Editing model composition for the shared NEO model."""

from __future__ import annotations

from models.neo import NEO
from tasks.image_editing.experiment_config import (
    Experiment,
    TrainingConfig,
    get_experiment,
)
from tasks.image_editing.models.program_execution import ProgramExecutor
from tasks.image_editing.models.quantizer import (
    ActionQuantizer,
    ActionQuantizerConfig,
)
from tasks.image_editing.models.theory_programmer import (
    LatentProgramConfig,
    TheoryProgrammer,
)
from tasks.image_editing.models.vae import Decoder, Encoder, VAEConfig
from tasks.image_editing.objective import ImageEditingMetrics, ImageEditingObjective


def latent_program_config(training: TrainingConfig, *, film_layers: int) -> LatentProgramConfig:
    """Project the experiment contract onto the latent-program module shape.

    ``film_layers`` selects between the programmer's and the executor's depth,
    which the contract records separately even though every released run used
    four for both.
    """

    return LatentProgramConfig(
        state_dim=training.state_dim,
        action_dim=training.action_dim,
        num_state_tokens=training.num_state_tokens,
        num_action_tokens=training.num_action_tokens,
        hidden_dim=training.policy_hidden_dim,
        num_film_layers=film_layers,
        dropout=training.dropout,
        max_transition_length=training.max_transition_length,
    )


def action_quantizer_config(training: TrainingConfig) -> ActionQuantizerConfig:
    """Project the experiment contract onto the action codebook."""

    return ActionQuantizerConfig(
        codebook_size=training.action_codebook_size,
        action_dim=training.action_dim,
        commitment_weight=training.action_commitment_weight,
        entropy_weight=training.action_entropy_weight,
        entropy_temperature=training.action_entropy_temperature,
        ema_decay=training.action_ema_decay,
        ema_epsilon=training.action_ema_epsilon,
    )


class ImageEditingNEO(NEO[ImageEditingMetrics]):
    """Shared NEO composed with the Image Editing modules.

    The latent-program modules are injected rather than built by the shared
    rollout: Image Editing stacks four FiLM blocks where GridWorld applies one,
    and its codebook is EMA-updated.

    Construction order is encoder, decoder, programmer, executor, quantizer,
    matching the paper-producing model so a released checkpoint loads and a
    retrained model consumes the same random numbers.
    """

    def __init__(self, experiment: str | Experiment) -> None:
        resolved = get_experiment(experiment) if isinstance(experiment, str) else experiment
        training = resolved.training
        observation = VAEConfig(
            state_dim=training.state_dim,
            num_state_tokens=training.num_state_tokens,
            dropout=training.dropout,
        )
        encoder = Encoder(observation)
        decoder = Decoder(observation)
        programmer_config = latent_program_config(
            training, film_layers=training.policy_film_layers
        )
        executor_config = latent_program_config(
            training, film_layers=training.transition_film_layers
        )
        super().__init__(
            resolved,
            encoder=encoder,
            decoder=decoder,
            objective=ImageEditingObjective(),
            theory_programmer=TheoryProgrammer(programmer_config),
            program_executor=ProgramExecutor(executor_config),
            quantizer=ActionQuantizer(action_quantizer_config(training)),
        )


def build_neo(experiment: str | Experiment) -> ImageEditingNEO:
    """Build the paper model for one alpha (``"0.33"``) or resolved experiment."""

    return ImageEditingNEO(experiment)


__all__ = [
    "ImageEditingNEO",
    "action_quantizer_config",
    "build_neo",
    "latent_program_config",
]
