"""GridWorld observation pretraining with reconstruction and auxiliary rollouts.

Auxiliary forward passes update BatchNorm statistics even when their losses
have zero weight. They are part of the pretraining computation."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tasks.gridworld.models.vae import VAE
from tasks.gridworld.observation_pretraining import (
    flatten_paper_observations,
)


OBSERVATION_PRETRAINING_ENCODER_FORWARDS_PER_STEP = 2
OBSERVATION_PRETRAINING_DECODER_FORWARDS_PER_STEP = 3
PAPER_OBSERVATION_CHECKPOINT_GLOBAL_STEP = 122_255
PAPER_OBSERVATION_CHECKPOINT_ENCODER_BATCH_NORM_UPDATES = 244_510
PAPER_OBSERVATION_CHECKPOINT_DECODER_BATCH_NORM_UPDATES = 366_765


@dataclass(frozen=True, slots=True)
class GridWorldObservationPretrainingOutput:
    """Reconstruction, KL and auxiliary rollout metrics."""

    loss: Tensor
    reconstruction_loss: Tensor
    kl_loss: Tensor
    pixel_accuracy: Tensor
    grid_accuracy: Tensor
    num_observations: int
    theorizer_reconstruction_loss: Tensor
    transition_consistency_loss: Tensor
    action_vq_loss: Tensor


@dataclass(frozen=True, slots=True)
class _QuantizedActions:
    values: Tensor
    loss: Tensor
    loss_per_sample: Tensor


class _PaperFiLMPolicy(nn.Module):
    """Auxiliary policy used during observation pretraining."""

    def __init__(self) -> None:
        super().__init__()
        hidden_dim = 128
        self.length_embedding = nn.Embedding(2, hidden_dim)
        self.input_layer = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.0),
        )
        self.film_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.film_beta = nn.Linear(hidden_dim, hidden_dim)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, 8),
        )

        # Preserve the initialization order and partial identity
        # overwrite.  The FiLM branch is constructed but receives no length in
        # the checkpoint-producing configuration.
        nn.init.ones_(self.film_gamma.weight.data.diagonal())
        nn.init.zeros_(self.film_gamma.bias.data)
        nn.init.zeros_(self.film_beta.weight.data)
        nn.init.zeros_(self.film_beta.bias.data)

    def forward(self, state: Tensor, target_state: Tensor) -> Tensor:
        batch_size = state.shape[0]
        combined = torch.cat(
            [state.view(batch_size, -1), target_state.view(batch_size, -1)],
            dim=-1,
        )
        hidden = self.input_layer(combined)
        return self.output_layers(hidden).view(batch_size, 1, 8)


class _PaperFiLMTransition(nn.Module):
    """Exact unused-transition architecture serialized by the checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        hidden_dim = 128
        self.state_layer = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.0),
        )
        self.action_proj = nn.Linear(8, hidden_dim)
        self.film_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.film_beta = nn.Linear(hidden_dim, hidden_dim)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, 32),
        )

        nn.init.ones_(self.film_gamma.weight.data.diagonal())
        nn.init.zeros_(self.film_gamma.bias.data)
        nn.init.zeros_(self.film_beta.weight.data)
        nn.init.zeros_(self.film_beta.bias.data)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        batch_size = state.shape[0]
        hidden = self.state_layer(state.reshape(batch_size, -1))
        action_embedding = self.action_proj(action.reshape(batch_size, -1))
        hidden = self.film_gamma(action_embedding) * hidden + self.film_beta(
            action_embedding
        )
        delta = self.output_layers(hidden).view(batch_size, 1, 32)
        return state + delta


class _PaperActionQuantizer(nn.Module):
    """Deterministic six-code VQ path used despite its zero loss weight."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(6, 8)
        self.embedding.weight.data.uniform_(-1.0 / 6.0, 1.0 / 6.0)

    def forward(self, actions: Tensor) -> _QuantizedActions:
        # Compute code distances in float32.
        with torch.autocast(device_type=actions.device.type, enabled=False):
            flattened = F.normalize(actions.flatten(end_dim=-2).float(), dim=-1)
            embedding = F.normalize(self.embedding.weight, dim=-1)
            distances = (
                flattened.square().sum(dim=1, keepdim=True)
                + embedding.square().sum(dim=1)
                - 2 * torch.einsum("bd,dn->bn", flattened, embedding.T)
            )
            indices = distances.argmin(dim=1)
            # Normalize codebook entries before and after reshaping.
            quantized = F.normalize(self.embedding(indices), dim=-1).view(actions.shape)
            quantized = F.normalize(quantized, dim=-1)
            normalized_actions = F.normalize(actions, dim=-1)
            commitment_loss = 0.25 * (
                quantized.detach() - normalized_actions
            ).square().mean()
            commitment_loss_per_sample = 0.25 * (
                quantized - normalized_actions.detach()
            ).square().mean(dim=(1, 2))
            codebook_loss = (
                quantized - normalized_actions.detach()
            ).square().mean()
            codebook_loss_per_sample = (
                quantized - normalized_actions.detach()
            ).square().mean(dim=(1, 2))
            loss = commitment_loss + codebook_loss
            loss_per_sample = (
                commitment_loss_per_sample + codebook_loss_per_sample
            )
            straight_through = normalized_actions + (
                quantized - normalized_actions
            ).detach()
        return _QuantizedActions(straight_through, loss, loss_per_sample)


class GridWorldObservationPretrainingModel(VAE):
    """Observation model with auxiliary programmer and executor modules.

    ``encoder`` and ``decoder`` are constructed first so their initial
    tensors match :class:`VAE` for the same random seed.
    """

    def __init__(self) -> None:
        super().__init__()
        self.policy = _PaperFiLMPolicy()
        self.transition = _PaperFiLMTransition()
        self.quantizer = _PaperActionQuantizer()

    def forward(
        self,
        episode_grids: Tensor,
    ) -> GridWorldObservationPretrainingOutput:
        observations = flatten_paper_observations(episode_grids)
        autoencoding = super().forward(observations)
        posterior = autoencoding.posterior
        if posterior is None:
            raise RuntimeError(
                "GridWorld observation pretraining requires a variational encoder"
            )

        reconstruction_loss = F.cross_entropy(
            autoencoding.logits.reshape(-1, autoencoding.logits.shape[-1]),
            observations.reshape(-1).long(),
            reduction="mean",
        )
        kl_loss = posterior.kl_to_standard_normal()
        pixel_accuracy, grid_accuracy = _historical_accuracy(
            autoencoding.logits,
            observations,
        )

        num_input_observations = observations.shape[0] // 2
        current_states = autoencoding.state[:num_input_observations]
        target_states = autoencoding.state[num_input_observations:]
        predicted_states = current_states.clone()
        actions = self.policy(predicted_states.detach(), target_states)
        if not self.training:
            if actions.shape[0] % 2:
                raise ValueError("paper evaluation requires paired support/query actions")
            actions = (
                actions[::2]
                .unsqueeze(1)
                .repeat(1, 2, 1, 1)
                .reshape(-1, actions.shape[1], actions.shape[2])
            )
        quantized = self.quantizer(actions)
        predicted_states = self.transition(predicted_states, quantized.values)
        selected_states = predicted_states.clone()

        # Both calls are required.  Although this branch has zero loss weight,
        # training-mode decoder/encoder calls update BatchNorm and consume the
        # dropout and VAE random streams used by the next batch.
        with torch.no_grad():
            decoded = self.decoder(predicted_states).argmax(dim=-1)
            cleaned_states, _ = self.encoder(decoded)

        transition_consistency_loss = F.mse_loss(
            predicted_states,
            cleaned_states,
            reduction="none",
        ).sum(dim=-1).mean()
        theorizer_logits = self.decoder(selected_states)
        targets = observations[num_input_observations:]

        # With max transition length one, length selection has only one
        # candidate. Keep the CopySlices graph so unused auxiliary parameters
        # receive explicit zero gradients.
        selected_logits = torch.zeros_like(theorizer_logits)
        selected_logits[:] = theorizer_logits
        evaluated_logits = selected_logits[::2] if not self.training else selected_logits
        evaluated_targets = targets[::2] if not self.training else targets
        theorizer_reconstruction_loss = F.cross_entropy(
            evaluated_logits.reshape(-1, evaluated_logits.shape[-1]),
            evaluated_targets.reshape(-1).long(),
            reduction="mean",
        )
        action_vq_loss = quantized.loss_per_sample.mean()

        # Match the source expression, including zero-weight graph edges.  They
        # cause AdamW to decay connected auxiliary parameters and thereby alter
        # later BatchNorm side-effect inputs even though only reconstruction is
        # optimized.
        loss = 0.0 * theorizer_reconstruction_loss
        loss += 0.0 * transition_consistency_loss
        loss += reconstruction_loss
        loss += 0.0 * action_vq_loss

        return GridWorldObservationPretrainingOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kl_loss=kl_loss,
            pixel_accuracy=pixel_accuracy,
            grid_accuracy=grid_accuracy,
            num_observations=observations.shape[0],
            theorizer_reconstruction_loss=theorizer_reconstruction_loss,
            transition_consistency_loss=transition_consistency_loss,
            action_vq_loss=action_vq_loss,
        )


@dataclass(frozen=True, slots=True)
class GridWorldObservationPretrainingObjective:
    """Callable adapter used by the shared observation optimization step."""

    kl_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.kl_weight < 0:
            raise ValueError("kl_weight must be non-negative")

    def __call__(
        self,
        model: nn.Module,
        episode_grids: Tensor,
    ) -> GridWorldObservationPretrainingOutput:
        output = model(episode_grids)
        if not isinstance(output, GridWorldObservationPretrainingOutput):
            raise TypeError(
                "observation pretraining objective requires "
                "GridWorldObservationPretrainingModel"
            )
        if self.kl_weight:
            return replace(output, loss=output.loss + self.kl_weight * output.kl_loss)
        return output


def observation_pretraining_batch_norm_updates(global_step: int) -> tuple[int, int]:
    """Return encoder/decoder update counts implied by a training step."""

    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    return (
        OBSERVATION_PRETRAINING_ENCODER_FORWARDS_PER_STEP * global_step,
        OBSERVATION_PRETRAINING_DECODER_FORWARDS_PER_STEP * global_step,
    )


def _historical_accuracy(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(targets)
    batch_size = targets.shape[0]
    pixel_correct = correct.float().mean(dim=(1, 2)).sum().item()
    grid_correct = correct.view(batch_size, -1).all(dim=1).sum().item()
    return (
        torch.tensor(pixel_correct / batch_size, device=logits.device),
        torch.tensor(grid_correct / batch_size, device=logits.device),
    )
