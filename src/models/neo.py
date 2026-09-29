"""Task-independent multi-step latent theory induction and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.config import NEOExperimentConfig
from models.program_execution import ProgramExecutor
from models.quantizer import ActionQuantizer, ActionQuantizerOutput
from models.theory_programmer import TheoryProgrammer


MetricT = TypeVar("MetricT")


class ObservationPosterior(Protocol):
    """Posterior operation required by the shared NEO objective."""

    def kl_to_standard_normal(self) -> Tensor: ...


class NEOObjective(Protocol[MetricT]):
    """Task operations required by the shared latent-program rollout."""

    def validate_batch(self, batch: Tensor, *, is_eval: bool) -> None: ...

    def split_batch(self, batch: Tensor) -> tuple[Tensor, Tensor]: ...

    def reconstruction_loss(self, prediction: Tensor, target: Tensor) -> Tensor: ...

    def per_sample_reconstruction_loss(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor: ...

    def metrics(self, prediction: Tensor, target: Tensor) -> MetricT: ...

    def decode_prediction(self, prediction: Tensor) -> Tensor: ...

    def exact_match(self, prediction: Tensor, target: Tensor) -> Tensor: ...

    def empty_metrics(self) -> MetricT: ...


@dataclass(frozen=True, slots=True)
class NEOLengthMetrics(Generic[MetricT]):
    """Task metrics for samples assigned one explanation length."""

    length: int
    count: int
    reconstruction_loss: Tensor
    metrics: MetricT

    @property
    def accuracy(self) -> MetricT:
        """Backward-compatible alias used by the GridWorld metric logger."""

        return self.metrics


@dataclass(frozen=True, slots=True)
class NEOOutput(Generic[MetricT]):
    """Losses, selections, predictions, and task metrics from one rollout."""

    loss: Tensor
    reconstruction_loss: Tensor
    metrics: MetricT
    query_reconstruction_loss: Tensor | None
    query_metrics: MetricT | None
    auto_reconstruction_loss: Tensor
    auto_reconstruction_metrics: MetricT
    observation_kl_loss: Tensor
    transition_consistency_loss: Tensor
    state_prediction_loss: Tensor
    action_vq_loss: Tensor
    action_vq_by_length: tuple[ActionQuantizerOutput, ...]
    selected_lengths: Tensor
    mean_explanation_length: Tensor
    length_counts: Tensor
    per_length_metrics: tuple[NEOLengthMetrics[MetricT], ...]
    predictions: Tensor
    targets: Tensor

    @property
    def accuracy(self) -> MetricT:
        """Backward-compatible alias for task metrics."""

        return self.metrics

    @property
    def query_accuracy(self) -> MetricT | None:
        """Backward-compatible alias for query task metrics."""

        return self.query_metrics

    @property
    def auto_reconstruction_accuracy(self) -> MetricT:
        """Backward-compatible alias for observation reconstruction metrics."""

        return self.auto_reconstruction_metrics

    @property
    def prediction_logits(self) -> Tensor:
        """Backward-compatible alias for task predictions."""

        return self.predictions

    @property
    def target_grids(self) -> Tensor:
        """Backward-compatible alias retained for the GridWorld evaluator."""

        return self.targets


class NEO(nn.Module, Generic[MetricT]):
    """Infer and execute reusable latent programs for one task adapter.

    The shared model owns the paper concepts and checkpoint names directly:
    ``encoder``, ``decoder``, ``theory_programmer``, ``quantizer``, and
    ``program_executor``.  Observation shapes, reconstruction semantics,
    exact-match rules, and reported metrics are supplied by ``objective``.
    """

    def __init__(
        self,
        experiment: NEOExperimentConfig,
        *,
        encoder: nn.Module,
        decoder: nn.Module,
        objective: NEOObjective[MetricT],
        theory_programmer: nn.Module | None = None,
        program_executor: nn.Module | None = None,
        quantizer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.experiment = experiment
        self.encoder = encoder
        self.decoder = decoder
        self.objective = objective

        training = experiment.training
        # A task may supply its own programmer, executor, or quantizer when its
        # released architecture differs. Constructing the defaults in place
        # keeps initialization order, and therefore RNG consumption, unchanged.
        self.theory_programmer = (
            TheoryProgrammer(training) if theory_programmer is None else theory_programmer
        )
        self.program_executor = (
            ProgramExecutor(training) if program_executor is None else program_executor
        )
        self.quantizer = ActionQuantizer(training) if quantizer is None else quantizer
        self._observation_frozen = False
        if training.freeze_observation_encoder and training.freeze_observation_decoder:
            self.freeze_observation_model()

    @property
    def length_control_coefficient(self) -> float:
        """Current MDL penalty on explanation length.

        Tasks that anneal the penalty during training override this.
        """

        return self.experiment.length_control_coefficient


    def consistency_distance(self, current: Tensor, cleaned: Tensor) -> Tensor:
        """Per-example distance from an executed state to its grounded state."""

        return F.mse_loss(
            current.flatten(1), cleaned.flatten(1), reduction="none"
        ).mean(dim=-1)

    def reduce_consistency_loss(
        self,
        per_step_distances: Tensor,
        selected_lengths: Tensor,
    ) -> Tensor:
        """Reduce per-step grounding distances to one scalar loss term.

        ``per_step_distances`` is ``(max_transition_length, batch)``. The
        default sums the per-step batch means and contributes nothing to a
        one-step model, where the term is degenerate. Tasks whose released
        objective normalizes differently override this.
        """

        if per_step_distances.shape[0] == 1:
            return torch.zeros(
                (),
                device=per_step_distances.device,
                dtype=per_step_distances.dtype,
            )
        if getattr(self.experiment.training, "adaptive_grounding", False):
            steps = torch.arange(
                1, per_step_distances.shape[0] + 1,
                device=selected_lengths.device,
            ).unsqueeze(1)
            mask = (steps <= selected_lengths.unsqueeze(0)).to(per_step_distances.dtype)
            return (per_step_distances * mask).sum() / selected_lengths.sum().to(
                per_step_distances.dtype
            )
        return per_step_distances.mean(dim=-1).sum()

    def freeze_observation_model(self) -> None:
        """Freeze the pretrained encoder/decoder and their BatchNorm buffers."""

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.decoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()
        self.decoder.eval()
        self._observation_frozen = True

    def train(self, mode: bool = True) -> "NEO[MetricT]":
        super().train(mode)
        if self._observation_frozen:
            self.encoder.eval()
            self.decoder.eval()
        return self

    def forward(
        self,
        batch: Tensor,
        *,
        is_eval: bool = False,
        length_control_coefficient: float | None = None,
        num_transitions: int | None = None,
        hard_grounding: bool = False,
    ) -> NEOOutput[MetricT]:
        """Run the shared rollout and MDL-style explanation selection.

        ``length_control_coefficient`` overrides the experiment's fixed value
        for this call.  Tasks that anneal the MDL coefficient over training
        pass the current value each step; omitting it uses the experiment
        contract's constant.

        ``num_transitions`` overrides how many operations the rollout applies.
        The length-OOD protocol evaluates a trained model beyond its training
        horizon; omitting it uses the contract's ``max_transition_length``.
        ``hard_grounding`` is evaluation-only: decoded predictions are reencoded
        before the next transition, while selection and scoring stay unchanged.
        """

        if hard_grounding and (not is_eval or self.training):
            raise ValueError("hard grounding is an evaluation-only intervention")
        coefficient = (
            self.length_control_coefficient
            if length_control_coefficient is None
            else length_control_coefficient
        )
        transitions = (
            self.experiment.training.max_transition_length
            if num_transitions is None
            else num_transitions
        )
        if transitions < 1:
            raise ValueError("num_transitions must be positive")
        self.objective.validate_batch(batch, is_eval=is_eval)
        inputs, targets = self.objective.split_batch(batch)

        all_observations = torch.cat([inputs, targets], dim=0)
        all_states, posterior = self.encoder(all_observations)
        if posterior is None:
            raise RuntimeError("NEO requires a variational observation encoder")
        auto_predictions = self.decoder(all_states)
        auto_reconstruction_loss = self.objective.reconstruction_loss(
            auto_predictions,
            all_observations,
        )
        auto_reconstruction_metrics = self.objective.metrics(
            auto_predictions,
            all_observations,
        )
        observation_kl_loss = posterior.kl_to_standard_normal()

        current_states = all_states[: inputs.shape[0]]
        target_states = all_states[inputs.shape[0] :]
        current = current_states.clone()
        predicted_states: list[Tensor] = []
        decoded_by_length: list[Tensor] = []
        action_vq_by_length: list[ActionQuantizerOutput] = []
        consistency_by_length: list[Tensor] = []

        for _ in range(transitions):
            actions = self.theory_programmer(current.detach(), target_states)
            if is_eval:
                actions = (
                    actions[::2]
                    .unsqueeze(1)
                    .repeat(1, 2, 1, 1)
                    .reshape(-1, actions.shape[1], actions.shape[2])
                )
            quantized = self.quantizer(actions, training_mode=not is_eval)
            action_vq_by_length.append(quantized)
            current = self.program_executor(current, quantized.values)
            predicted_states.append(current.clone())

            with torch.no_grad():
                decoded = self.objective.decode_prediction(self.decoder(current))
                decoded_by_length.append(decoded)
                cleaned_state, _ = self.encoder(decoded)
            consistency_by_length.append(
                self.consistency_distance(current, cleaned_state)
            )
            if hard_grounding:
                current = cleaned_state

        predictions_by_length: list[Tensor] = []
        weighted_losses: list[Tensor] = []
        for index, state in enumerate(predicted_states):
            length = index + 1
            prediction = self.decoder(state)
            predictions_by_length.append(prediction)
            per_sample_loss = self.objective.per_sample_reconstruction_loss(
                prediction,
                targets,
            )
            weighted_losses.append(per_sample_loss * coefficient**length)

        with torch.no_grad():
            selected_lengths = torch.stack(weighted_losses, dim=1).argmin(dim=1) + 1
            # Descending assignment makes the shortest exact solution win.
            for length in range(len(decoded_by_length), 0, -1):
                correct = self.objective.exact_match(
                    decoded_by_length[length - 1],
                    targets,
                )
                selected_lengths[correct] = length

            if is_eval:
                # A transferred program includes its stopping length. Infer it
                # from support only; query targets are used solely for scoring.
                selected_lengths[1::2] = selected_lengths[::2]

        transition_consistency_loss = self.reduce_consistency_loss(
            torch.stack(consistency_by_length, dim=0),
            selected_lengths,
        )

        length_counts = torch.bincount(
            selected_lengths,
            minlength=transitions + 1,
        )[1:]
        mean_explanation_length = selected_lengths.sum() / len(selected_lengths)

        selected_predictions = torch.zeros_like(predictions_by_length[0])
        selected_states = torch.zeros_like(predicted_states[0])
        for length, prediction in enumerate(predictions_by_length, start=1):
            mask = selected_lengths.eq(length)
            if mask.any():
                selected_predictions[mask] = prediction[mask]
                selected_states[mask] = predicted_states[length - 1][mask]

        # Supervise the selected rollout's final state against the encoded
        # target. Distinct from grounding, which pulls each intermediate state
        # onto the manifold of valid observations; this pulls the endpoint onto
        # the target itself. Tasks that do not use it leave the weight at zero.
        state_prediction_loss = F.mse_loss(selected_states, target_states.detach())

        action_vq_loss = torch.zeros(
            (),
            device=batch.device,
            dtype=all_states.dtype,
        )
        for length, quantized in enumerate(action_vq_by_length, start=1):
            mask = selected_lengths.eq(length)
            if mask.any():
                action_vq_loss = action_vq_loss + quantized.loss_per_sample[mask].sum()
        action_vq_loss = action_vq_loss / len(selected_lengths)

        if is_eval:
            reconstruction_loss = self.objective.reconstruction_loss(
                selected_predictions[::2],
                targets[::2],
            )
            metrics = self.objective.metrics(
                selected_predictions[::2],
                targets[::2],
            )
            query_reconstruction_loss = self.objective.reconstruction_loss(
                selected_predictions[1::2],
                targets[1::2],
            )
            query_metrics = self.objective.metrics(
                selected_predictions[1::2],
                targets[1::2],
            )
            metric_predictions = selected_predictions[::2]
            metric_targets = targets[::2]
            metric_lengths = selected_lengths[::2]
        else:
            reconstruction_loss = self.objective.reconstruction_loss(
                selected_predictions,
                targets,
            )
            metrics = self.objective.metrics(selected_predictions, targets)
            query_reconstruction_loss = None
            query_metrics = None
            metric_predictions = selected_predictions
            metric_targets = targets
            metric_lengths = selected_lengths

        per_length_metrics = self._per_length_metrics(
            transitions,
            metric_predictions,
            metric_targets,
            metric_lengths,
        )
        training = self.experiment.training
        loss = training.reconstruction_weight * reconstruction_loss
        loss = loss + training.grounding_weight * transition_consistency_loss
        loss = loss + training.auto_reconstruction_weight * auto_reconstruction_loss
        loss = loss + training.action_vq_weight * action_vq_loss
        loss = loss + getattr(training, "state_prediction_weight", 0.0) * (
            state_prediction_loss
        )

        return NEOOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            metrics=metrics,
            query_reconstruction_loss=query_reconstruction_loss,
            query_metrics=query_metrics,
            auto_reconstruction_loss=auto_reconstruction_loss,
            auto_reconstruction_metrics=auto_reconstruction_metrics,
            observation_kl_loss=observation_kl_loss,
            transition_consistency_loss=transition_consistency_loss,
            state_prediction_loss=state_prediction_loss,
            action_vq_loss=action_vq_loss,
            action_vq_by_length=tuple(action_vq_by_length),
            selected_lengths=selected_lengths,
            mean_explanation_length=mean_explanation_length,
            length_counts=length_counts,
            per_length_metrics=per_length_metrics,
            predictions=selected_predictions,
            targets=targets,
        )

    def _per_length_metrics(
        self,
        transitions: int,
        predictions: Tensor,
        targets: Tensor,
        selected_lengths: Tensor,
    ) -> tuple[NEOLengthMetrics[MetricT], ...]:
        metrics: list[NEOLengthMetrics[MetricT]] = []
        for length in range(1, transitions + 1):
            mask = selected_lengths.eq(length)
            count = int(mask.sum().item())
            if count:
                loss = self.objective.reconstruction_loss(
                    predictions[mask],
                    targets[mask],
                )
                task_metrics = self.objective.metrics(
                    predictions[mask],
                    targets[mask],
                )
            else:
                loss = torch.zeros(
                    (),
                    device=predictions.device,
                    dtype=predictions.dtype,
                )
                task_metrics = self.objective.empty_metrics()
            metrics.append(
                NEOLengthMetrics(
                    length=length,
                    count=count,
                    reconstruction_loss=loss,
                    metrics=task_metrics,
                )
            )
        return tuple(metrics)


__all__ = [
    "NEO",
    "NEOLengthMetrics",
    "NEOObjective",
    "NEOOutput",
    "ObservationPosterior",
]
