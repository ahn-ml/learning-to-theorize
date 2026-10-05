"""Query labels must score a transferred program, never choose its length."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models.neo import NEO
from models.quantizer import ActionQuantizerOutput
from tasks.gridworld.theorizer_evaluation import transfer_rollout
from tasks.image_editing.objective import ImageEditingObjective


class Encoder(nn.Module):
    def forward(self, observations):
        return observations.float(), SimpleNamespace(
            kl_to_standard_normal=lambda: observations.new_zeros((), dtype=torch.float))


class Programmer(nn.Module):
    def forward(self, current, target):
        return torch.ones_like(current)


class Executor(nn.Module):
    def forward(self, state, action):
        return state + action


class GridDecoder(nn.Module):
    def forward(self, state):
        return F.one_hot(state.long(), 9).float()


class Quantizer(nn.Module):
    def forward(self, actions, *, training_mode=True):
        zero = actions.new_zeros(())
        return ActionQuantizerOutput(
            values=actions, loss=zero, loss_per_sample=actions.new_zeros(actions.shape[0]),
            commitment_loss=zero, codebook_loss=zero, temperature=zero,
            indices=torch.zeros(actions.shape[0], dtype=torch.long),
            logits=actions.new_zeros(actions.shape[0], 1))


class Objective:
    def validate_batch(self, batch, *, is_eval):
        assert batch.shape[1] == 4

    def split_batch(self, batch):
        return batch[:, ::2].flatten(0, 1), batch[:, 1::2].flatten(0, 1)

    def reconstruction_loss(self, prediction, target):
        return F.mse_loss(prediction, target)

    def per_sample_reconstruction_loss(self, prediction, target):
        return (prediction - target).square().flatten(1).mean(1)

    def metrics(self, prediction, target):
        return float(self.reconstruction_loss(prediction, target))

    def decode_prediction(self, prediction):
        return prediction

    def exact_match(self, prediction, target):
        return prediction.eq(target).flatten(1).all(1)

    def empty_metrics(self):
        return 0.0


def make_model():
    training = SimpleNamespace(
        max_transition_length=3, freeze_observation_encoder=False,
        freeze_observation_decoder=False, reconstruction_weight=1.0,
        grounding_weight=0.1, auto_reconstruction_weight=0.0, action_vq_weight=1.0)
    return NEO(SimpleNamespace(training=training, length_control_coefficient=1.0),
               encoder=Encoder(), decoder=nn.Identity(), objective=Objective(),
               theory_programmer=Programmer(), program_executor=Executor(),
               quantizer=Quantizer())


def test_query_target_does_not_choose_shared_rollout_length():
    model = make_model().eval()
    batch = torch.tensor([0., 2., 10., 11.]).reshape(1, 4, 1, 1)
    first = model(batch, is_eval=True)
    batch[:, 3] = 13
    second = model(batch, is_eval=True)
    assert first.selected_lengths.tolist() == [2, 2]
    assert second.selected_lengths.tolist() == [2, 2]
    assert torch.equal(first.predictions, second.predictions)
    assert first.predictions[1].item() == 12


def test_gridworld_replays_the_whole_support_program():
    model = make_model().eval()
    model.decoder = GridDecoder()
    results = [transfer_rollout(
        model, torch.tensor([[0]]), torch.tensor([[2]]), torch.tensor([[3]]),
        torch.tensor([[target]]), max_steps=3, device=torch.device("cpu"))
        for target in [4, 5]]
    assert [result.num_applied_steps for result in results] == [2, 2]
    assert torch.equal(results[0].final_prediction, results[1].final_prediction)
    assert not results[0].solved and results[1].solved


def test_image_evaluation_does_not_pair_unrelated_episodes():
    objective = ImageEditingObjective()
    with pytest.raises(ValueError, match="one support and one query"):
        objective.validate_batch(torch.zeros(2, 2, 32, 32, 3), is_eval=True)
    objective.validate_batch(torch.zeros(2, 4, 32, 32, 3), is_eval=True)
