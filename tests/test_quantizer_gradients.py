"""Check the gradient contract, not just the numerically identical VQ losses."""

import pytest
import torch
from torch.nn import functional as F

from models.quantizer import ActionQuantizer
from tasks.gridworld.theorizer_config import get_theorizer_experiment
from tasks.image_editing.models.quantizer import ActionQuantizer as ImageQuantizer
from tasks.image_editing.models.quantizer import ActionQuantizerConfig
from tasks.arithmetic_factorization.models.quantizer import VectorQuantizer
from tasks.arithmetic_factorization.models.adapters import ArithmeticActionQuantizer


def make_quantizer(task, ema):
    if task == "gridworld":
        return ActionQuantizer(get_theorizer_experiment("alpha-0.33").training)
    if task == "image_editing":
        return ImageQuantizer(ActionQuantizerConfig(use_ema=ema, entropy_weight=0))
    return ArithmeticActionQuantizer(VectorQuantizer(
        codebook_size=6, embedding_dim=16, use_ema=ema, entropy_loss_weight=0))


CASES = [("gridworld", False), ("image_editing", False),
         ("image_editing", True), ("arithmetic", False), ("arithmetic", True)]


@pytest.mark.parametrize("task,ema", CASES)
def test_commitment_reductions_have_the_same_gradients(task, ema):
    torch.manual_seed(42)
    quantizer = make_quantizer(task, ema).eval()
    actions = torch.randn(4, 1, 16, requires_grad=True)
    output = quantizer(actions, training_mode=False)
    parameters = [actions] + [p for p in quantizer.parameters() if p.requires_grad]
    scalar = torch.autograd.grad(output.loss, parameters, retain_graph=True)
    per_sample = torch.autograd.grad(output.loss_per_sample.mean(), parameters)
    assert scalar[0].norm() > 0
    for expected, actual in zip(scalar, per_sample):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("task,ema", CASES)
def test_straight_through_matches_normalized_programmer_gradient(task, ema):
    torch.manual_seed(43)
    quantizer = make_quantizer(task, ema).eval()
    actions = torch.randn(4, 1, 16, requires_grad=True)
    output = quantizer(actions, training_mode=False)
    upstream = torch.randn_like(actions)
    actual = torch.autograd.grad((output.values * upstream).sum(), actions,
                                 retain_graph=True)[0]
    expected = torch.autograd.grad((F.normalize(actions, dim=-1) * upstream).sum(),
                                   actions)[0]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("task", ["image_editing", "arithmetic"])
def test_ema_buffers_do_not_retain_training_graphs(task):
    quantizer = make_quantizer(task, True).train()
    core = quantizer.quantizer if task == "arithmetic" else quantizer
    before = core.embedding.weight.detach().clone()
    for _ in range(3):
        actions = torch.randn(4, 1, 16, requires_grad=True)
        output = quantizer(actions)
        output.loss_per_sample.mean().backward()
        quantizer.step()
        assert actions.grad is not None and actions.grad.norm() > 0
        assert not core.ema_weight.requires_grad
        assert core.ema_weight.grad_fn is None
        assert core.embedding.weight.grad is None
    assert not torch.equal(before, core.embedding.weight)


@pytest.mark.parametrize("task,ema", CASES)
def test_evaluation_does_not_mutate_codebook(task, ema):
    quantizer = make_quantizer(task, ema).eval()
    before = {key: value.clone() for key, value in quantizer.state_dict().items()}
    quantizer(torch.randn(4, 1, 16), training_mode=False)
    for key, value in quantizer.state_dict().items():
        assert torch.equal(value, before[key]), key
