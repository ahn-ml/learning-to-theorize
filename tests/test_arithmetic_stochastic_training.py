"""Verify active temperature-controlled assignments, not just config values."""
import math

import pytest
import torch
import torch.nn.functional as F

from tasks.arithmetic_factorization.experiment import get_experiment
from tasks.arithmetic_factorization.models.quantizer import VectorQuantizer
from tasks.arithmetic_factorization.task import build_neo


@pytest.mark.parametrize('alpha', ['0.33', '0.66', '1.00'])
def test_published_schedule_reaches_the_real_quantizer(alpha):
    config = get_experiment('neo', alpha)
    core = build_neo('neo', alpha).quantizer.quantizer
    assert core.tau_steps == int(config.total_steps * 0.25)
    for progress, expected in [(0, .3), (.5, .175), (1, .05), (2, .05)]:
        core.current_step = int(core.tau_steps * progress)
        assert core.get_temperature() == pytest.approx(expected)


def test_training_assignments_follow_active_temperature():
    quantizer = VectorQuantizer(codebook_size=2, embedding_dim=2,
                                tau_start=.3, tau_end=.05, tau_steps=100)
    with torch.no_grad():
        quantizer.embedding.weight.copy_(torch.eye(2))
    z = torch.tensor([1., .7]).expand(8192, 1, 2).clone().requires_grad_()
    frequencies = []
    for step, temperature in [(0, .3), (100, .05)]:
        quantizer.current_step = step
        torch.manual_seed(42)
        values, out = quantizer(z)
        # Nearest-only sampling would yield 100% at both temperatures.
        gap = 2 * (1 - .7) / math.sqrt(1 + .7**2)
        expected = 1 / (1 + math.exp(-gap / temperature))
        actual = (out['min_encoding_indices'] == 0).float().mean().item()
        assert actual == pytest.approx(expected, abs=.015)
        assert out['temperature'].item() == pytest.approx(temperature)
        frequencies.append(actual)
    assert frequencies[1] > frequencies[0] + .10
    (values[..., 0].mean() + out['quantizer_loss']).backward()
    assert torch.isfinite(z.grad).all() and z.grad.norm() > 0


def test_rollout_does_not_advance_schedule_and_eval_takes_nearest_code():
    core = VectorQuantizer(codebook_size=4, embedding_dim=2, use_ema=True,
                           tau_start=.3, tau_end=.05, tau_steps=100)
    for _ in range(3):
        _, out = core(torch.randn(16, 1, 2))
        assert out['temperature'].item() == pytest.approx(.3)
    assert core.current_step == 0
    core.step()
    assert core.current_step == 1
    state = {k: v.clone() for k, v in core.state_dict().items()}
    core.eval()
    z = torch.randn(16, 1, 2)
    rng = torch.get_rng_state().clone()
    values, out = core(z)  # eval() must be deterministic even with the default flag.
    codes = F.normalize(core.embedding.weight, dim=-1)
    nearest = torch.cdist(F.normalize(z.flatten(end_dim=-2), dim=-1), codes).argmin(-1)
    assert torch.equal(out['min_encoding_indices'], nearest)
    torch.testing.assert_close(values.flatten(end_dim=-2), codes[nearest])
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(core.state_dict()[k], v) for k, v in state.items())
    assert core.current_step == 1
