"""Pretraining and downstream grounding use separate objectives."""
import torch
from models.neo import NEO
from tasks.arithmetic_factorization.observation_pretraining import observation_pretraining_config
from tasks.arithmetic_factorization.observation_runner import pretraining_parameters
from tasks.arithmetic_factorization.task import ArithmeticObservationModel, ArithmeticNEO


def test_pretraining_is_invariant_to_positive_state_scale_but_downstream_is_not():
    target = torch.tensor([[[1., 2.], [3., 4.]], [[-1., 2.], [1., -2.]]])
    current = (target * 3).requires_grad_()
    loss = ArithmeticObservationModel.consistency_distance(None, current, target)
    assert torch.allclose(loss, torch.zeros(2), atol=1e-7)
    loss.sum().backward()
    assert torch.allclose(current.grad, torch.zeros_like(current), atol=1e-7)
    assert ArithmeticNEO.consistency_distance is NEO.consistency_distance
    mse = ArithmeticNEO.consistency_distance(None, current.detach(), target)
    assert torch.equal(mse, torch.tensor([30., 10.]))


def test_observation_positions_are_learned_in_both_numerical_profiles():
    for profile in ('recovered', 'appendix'):
        config = observation_pretraining_config(profile)
        params = pretraining_parameters(config, steps_per_epoch=5)
        model = ArithmeticObservationModel(params, config=config)
        for module in (model.theory_programmer, model.program_executor):
            assert 'state_positional_embedding' in dict(module.named_parameters())
            assert 'state_positional_embedding' not in dict(module.named_buffers())
        assert config.epochs == (50 if profile == 'recovered' else 500)
        assert config.minimum_learning_rate_ratio == (.1 if profile == 'recovered' else .005)
