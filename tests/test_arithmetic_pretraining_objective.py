"""Arithmetic observation pretraining settings and objective."""
from dataclasses import asdict

import torch

from tasks.arithmetic_factorization.observation_pretraining import ArithmeticObservationPretrainingConfig
from tasks.arithmetic_factorization.observation_runner import pretraining_parameters
from tasks.arithmetic_factorization.task import ArithmeticObservationModel

# Settings recorded in the distributed observation checkpoint.
RELEASE = {
    "seed": 42, "epochs": 500, "per_rank_batch_size": 512, "effective_world_size": 4,
    "log_interval_steps": 50, "learning_rate": 0.003, "weight_decay": 0.01, "max_gradient_norm": 1.0,
    "warmup_ratio": 0.05, "minimum_learning_rate_ratio": 0.005,
    "theory_programmer_learning_rate_scale": 0.3, "program_executor_learning_rate_scale": 1.0,
    "num_digits": 4, "num_symbols": 10, "state_dim": 8, "action_dim": 2, "num_state_tokens": 4,
    "num_action_tokens": 1, "action_codebook_size": 8, "max_transition_length": 1,
    "auto_reconstruction_weight": 1.0, "grounding_weight": 0.0, "reconstruction_weight": 0.0,
    "action_vq_weight": 0.0,
}


def model():
    config = ArithmeticObservationPretrainingConfig()
    return ArithmeticObservationModel(pretraining_parameters(config, steps_per_epoch=5), config)


def test_default_settings_are_the_release_recipe():
    assert asdict(ArithmeticObservationPretrainingConfig()) == RELEASE


def test_loss_is_digit_reconstruction_only():
    torch.manual_seed(0)
    output = model().train()(torch.randint(0, 10, (8, 4, 1, 4)))
    assert torch.equal(output.loss, output.auto_reconstruction_loss)


def test_observation_positions_are_learned():
    pretraining = model()
    for module in (pretraining.theory_programmer, pretraining.program_executor):
        assert 'state_positional_embedding' in dict(module.named_parameters())
        assert 'state_positional_embedding' not in dict(module.named_buffers())
