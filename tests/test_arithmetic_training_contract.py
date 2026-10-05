"""Resolved Arithmetic NEO training settings."""
from dataclasses import asdict

import pytest

from tasks.arithmetic_factorization.experiment import get_experiment
from tasks.arithmetic_factorization.task import build_neo

EXPECTED_0_33 = {
    "action_codebook_size": 16,
    "action_dim": 4,
    "alpha": "0.33",
    "length_control_coefficient": 1.01,
    "length_control_coefficient_end": 0.99,
    "length_control_scheduling_ratio": 0.1,
    "loss_weights": {"action_vq": 1.0, "auto_reconstruction": 0.0, "grounding": 0.5, "reconstruction": 1.0},
    "max_transition_length": 3,
    "num_action_tokens": 1,
    "num_digits": 6,
    "num_state_tokens": 6,
    "num_symbols": 10,
    "optimization": {
        "batch_size": 512, "epochs": 200, "learning_rate": 0.0015, "max_gradient_norm": 1.0,
        "minimum_learning_rate_ratio": 0.1, "precision": "bf16-mixed",
        "program_executor_learning_rate_scale": 1.0, "theory_programmer_learning_rate_scale": 0.5,
        "warmup_ratio": 0.05, "weight_decay": 0.01,
    },
    "program_executor": {"dropout": 0.0, "feedforward_dim": 128, "model_dim": 32, "num_heads": 2, "num_layers": 4},
    "quantizer": {
        "commitment_weight": 0.25, "exponential_moving_average_decay": 0.99, "scheduling_ratio": 0.25,
        "tau_end": 0.05, "tau_start": 0.3, "use_exponential_moving_average": True,
    },
    "state_dim": 8,
    "theory_programmer": {"dropout": 0.0, "feedforward_dim": 256, "model_dim": 64, "num_heads": 4, "num_layers": 6},
    "total_steps": 57400,
}


@pytest.mark.parametrize("alpha,steps", [("0.33", 57400), ("0.66", 80800), ("1.00", 109400)])
def test_training_settings(alpha, steps):
    assert asdict(get_experiment("neo", alpha)) == {**EXPECTED_0_33, "alpha": alpha, "total_steps": steps}


def test_only_neo_is_available():
    with pytest.raises(ValueError):
        get_experiment("cont-mono", "0.33")


def test_settings_reach_the_model():
    model = build_neo("neo", "0.33")
    q = model.quantizer.quantizer
    assert (q.tau_start, q.tau_end, q.tau_steps, q.ema_decay) == (0.3, 0.05, 14350, 0.99)
    assert model.length_control_coefficient == 1.01
    assert not any(p.requires_grad for p in [*model.encoder.parameters(), *model.decoder.parameters()])
