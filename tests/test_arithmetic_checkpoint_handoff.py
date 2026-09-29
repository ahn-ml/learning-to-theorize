from dataclasses import asdict

import pytest
import torch

from tasks.arithmetic_factorization.experiment import get_experiment
from tasks.checkpoint_selection import load_candidate
from tasks.arithmetic_factorization.task import build_neo


@pytest.mark.parametrize("alpha", ["0.33", "0.66", "1.00"])
def test_public_training_checkpoint_restores_weights_and_schedule(tmp_path, alpha):
    method = "neo"
    original = build_neo(method, alpha)
    path = tmp_path / "checkpoint.pth"
    torch.save({
        "format_version": 1,
        "global_step": 1000,
        "model_state_dict": original.state_dict(),
        "config": {
            "method": method,
            "alpha": alpha,
            "seed": 42,
            "training": asdict(get_experiment(method, alpha)),
        },
    }, path)
    restored = build_neo(method, alpha)
    metadata = load_candidate(restored, "arithmetic", alpha, 42, path)
    original.update_length_control(1000, get_experiment(method, alpha).total_steps)
    assert metadata["global_step"] == 1000
    assert restored.length_control_coefficient == original.length_control_coefficient
    assert restored.experiment.training.max_transition_length == 3
    assert restored.state_dict().keys() == original.state_dict().keys()
    assert all(torch.equal(value, restored.state_dict()[key])
               for key, value in original.state_dict().items())
