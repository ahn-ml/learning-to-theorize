import json
from dataclasses import asdict

import pytest
import torch

from tasks.gridworld.observation_runner import save_best_reconstruction_checkpoint
from training import EpochTrainingState, OptimizationConfig


def test_best_snapshot_retains_first_tie_and_does_not_change_training(tmp_path):
    torch.manual_seed(42)
    model = torch.nn.Linear(2, 2)
    control = torch.nn.Linear(2, 2)
    control.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    configuration = OptimizationConfig(10, .001, .01, 1., 0., .005)
    best = 0.
    saved_first_max = None
    for step, score in enumerate([.5, 1., 1., .9]):
        rng = torch.get_rng_state().clone()
        best = save_best_reconstruction_checkpoint(
            tmp_path, score=score, best_score=best, model=model,
            optimizer=optimizer, scheduler=scheduler,
            state=EpochTrainingState(completed_epochs=step, global_step=step),
            optimization=configuration, metadata={"canonical": False},
        )
        assert torch.equal(rng, torch.get_rng_state())
        record = json.loads((tmp_path / 'best_reconstruction.json').read_text())
        if step == 1:
            saved_first_max = record
        if step >= 1:
            assert record == saved_first_max
        # Continue the actual optimizer updates after every save, including
        # ties and regression. The unsaved control must remain identical.
        data = torch.randn(3, 2)
        for network, opt in [(model, optimizer), (control, control_optimizer)]:
            opt.zero_grad()
            network(data).square().mean().backward()
            opt.step()
        for actual, expected in zip(model.parameters(), control.parameters()):
            assert torch.equal(actual, expected)
    assert len(list((tmp_path / 'best_reconstruction').glob('*.pt'))) == 2
    payload = torch.load(record['checkpoint'], weights_only=True)
    assert payload['training_state'] == asdict(EpochTrainingState(1, 1))
    assert payload['optimization'] == asdict(configuration)
    assert record['score'] == 1. and not record['selection_uses_ood_scores']
    assert any(not torch.equal(payload['model_state_dict'][k], value)
               for k, value in model.state_dict().items())


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -0.1, 1.1])
def test_invalid_validation_score_cannot_publish_a_checkpoint(tmp_path, score):
    with pytest.raises(ValueError, match='validation grid accuracy'):
        save_best_reconstruction_checkpoint(
            tmp_path, score=score, best_score=0., model=None, optimizer=None,
            scheduler=None, state=EpochTrainingState(0, 0), optimization=None,
            metadata={},
        )
    assert not list(tmp_path.iterdir())
