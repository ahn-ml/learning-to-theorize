"""NEO training uses the observation checkpoint it is given, after minimal checks."""
import json
import shutil
from dataclasses import asdict

import pytest
import torch

from tasks.gridworld.models.vae import VAE
from tasks.gridworld.observation_checkpoint import (
    inspect_observation_checkpoint, load_gridworld_observation_checkpoint,
)
from tasks.gridworld.observation_pretraining import GridWorldObservationPretrainingConfig
from tasks.gridworld.task import build_neo
from training.runtime import sha256_file


def save_release_checkpoint(path, state_dict, **overrides):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(format_version=1, model_state_dict=state_dict,
        optimizer_state_dict={}, scheduler_state_dict={},
        training_state={'completed_epochs': 9, 'global_step': 2205},
        optimization=asdict(GridWorldObservationPretrainingConfig().optimization()),
        metadata={'domain': 'gridworld', 'stage': 'observation_pretraining', 'evaluated_epoch': 9})
    payload.update(overrides)
    torch.save(payload, path)
    return path


@pytest.fixture
def released_run(tmp_path):
    """The released layout: a relative checkpoint path in best_reconstruction.json."""
    torch.manual_seed(7)
    run = tmp_path / 'observation'
    checkpoint = save_release_checkpoint(
        run / 'best_reconstruction/checkpoint_2206.pt', VAE().state_dict())
    (run / 'best_reconstruction.json').write_text(json.dumps({
        'checkpoint': 'best_reconstruction/checkpoint_2206.pt',
        'checkpoint_sha256': sha256_file(checkpoint), 'metric': 'grid_accuracy', 'score': 1.0,
        'training_state': {'completed_epochs': 9, 'global_step': 2205}}))
    return checkpoint


def test_named_checkpoint_loads_exact_observation_weights(released_run):
    assert inspect_observation_checkpoint(released_run) == sha256_file(released_run)
    model = build_neo('alpha-0.33')
    loaded = load_gridworld_observation_checkpoint(
        model, released_run, expected_sha256=sha256_file(released_run))
    expected = torch.load(released_run, weights_only=True)['model_state_dict']
    assert loaded == tuple(sorted(expected))
    assert all(torch.equal(model.state_dict()[key], expected[key]) for key in loaded)


def test_checkpoint_outside_a_run_needs_no_selection_record(released_run, tmp_path):
    copied = tmp_path / 'elsewhere' / 'observation.pt'
    copied.parent.mkdir()
    shutil.copy2(released_run, copied)
    assert load_gridworld_observation_checkpoint(build_neo('alpha-0.66'), copied)


def test_selection_record_must_name_the_checkpoint(released_run):
    other = save_release_checkpoint(
        released_run.parent / 'checkpoint_246.pt', VAE().state_dict())
    with pytest.raises(ValueError, match='pass the checkpoint it names'):
        inspect_observation_checkpoint(other)
    record_path = released_run.parent.parent / 'best_reconstruction.json'
    record = json.loads(record_path.read_text())
    record_path.write_text(json.dumps({**record, 'checkpoint_sha256': '0' * 64}))
    with pytest.raises(ValueError, match='pass the checkpoint it names'):
        inspect_observation_checkpoint(released_run)


@pytest.mark.parametrize('change', ['format', 'missing_field', 'extra_key', 'missing_key'])
def test_rejects_checkpoints_that_are_not_observation_vaes(tmp_path, change):
    state = VAE().state_dict()
    overrides = {}
    if change == 'format':
        overrides['format_version'] = 2
    elif change == 'extra_key':
        state['encoder.unexpected'] = torch.zeros(1)
    elif change == 'missing_key':
        del state['decoder.from_state.weight']
    path = save_release_checkpoint(tmp_path / 'observation.pt', state, **overrides)
    if change == 'missing_field':
        payload = torch.load(path, weights_only=True)
        del payload['optimization']
        torch.save(payload, path)
    with pytest.raises(ValueError):
        load_gridworld_observation_checkpoint(build_neo('alpha-0.33'), path)


def test_changed_checkpoint_is_not_loaded(released_run):
    with pytest.raises(ValueError, match='SHA-256 changed'):
        load_gridworld_observation_checkpoint(
            build_neo('alpha-0.33'), released_run, expected_sha256='0' * 64)
