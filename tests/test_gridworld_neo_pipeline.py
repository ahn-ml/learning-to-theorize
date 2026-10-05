"""GridWorld NEO training contract and a small CPU run through checkpoint loading and evaluation."""
import json
from dataclasses import replace

import pytest
import torch

from test_gridworld_observation_handoff import save_release_checkpoint
from test_gridworld_single_observation import write_artifact
from tasks.gridworld import task, theorizer_runner
from tasks.gridworld.models.vae import VAE
from tasks.gridworld.theorizer_config import get_theorizer_experiment
from tasks.gridworld.theorizer_evaluation import (
    GridWorldEvaluationRunConfig, load_theorizer_checkpoint, run_theorizer_evaluation,
)
from tasks.gridworld.theorizer_runner import (
    GridWorldTheorizerRunConfig, _config_from_arguments, build_parser, run_gridworld_theorizer,
)
from training.runtime import sha256_file

ARGV = ['--experiment', 'alpha-0.33', '--seed', '42', '--train-h5', 'train.h5',
        '--test-h5', 'test.h5', '--observation-checkpoint', 'observation.pt', '--output-root', 'out']


def test_cli_is_the_documented_training_contract():
    config = _config_from_arguments(build_parser().parse_args(ARGV))
    assert config.run_name == 'gridworld-alpha-0.33-seed-42'
    assert config.paper_experiment == get_theorizer_experiment('alpha-0.33')
    for removed in (['--method', 'neo'], ['--observation-selection', 'best-reconstruction'],
                    ['--resume-checkpoint', 'x.pth'], ['--development'], ['--wandb-mode', 'disabled']):
        with pytest.raises(SystemExit):
            build_parser().parse_args(ARGV + removed)


def test_cpu_run_saves_candidates_that_selection_and_evaluation_load(tmp_path, monkeypatch):
    real = get_theorizer_experiment('alpha-0.33')
    tiny = replace(real, training=replace(
        real.training, epochs=2, train_episodes=8, test_episodes=4, batch_size=4,
        num_workers=0, checkpoint_interval_epochs=1, log_interval_steps=1))
    for module in (task, theorizer_runner):
        monkeypatch.setattr(module, 'get_theorizer_experiment', lambda name: tiny)
    write_artifact(tmp_path / 'train.h5', 8)
    write_artifact(tmp_path / 'test.h5', 4)
    observation = save_release_checkpoint(tmp_path / 'observation.pt', VAE().state_dict())

    result = run_gridworld_theorizer(GridWorldTheorizerRunConfig(
        experiment='alpha-0.33', seed=42, train_h5=tmp_path / 'train.h5',
        test_h5=tmp_path / 'test.h5', observation_checkpoint=observation,
        output_root=tmp_path / 'runs', wandb_mode='disabled'))
    run = result.output_directory
    assert run.parent.name == 'gridworld-alpha-0.33-seed-42'
    assert result.state.completed_epochs == 2 and result.state.global_step == 4
    assert sorted(path.name for path in (run / 'checkpoints').glob('*.pth')) == [
        'best_model.pth', 'checkpoint_1.pth', 'checkpoint_3.pth']
    config = json.loads((run / 'config.json').read_text())
    assert config['method'] == 'neo'
    assert config['artifacts']['observation']['sha256'] == sha256_file(observation)
    assert json.loads((run / 'status.json').read_text())['status'] == 'completed'
    rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    assert sum('eval/support_grid_accuracy' in row for row in rows) == 2
    assert sum('train/loss' in row for row in rows) == 4

    checkpoint = run / 'checkpoints' / 'checkpoint_3.pth'
    payload = torch.load(checkpoint, weights_only=True)
    assert set(payload) == {'format_version', 'epoch', 'global_step', 'model_state_dict', 'args'}
    assert (payload['epoch'], payload['global_step']) == (1, 2)
    load_theorizer_checkpoint(task.build_neo('alpha-0.33'), checkpoint,
        expected_experiment='alpha-0.33', expected_method='neo', expected_seed=42)

    for protocol in ('standard', 'test-time-scaling'):
        output, evaluation = run_theorizer_evaluation(GridWorldEvaluationRunConfig(
            experiment='alpha-0.33', model_seed=42, split='id', checkpoint=checkpoint,
            data_h5=tmp_path / 'test.h5', output_root=tmp_path / 'evaluation',
            protocol=protocol, num_samples=2, strict=False, force_cpu=True,
            wandb_mode='disabled'))
        assert json.loads((output / 'status.json').read_text())['status'] == 'completed'
        assert json.loads((output / 'result.json').read_text()) == evaluation.legacy_dict()
