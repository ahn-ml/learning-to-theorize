"""Exercise public commands: the default recipe must reach the real task runner."""
import importlib.util
import argparse
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('public_common', ROOT / 'scripts/_common.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


@pytest.mark.parametrize('devices', ['', '0,', '-1', 'cpu', '0,0', '0,00'])
def test_invalid_device_lists_are_rejected(devices):
    with pytest.raises(argparse.ArgumentTypeError):
        common.parse_devices(devices)


def test_device_whitespace_and_leading_zeros_are_normalized():
    assert common.parse_devices(' 00, 01 ') == '0,1'


def test_home_relative_config_resolves_its_override_beside_the_config(tmp_path, monkeypatch):
    recipe = tmp_path / 'recipe.yaml'
    recipe.write_text((ROOT / 'configs/gridworld/reproduction.yaml').read_text())
    override = tmp_path / 'reproduction-pretraining.json'
    override.write_text('{"epochs": 50}')
    original = Path.expanduser
    monkeypatch.setattr(Path, 'expanduser', lambda path: recipe if str(path) == '~/recipe.yaml' else original(path))
    config = common.load_task_config(Path('~/recipe.yaml'), 'gridworld')
    assert config['pretraining']['overrides'] == str(override)


def command(script, task, *extra):
    return subprocess.run([sys.executable, str(ROOT / 'scripts' / script),
        '--task', task, '--data-root', '/data', '--output-root', '/output',
        '--dry-run', *extra], capture_output=True, text=True)


@pytest.mark.parametrize('task', ['gridworld', 'arithmetic_factorization'])
def test_default_pretraining_uses_actual_appendix_runner(task):
    result = command('pretrain.py', task, '--devices', '0,1,2,3')
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)['commands'][0]['argv']
    assert argv[argv.index('--pretraining-profile') + 1] == 'appendix'
    if task == 'gridworld':
        config = Path(argv[argv.index('--pretraining-overrides') + 1])
        assert json.loads(config.read_text()) == {'epochs': 50, 'kl_weight': 1e-5, 'learning_rate': .0025}


@pytest.mark.parametrize('seed', [43, 44])
def test_arithmetic_pretraining_seed_reaches_runner(seed):
    result = command('pretrain.py', 'arithmetic_factorization', '--devices', '4,5,6,7',
                     '--seed', str(seed))
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)['commands'][0]['argv']
    assert argv[argv.index('--seed') + 1] == str(seed)


def test_custom_pretraining_file_is_forwarded_not_silently_ignored(tmp_path):
    config = tmp_path / 'recipe.yaml'
    original = (ROOT / 'configs/gridworld/reproduction.yaml').read_text()
    config.write_text(original.replace('reproduction-pretraining.json', 'override.json'))
    override = tmp_path / 'override.json'
    override.write_text('{"epochs": 25, "learning_rate": 0.003}')
    result = command('pretrain.py', 'gridworld', '--devices', '0,1,2,3', '--config', str(config))
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)['commands'][0]['argv']
    assert argv[argv.index('--pretraining-overrides') + 1] == str(override)


def test_all_nine_grid_train_commands_use_selected_observation():
    result = command('train.py', 'gridworld', '--devices', '0', '--method', 'neo',
                     '--alpha', 'all', '--seed', 'all', '--observation-checkpoint', '/observation')
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['command_count'] == 9
    for row in plan['commands']:
        argv = row['argv']
        assert argv[argv.index('--observation-selection') + 1] == 'best-reconstruction'
        assert '--development' not in argv


def test_image_training_launches_two_ranks():
    args = ('--method', 'neo', '--alpha', '0.33', '--seed', '42', '--observation-checkpoint', '/observation')
    result = command('train.py', 'image_editing', '--devices', '0,1', *args)
    assert result.returncode == 0, result.stderr
    assert 'torch.distributed.run' in result.stdout and '--nproc-per-node=2' in result.stdout
    result = command('train.py', 'image_editing', '--devices', '0', *args)
    assert result.returncode != 0 and 'requires 2 devices' in result.stderr
    result = command('train.py', 'image_editing', '--devices', '0,0', *args)
    assert result.returncode != 0 and 'distinct GPU indices' in result.stderr


@pytest.mark.parametrize('task', common.TASKS)
@pytest.mark.parametrize('script', ['train.py', 'evaluate.py'])
@pytest.mark.parametrize('method', ['disc-mono', 'cont-mono', 'cont-mono-opt'])
def test_public_commands_reject_removed_methods(task, script, method):
    result = command(script, task, '--devices', '0', '--method', method,
                     '--alpha', '0.33', '--seed', '42')
    assert result.returncode != 0 and 'invalid choice' in result.stderr


def test_command_failure_stops_subsequent_runs(monkeypatch):
    from types import SimpleNamespace
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(common.subprocess, 'run', run)
    assert common.run_commands([['first'], ['second']], devices='2') == 7
    assert len(calls) == 1
    assert calls[0][1]['env']['CUDA_VISIBLE_DEVICES'] == '2'


def test_image_pretraining_explains_checkpoint_entry_point():
    result = command('pretrain.py', 'image_editing', '--devices', '0,1')
    assert result.returncode != 0 and 'provided observation checkpoint' in result.stderr


@pytest.mark.parametrize('task', common.TASKS)
def test_task_cli_dispatches_help_without_starting_a_run(task):
    result = subprocess.run([sys.executable, '-m', f'tasks.{task}', 'train', '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--output-root' in result.stdout
