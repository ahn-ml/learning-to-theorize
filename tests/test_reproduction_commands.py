"""Exercise public commands: the release recipe must reach the real task runner."""
import importlib
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

TRACKING = ['--wandb-project', 'LearningToTheorize', '--wandb-mode', 'online']


@pytest.mark.parametrize('devices', ['', '0,', '-1', 'cpu', '0,0', '0,00'])
def test_invalid_device_lists_are_rejected(devices):
    with pytest.raises(argparse.ArgumentTypeError):
        common.parse_devices(devices)


def test_device_whitespace_and_leading_zeros_are_normalized():
    assert common.parse_devices(' 00, 01 ') == '0,1'


def command(script, task, *extra):
    return subprocess.run([sys.executable, str(ROOT / 'scripts' / script),
        '--task', task, '--data-root', '/data', '--output-root', '/output',
        '--dry-run', *extra], capture_output=True, text=True)


def plan(script, task, *extra):
    if script == 'generate_data.py':
        result = subprocess.run([sys.executable, str(ROOT / 'scripts' / script), '--task', task,
                                 '--data-root', '/data', '--dry-run'], capture_output=True, text=True)
    else:
        result = command(script, task, *extra)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def runner_argv(argv):
    index = len(argv) - 1 - argv[::-1].index('-m')
    return argv[index + 1], argv[index + 2:]


@pytest.mark.parametrize('task', ['gridworld', 'arithmetic_factorization'])
def test_pretraining_passes_only_data_output_and_tracking(task):
    commands = plan('pretrain.py', task, '--devices', '0,1,2,3')['commands']
    assert len(commands) == 1
    argv = commands[0]['argv']
    assert '--nproc-per-node=4' in argv
    module, options = runner_argv(argv)
    assert module == f'tasks.{task}.observation_runner'
    assert options[0::2][:3] == ['--train-h5', '--test-h5', '--output-root']
    assert options[5:] == ['/output', *TRACKING]


def test_all_nine_grid_train_commands_use_the_given_observation_checkpoint():
    result = plan('train.py', 'gridworld', '--devices', '0', '--alpha', 'all', '--seed', 'all',
                  '--observation-checkpoint', '/observation')
    assert result['command_count'] == 9
    for row in result['commands']:
        module, options = runner_argv(row['argv'])
        assert module == 'tasks.gridworld.theorizer_runner'
        assert options[0::2] == ['--seed', '--train-h5', '--test-h5', '--observation-checkpoint',
                                 '--output-root', '--wandb-project', '--wandb-mode', '--experiment']
        assert options[options.index('--observation-checkpoint') + 1] == '/observation'


def test_image_training_launches_two_ranks():
    args = ('--alpha', '0.33', '--seed', '42', '--observation-checkpoint', '/observation')
    argv = plan('train.py', 'image_editing', '--devices', '0,1', *args)['commands'][0]['argv']
    assert 'torch.distributed.run' in argv and '--nproc-per-node=2' in argv
    module, options = runner_argv(argv)
    assert options[-2:] == ['--alpha', '0.33']
    result = command('train.py', 'image_editing', '--devices', '0', *args)
    assert result.returncode != 0 and 'requires 2 devices' in result.stderr
    result = command('train.py', 'image_editing', '--devices', '0,0', *args)
    assert result.returncode != 0 and 'distinct GPU indices' in result.stderr


class Parsed(Exception):
    pass


@pytest.mark.parametrize('script,task', [
    ('generate_data.py', 'gridworld'), ('generate_data.py', 'arithmetic_factorization'),
    ('generate_data.py', 'image_editing'),
    ('pretrain.py', 'gridworld'), ('pretrain.py', 'arithmetic_factorization'),
    ('train.py', 'gridworld'), ('train.py', 'arithmetic_factorization'), ('train.py', 'image_editing'),
])
def test_task_runner_accepts_the_public_argv(monkeypatch, script, task):
    devices = {'pretrain.py': '0,1,2,3', 'train.py': '0,1' if task == 'image_editing' else '0'}
    extra = [] if script == 'generate_data.py' else ['--devices', devices[script]]
    if script == 'train.py':
        extra += ['--alpha', '0.33', '--seed', '42', '--observation-checkpoint', '/observation']
    module, options = runner_argv(plan(script, task, *extra)['commands'][0]['argv'])
    parse = argparse.ArgumentParser.parse_args
    def parse_only(self, args=None, namespace=None):
        raise Parsed(parse(self, args, namespace))
    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', parse_only)
    with pytest.raises(Parsed):
        importlib.import_module(module).main(options)


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
