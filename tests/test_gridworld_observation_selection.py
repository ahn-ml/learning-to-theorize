"""Observation selection requires a completed pretraining run."""
import json
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tasks.gridworld.data.artifacts import get_artifact_spec
from tasks.gridworld.models.vae import VAE
from tasks.gridworld.observation_checkpoint import (
    inspect_observation_checkpoint, load_gridworld_observation_checkpoint,
)
from tasks.gridworld.observation_pretraining import GridWorldObservationPretrainingConfig
from training.runtime import sha256_file


def test_public_campaign_forwards_selection_to_all_nine_runs():
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / 'scripts/train.py'), '--task', 'gridworld',
        '--config', str(root / 'configs/gridworld/reproduction.yaml'), '--method', 'neo',
        '--alpha', 'all', '--seed', 'all', '--data-root', '/data', '--output-root', '/output',
        '--observation-checkpoint', '/observation', '--devices', '0', '--dry-run']
    original = json.loads(subprocess.check_output(command + ['--observation-selection', 'recorded-final'], text=True))
    selected = json.loads(subprocess.check_output(command + [
        '--observation-selection', 'best-reconstruction'], text=True))
    assert original['command_count'] == selected['command_count'] == 9
    for before, after in zip(original['commands'], selected['commands']):
        argv = after['argv'].copy()
        index = argv.index('--observation-selection')
        assert argv[index + 1] == 'best-reconstruction'
        del argv[index:index + 2]
        assert argv == before['argv']
        assert after['environment'] == before['environment']


def test_cli_selection_is_explicit_without_numerical_overrides():
    from tasks.gridworld.theorizer_runner import build_parser, _config_from_arguments
    argv = ['--experiment', 'alpha-0.33', '--seed', '42', '--train-h5', '/train',
            '--test-h5', '/test', '--observation-checkpoint', '/observation',
            '--output-root', '/output']
    original = _config_from_arguments(build_parser().parse_args(argv))
    selected = _config_from_arguments(build_parser().parse_args(
        argv + ['--observation-selection', 'best-reconstruction']))
    assert original.observation_selection == 'recorded-final'
    assert selected.canonical and original.canonical
    assert selected.paper_experiment == original.paper_experiment
    assert selected.epochs == original.epochs
    assert selected.batch_size == original.batch_size


@pytest.fixture
def completed_run(tmp_path):
    cfg = GridWorldObservationPretrainingConfig(kl_weight=1e-5)
    checkpoint = tmp_path / 'best_reconstruction/checkpoint_2941.pt'
    checkpoint.parent.mkdir()
    state = {'completed_epochs': 12, 'global_step': 2940}
    model = VAE()
    torch.save(dict(format_version=1, model_state_dict=model.state_dict(),
        optimizer_state_dict={}, scheduler_state_dict={}, training_state=state,
        optimization=asdict(cfg.optimization()), metadata=dict(
            domain='gridworld', stage='observation_pretraining',
            position='after_eval_before_train', evaluated_epoch=12,
            historical_checkpoint_number=2941, source_commit='a' * 40,
            canonical=True, pretraining_profile='appendix', kl_weight=1e-5)), checkpoint)
    records = {
        'config.json': dict(canonical=True, pretraining_profile='appendix',
            pretraining=asdict(cfg), source=dict(commit='a' * 40, clean=True, pushed=True),
            artifacts={name: {'sha256': get_artifact_spec('vae-pretraining', split).sha256}
                for name, split in [('train', 'practice'), ('test', 'exam')]}),
        'status.json': dict(status='completed', completed_epochs=500,
            global_step=122500, stopped_early=False),
        'best_reconstruction.json': dict(
            policy='global-validation-reconstruction-first-maximum-v1',
            metric='grid_accuracy', score=1., direction='max',
            aggregation='global_sample_weighted', tie_break='earliest',
            selection_uses_downstream_scores=False, selection_uses_ood_scores=False,
            training_continues_after_selection=True, training_state=state,
            checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint)),
    }
    for name, record in records.items():
        (tmp_path / name).write_text(json.dumps(record))
    rows = [dict(epoch=e, global_step=(e - 1) * 245,
                 **{'eval/grid_accuracy': 1. if e >= 13 else .5}) for e in range(1, 501)]
    (tmp_path / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return checkpoint


def test_explicit_selection_loads_exact_weights_without_relabeling_history(completed_run):
    with pytest.raises(ValueError, match='supported pretraining record'):
        inspect_observation_checkpoint(completed_run, require_canonical=True)
    model = VAE()
    result = load_gridworld_observation_checkpoint(model, completed_run,
        require_canonical=True, selection_policy='best-reconstruction')
    evidence = result.evidence
    assert not evidence.matches_paper_pretraining_record
    assert not evidence.matches_appendix_pretraining_record
    assert evidence.selection_policy == 'best-reconstruction'
    assert evidence.selection_metrics_sha256
    expected = torch.load(completed_run, weights_only=True)['model_state_dict']
    assert result.loaded_keys
    assert all(torch.equal(model.state_dict()[k], expected[k]) for k in result.loaded_keys)


@pytest.mark.parametrize('change', ['incomplete', 'earlier_maximum', 'missing_epoch', 'wrong_hash', 'changed_config'])
def test_selection_rejects_incomplete_or_inconsistent_evidence(completed_run, change):
    run = completed_run.parent.parent
    if change in ('earlier_maximum', 'missing_epoch'):
        path = run / 'metrics.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if change == 'earlier_maximum':
            rows[0]['eval/grid_accuracy'] = 1.
        else:
            rows.pop()
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    else:
        name = {'incomplete': 'status.json', 'wrong_hash': 'best_reconstruction.json',
                'changed_config': 'config.json'}[change]
        path = run / name
        record = json.loads(path.read_text())
        if change == 'incomplete':
            record['completed_epochs'] = 499
        elif change == 'wrong_hash':
            record['checkpoint_sha256'] = '0' * 64
        else:
            record['pretraining']['kl_weight'] = 0.
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        inspect_observation_checkpoint(completed_run, require_canonical=True,
                                       selection_policy='best-reconstruction')


def test_completed_sensitivity_pretraining_can_be_moved(completed_run, tmp_path):
    from dataclasses import replace
    import shutil
    run = completed_run.parent.parent
    cfg = replace(GridWorldObservationPretrainingConfig(), epochs=50, learning_rate=.0025, kl_weight=1e-5)
    config = json.loads((run / 'config.json').read_text())
    config.update(canonical=False, pretraining_sweep=True, pretraining=asdict(cfg))
    (run / 'config.json').write_text(json.dumps(config))
    status = json.loads((run / 'status.json').read_text())
    status.update(completed_epochs=50, global_step=cfg.total_steps)
    (run / 'status.json').write_text(json.dumps(status))
    rows = (run / 'metrics.jsonl').read_text().splitlines()[:50]
    (run / 'metrics.jsonl').write_text('\n'.join(rows) + '\n')
    payload = torch.load(completed_run, weights_only=True)
    payload['metadata'].update(canonical=False, pretraining_sweep=True)
    payload['optimization'] = asdict(cfg.optimization())
    torch.save(payload, completed_run)
    selected = json.loads((run / 'best_reconstruction.json').read_text())
    selected.update(checkpoint='best_reconstruction/' + completed_run.name, checkpoint_sha256=sha256_file(completed_run))
    (run / 'best_reconstruction.json').write_text(json.dumps(selected))
    moved = tmp_path / 'downloaded'
    moved.mkdir()
    for name in ('config.json', 'status.json', 'metrics.jsonl', 'best_reconstruction.json'):
        shutil.copy2(run / name, moved / name)
    shutil.copytree(completed_run.parent, moved / 'best_reconstruction')
    checkpoint = moved / 'best_reconstruction' / completed_run.name
    evidence = inspect_observation_checkpoint(checkpoint, require_canonical=True, selection_policy='best-reconstruction')
    assert not evidence.matches_appendix_pretraining_record
    assert not evidence.matches_paper_pretraining_record
    assert evidence.sha256 == selected['checkpoint_sha256']
    status['completed_epochs'] = 49
    (moved / 'status.json').write_text(json.dumps(status))
    with pytest.raises(ValueError, match='completed pretraining'):
        inspect_observation_checkpoint(checkpoint, require_canonical=True, selection_policy='best-reconstruction')
