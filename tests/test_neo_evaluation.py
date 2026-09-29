import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch

from tasks import neo_evaluation as evaluation


def fake_selection(monkeypatch, tmp_path):
    data = tmp_path / 'id.h5'
    with h5py.File(data, 'w') as f:
        f.attrs['dataset_length'] = 2
        for i in range(2):
            sample = f.create_group(f'sample_{i}')
            sample['grids'] = np.zeros((3, 1, 6), dtype=np.uint8)
            sample['answer'] = np.zeros((1, 6), dtype=np.uint8)
    checkpoints = tmp_path / 'checkpoints'
    checkpoints.mkdir()
    for step in (1, 2, 3):
        (checkpoints / f'{step}.pth').write_text(str(step))
    model = {'step': 0}
    monkeypatch.setattr(evaluation, 'build_model', lambda *a: model)
    def load(model, domain, alpha, seed, path):
        model['step'] = int(path.stem)
        return {'global_step': model['step'], 'coefficient': 1.0}
    monkeypatch.setattr(evaluation, 'load_candidate', load)
    def score(model, domain, episodes, **kwargs):
        return {'score': min(model['step'], 2) / 2, 'metric': 'query', 'direction': 'max',
                'episodes': len(episodes), 'split': 'id', 'hard_grounding': True}
    monkeypatch.setattr(evaluation, 'score_id', score)
    return dict(domain='arithmetic', alpha='0.33', seed=42, directory=checkpoints,
                id_path=data, output=tmp_path / 'selection', device='cpu',
                tracking={'project': 'test', 'mode': 'disabled'}, source={'commit': 'test'})


def test_selects_all_id_candidates_caches_and_rejects_changed_inventory(monkeypatch, tmp_path):
    kwargs = fake_selection(monkeypatch, tmp_path)
    winner = evaluation.select_checkpoint(**kwargs)
    assert winner['selection']['global_step'] == 3
    assert winner['candidate_count'] == 3
    monkeypatch.setattr(evaluation, 'score_id', lambda *a, **kw: pytest.fail('cache rescored'))
    assert evaluation.select_checkpoint(**kwargs) == winner
    (kwargs['directory'] / '4.pth').write_text('4')
    with pytest.raises(ValueError, match='changed'):
        evaluation.select_checkpoint(**kwargs)


def test_failed_sweep_never_publishes_winner(monkeypatch, tmp_path):
    kwargs = fake_selection(monkeypatch, tmp_path)
    def fail(*a, **kw):
        raise RuntimeError('evaluation failed')
    monkeypatch.setattr(evaluation, 'score_id', fail)
    with pytest.raises(RuntimeError):
        evaluation.select_checkpoint(**kwargs)
    assert not (kwargs['output'] / 'selected.json').exists()


def test_frozen_selection_rejects_mismatched_seed_hash_and_ood(monkeypatch, tmp_path):
    kwargs = fake_selection(monkeypatch, tmp_path)
    winner = evaluation.select_checkpoint(**kwargs)
    path = kwargs['output'] / 'selected.json'
    request = dict(domain='arithmetic', alpha='0.33', seed=42, id_path=kwargs['id_path'], device='cpu')
    assert evaluation.read_selection(path, **request)['checkpoint'] == winner['checkpoint']
    with pytest.raises(ValueError, match='different model'):
        evaluation.read_selection(path, **{**request, 'seed': 43})
    changed = json.loads(path.read_text())
    changed['selection']['split'] = 'length-ood'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='ID'):
        evaluation.read_selection(path, **request)
    changed = json.loads(json.dumps(winner))
    changed['selection']['hard_grounding'] = False
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='grounding setting'):
        evaluation.read_selection(path, **request)
    path.write_text(json.dumps(winner))
    Path(winner['checkpoint']).write_text('changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        evaluation.read_selection(path, **request)


@pytest.mark.parametrize('domain,grounding', [('gridworld', True), ('arithmetic', True), ('image', False)])
def test_final_evaluation_rejects_different_selection_grounding(domain, grounding, tmp_path):
    with pytest.raises(ValueError, match='same grounding setting'):
        evaluation.evaluate_selected(domain=domain, alpha='0.33', seed=42,
            selected={'selection': {'hard_grounding': not grounding}},
            data_root=tmp_path, output=tmp_path / 'out', protocol='standard', device='cpu',
            tracking={}, source={})


@pytest.mark.parametrize("source_state", ["pushed", "local", "archive"])
def test_main_freezes_selection_before_resolving_ood(monkeypatch, tmp_path, source_state):
    stages = []
    from training.runtime import GitSourceState
    source = evaluation_source() if source_state == "pushed" else GitSourceState(
        None, None, None, None, False, False, None
    ) if source_state == "archive" else GitSourceState(
        '/source', 'test', 'local', None, False, False, 'working-tree-fingerprint'
    )
    monkeypatch.setattr(evaluation, 'capture_git_source', lambda _: source)
    def data_path(domain, alpha, split, root):
        assert split == 'id'
        stages.append('id')
        return root / 'id.h5'
    monkeypatch.setattr(evaluation, 'data_path', data_path)
    def select(**kwargs):
        stages.append('selection_complete')
        return {'checkpoint': 'chosen'}
    monkeypatch.setattr(evaluation, 'select_checkpoint', select)
    def evaluate(**kwargs):
        assert stages == ['id', 'selection_complete']
        assert kwargs['selected']['checkpoint'] == 'chosen'
        stages.append('final')
    monkeypatch.setattr(evaluation, 'evaluate_selected', evaluate)
    assert evaluation.main(['--task', 'gridworld', '--alpha', '0.33', '--seed', '42',
        '--data-root', str(tmp_path), '--checkpoint-directory', str(tmp_path),
        '--output-root', str(tmp_path / 'out'), '--device', 'cpu']) == 0
    assert stages == ['id', 'selection_complete', 'final']


def evaluation_source():
    from training.runtime import GitSourceState
    return GitSourceState('/source', 'test', 'main', 'https://example.test/repo', True, True, None)


def test_public_script_routes_neo_to_id_selection(monkeypatch, tmp_path):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('public_evaluate', scripts / 'evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'load_task_config', lambda *args: {})
    calls = []
    monkeypatch.setattr(evaluation, 'main', lambda argv: calls.append(('current', argv)) or 0)
    argv = ['evaluate.py', '--task', 'arithmetic_factorization', '--config', 'reproduction.yaml',
            '--method', 'neo', '--seed', '42', '--devices', '0', '--data-root', str(tmp_path),
            '--training-root', str(tmp_path), '--output-root', str(tmp_path / 'out')]
    monkeypatch.setattr(sys, 'argv', argv)
    assert module.main() == 0 and calls[-1][0] == 'current'



@pytest.mark.parametrize('domain', ['arithmetic', 'image'])
def test_checkpoint_metadata_rejects_wrong_seed(domain, monkeypatch, tmp_path):
    from tasks.checkpoint_selection import load_candidate
    if domain == 'image':
        from tasks.image_editing import evaluation as image_evaluation
        def fake_load(model, path, metadata):
            metadata.update({'global_step': 10, 'config': {'method': 'neo', 'alpha': '0.33', 'seed': 43}})
        monkeypatch.setattr(image_evaluation, 'load_model_checkpoint', fake_load)
        monkeypatch.setattr(image_evaluation, 'checkpoint_length_control', lambda _: 1.0)
        model = object()
    else:
        from tasks.arithmetic_factorization.task import build_neo
        model = build_neo('neo', '0.33')
        torch.save({'model_state_dict': model.state_dict(), 'global_step': 10,
                    'config': {'method': 'neo', 'alpha': '0.33', 'seed': 43}}, tmp_path / 'checkpoint.pth')
    with pytest.raises(ValueError, match='metadata'):
        load_candidate(model, domain, '0.33', 42, tmp_path / 'checkpoint.pth')


def test_portable_selection_keeps_hash_checks(monkeypatch, tmp_path):
    kwargs = fake_selection(monkeypatch, tmp_path)
    winner = evaluation.select_checkpoint(**kwargs)
    directory = tmp_path / 'downloaded'
    directory.mkdir()
    source = Path(winner['checkpoint'])
    copied = directory / source.name
    copied.write_bytes(source.read_bytes())
    winner['checkpoint'] = copied.name
    winner['selection']['checkpoint'] = copied.name
    record = directory / 'selected.json'
    record.write_text(json.dumps(winner))
    request = dict(domain='arithmetic', alpha='0.33', seed=42, id_path=kwargs['id_path'], device='cpu')
    assert evaluation.read_selection(record, **request)['checkpoint'] == str(copied)
    copied.write_text('corrupt')
    with pytest.raises(ValueError, match='hash mismatch'):
        evaluation.read_selection(record, **request)


def test_gridworld_task_evaluator_resolves_neo_artifacts():
    from tasks.gridworld.theorizer_evaluation import evaluation_artifact

    for alpha in ('0.33', '0.66', '1.00'):
        experiment = f'alpha-{alpha}'
        assert evaluation_artifact(experiment, 'id').episodes == 5000
        assert evaluation_artifact(experiment, 'id').max_steps == 4
        assert evaluation_artifact(experiment, 'length-ood').episodes == 10000
        assert evaluation_artifact(experiment, 'length-ood').max_steps == 10
        if alpha != '1.00':
            assert evaluation_artifact(experiment, 'compositional-ood').episodes == 5000
        else:
            with pytest.raises(ValueError, match='no held-out compositional split'):
                evaluation_artifact(experiment, 'compositional-ood')
