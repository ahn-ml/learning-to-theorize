"""Image evaluation honors tracking options and closes failed runs."""
import json
import sys
from types import SimpleNamespace

import pytest

from tasks.image_editing import evaluation


@pytest.mark.parametrize('fails', [False, True])
def test_evaluation_tracks_requested_destination_and_exit(tmp_path, monkeypatch, fails):
    calls = {}
    def init(**kwargs):
        calls['init'] = kwargs
        return SimpleNamespace(url='offline')
    wandb = SimpleNamespace(init=init, log=lambda metrics: calls.update(metrics=metrics),
                            finish=lambda **kwargs: calls.update(finish=kwargs))
    monkeypatch.setitem(sys.modules, 'wandb', wandb)
    def evaluate(config):
        assert config.method == 'neo'
        assert config.hard_grounding is False
        if fails:
            raise RuntimeError('evaluation failed')
        return {'query_l1': 0.1, 'episodes': 2, 'method': 'neo'}
    monkeypatch.setattr(evaluation, 'evaluate_checkpoint', evaluate)
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'test input')
    output = tmp_path / 'results.json'
    args = ['--method', 'neo', '--alpha', '0.33', '--protocol', 'id',
            '--checkpoint', str(checkpoint), '--data-root', str(tmp_path),
            '--output', str(output), '--wandb-mode', 'offline',
            '--wandb-project', 'chosen', '--wandb-entity', 'owner', '--wandb-group', 'group']
    if fails:
        with pytest.raises(RuntimeError, match='evaluation failed'):
            evaluation.main(args)
        assert not output.exists()
    else:
        assert evaluation.main(args) == 0
        assert json.loads(output.read_text())['query_l1'] == 0.1
        assert json.loads(output.with_suffix('.metrics.jsonl').read_text()) == calls['metrics']
    assert calls['init']['project'] == 'chosen'
    assert calls['init']['entity'] == 'owner'
    assert calls['init']['group'] == 'group'
    assert calls['init']['mode'] == 'offline'
    assert calls['finish']['exit_code'] == int(fails)


def test_tracking_requires_a_log_destination(tmp_path):
    with pytest.raises(SystemExit):
        evaluation.main(['--method', 'neo', '--alpha', '0.33', '--protocol', 'id',
                         '--checkpoint', str(tmp_path / 'model.pt'), '--data-root', str(tmp_path),
                         '--wandb-mode', 'offline'])
