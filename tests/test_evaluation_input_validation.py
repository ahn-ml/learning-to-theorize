"""Reject mislabeled checkpoints and incomplete or invalid evaluation results."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tasks.image_editing import evaluation as image
from tasks.image_editing.data.dataset import EpisodeBatch


class ImageModel(nn.Module):
    def __init__(self, value=0.25):
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self.value = value

    def forward(self, grids, **kwargs):
        metrics = SimpleNamespace(l1=self.value, lpips=0.0)
        return SimpleNamespace(metrics=metrics, query_metrics=metrics)


def image_case(tmp_path, monkeypatch, *, stored=None, count=3, value=0.25):
    model = ImageModel(value)
    payload = {'model_state_dict': model.state_dict()}
    if stored is not None:
        payload['config'] = stored
    checkpoint = tmp_path / 'model.pt'
    torch.save(payload, checkpoint)
    monkeypatch.setattr(image, 'build_neo', lambda _: model)
    monkeypatch.setattr(image, 'ImageEditingDataset', lambda _: [torch.zeros(1)] * count)
    monkeypatch.setattr(image, 'collate_episodes', lambda rows: EpisodeBatch(
        torch.stack(rows), torch.stack(rows), ()))
    return image.EvaluationRunConfig(method='neo', alpha='0.33', seed=42,
        protocol='id', checkpoint=checkpoint, data_h5=tmp_path / 'data.h5',
        device='cpu', batch_size=2, num_workers=0)


@pytest.mark.parametrize('changed', [{'seed': 43}, {'alpha': '0.66'}, {'method': 'other'}])
def test_image_rejects_checkpoint_from_another_condition(tmp_path, monkeypatch, changed):
    stored = {'method': 'neo', 'alpha': '0.33', 'seed': 42, **changed}
    config = image_case(tmp_path, monkeypatch, stored=stored)
    with pytest.raises(ValueError, match='checkpoint.*condition'):
        image.evaluate_checkpoint(config)


def test_image_accepts_matching_checkpoint_and_raw_weights(tmp_path, monkeypatch):
    config = image_case(tmp_path, monkeypatch,
                        stored={'method': 'neo', 'alpha': '0.33', 'seed': 42})
    assert image.evaluate_checkpoint(config)['episodes'] == 3
    config = image_case(tmp_path, monkeypatch)
    assert image.evaluate_checkpoint(config)['query_l1'] == 0.25


@pytest.mark.parametrize('count,value', [(0, 0.25), (2, float('nan')), (2, float('inf'))])
def test_image_rejects_empty_or_nonfinite_results(tmp_path, monkeypatch, count, value):
    config = image_case(tmp_path, monkeypatch, count=count, value=value)
    with pytest.raises(ValueError, match='empty|non-finite'):
        image.evaluate_checkpoint(config)
