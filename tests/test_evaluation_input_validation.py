"""Reject mislabeled checkpoints and incomplete or invalid evaluation results."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tasks.image_editing import evaluation as image
from tasks.image_editing.data.dataset import EpisodeBatch

CONDITION = {'method': 'neo', 'alpha': '0.33', 'seed': 42, 'training.length_control_start': 1.01,
             'training.length_control_scheduling': False}


class ImageModel(nn.Module):
    def __init__(self, value=0.25):
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self.value = value

    def forward(self, grids, **kwargs):
        metrics = SimpleNamespace(l1=self.value)
        return SimpleNamespace(metrics=metrics, query_metrics=metrics, mean_explanation_length=1.0)


def image_case(tmp_path, monkeypatch, *, stored=CONDITION, count=3, value=0.25):
    model = ImageModel(value)
    checkpoint = tmp_path / 'model.pt'
    torch.save({'model_state_dict': model.state_dict(), 'config': stored}, checkpoint)
    monkeypatch.setattr(image, 'build_neo', lambda _: model)
    monkeypatch.setattr(image, 'ImageEditingDataset', lambda _: [torch.zeros(1)] * count)
    monkeypatch.setattr(image, 'collate_episodes', lambda rows: EpisodeBatch(torch.stack(rows)))
    return image.EvaluationRunConfig(alpha='0.33', seed=42, protocol='id', checkpoint=checkpoint,
                                     data_h5=tmp_path / 'data.h5', device='cpu')


@pytest.mark.parametrize('changed', [{'seed': 43}, {'alpha': '0.66'}, {'method': 'other'}])
def test_image_rejects_checkpoint_from_another_condition(tmp_path, monkeypatch, changed):
    config = image_case(tmp_path, monkeypatch, stored={**CONDITION, **changed})
    with pytest.raises(ValueError, match='checkpoint.*condition'):
        image.evaluate_checkpoint(config)


def test_image_accepts_matching_checkpoint(tmp_path, monkeypatch):
    result = image.evaluate_checkpoint(image_case(tmp_path, monkeypatch))
    assert result['episodes'] == 3 and result['query_l1'] == 0.25
    assert result['length_control_coefficient'] == 1.01


@pytest.mark.parametrize('count,value', [(0, 0.25), (2, float('nan')), (2, float('inf'))])
def test_image_rejects_empty_or_nonfinite_results(tmp_path, monkeypatch, count, value):
    config = image_case(tmp_path, monkeypatch, count=count, value=value)
    with pytest.raises(ValueError, match='empty|non-finite'):
        image.evaluate_checkpoint(config)
