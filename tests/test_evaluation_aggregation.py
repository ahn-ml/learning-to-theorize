"""Episode-weighted evaluation must not depend on batch or rank partitioning."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.multiprocessing import spawn

from tasks.image_editing.theorizer_runner import _loader, evaluate
from training.runtime import DistributedContext


class MetricModel(nn.Module):
    def forward(self, grids, **kwargs):
        value = grids.float().mean()
        return SimpleNamespace(
            metrics=SimpleNamespace(l1=value), query_metrics=SimpleNamespace(l1=value),
            query_reconstruction_loss=value, mean_explanation_length=value,
        )


def metric_loader(values, rank=0, world_size=1, batch_size=2):
    dataset = [(torch.tensor([value]), torch.tensor([value]), ()) for value in values]
    return _loader(dataset, batch_size=batch_size, num_workers=0, shuffle=False, seed=42,
                   context=DistributedContext(rank, rank, world_size, torch.device('cpu')))[0]


def test_uneven_final_batch_is_episode_weighted():
    metrics = evaluate(MetricModel(), metric_loader([0., 0., 3.]), torch.device('cpu'),
                       coefficient=1.0, precision='32-true')
    assert metrics['query_l1'] == pytest.approx(1.0)


def distributed_worker(rank, rendezvous, output, values):
    torch.distributed.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        loader = metric_loader(values, rank, 2)
        result = evaluate(MetricModel(), loader, torch.device('cpu'), coefficient=1.0, precision='32-true')
        torch.save({'result': result, 'indices': list(loader.dataset.indices)}, f'{output}/{rank}.pt')
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize('values', [[0., 1., 2., 3., 9.], [7.]])
def test_distributed_evaluation_matches_whole_dataset(tmp_path, values):
    spawn(distributed_worker, args=(f'file://{tmp_path}/rendezvous', str(tmp_path), values), nprocs=2)
    records = [torch.load(tmp_path / f'{rank}.pt', weights_only=False) for rank in range(2)]
    assert sorted(i for record in records for i in record['indices']) == list(range(len(values)))
    for record in records:
        assert record['result']['query_l1'] == pytest.approx(sum(values) / len(values))
