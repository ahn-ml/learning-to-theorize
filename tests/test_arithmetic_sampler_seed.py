"""Runner seeds must reach Fabric's automatically injected sampler."""
import os
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


@pytest.mark.parametrize('stage,seed', [('observation', 42), ('theorizer', 43)])
@pytest.mark.parametrize('inherited_seed', [None, '7'])
def test_runner_sets_distributed_sampler_seed(monkeypatch, tmp_path, stage, seed, inherited_seed):
    fabric_module = pytest.importorskip('lightning.fabric')
    real_fabric = fabric_module.Fabric
    from tasks.arithmetic_factorization import observation_runner, theorizer_runner
    from tasks.arithmetic_factorization.data import dataset
    from training import runtime

    if inherited_seed is None:
        monkeypatch.delenv('PL_GLOBAL_SEED', raising=False)
    else:
        monkeypatch.setenv('PL_GLOBAL_SEED', inherited_seed)
    monkeypatch.setenv('PL_SEED_WORKERS', '0')
    monkeypatch.setattr(runtime, 'capture_git_source', lambda _: SimpleNamespace(
        commit='test-source', clean=True, pushed=True))

    class LoaderReached(Exception):
        pass

    class InspectFabric:
        def __init__(self, **kwargs):
            pass

        def launch(self):
            pass

        def seed_everything(self, value):
            real_fabric.seed_everything(value)

        def setup_dataloaders(self, loader):
            # Use Fabric's real sampler construction for a four-rank layout.
            actual = real_fabric._get_distributed_sampler(loader, num_replicas=4, rank=0)
            expected = DistributedSampler(loader.dataset, num_replicas=4, rank=0,
                                          shuffle=True, seed=seed)
            assert actual.seed == seed
            assert list(actual) == list(expected)
            assert os.environ['PL_GLOBAL_SEED'] == str(seed)
            raise LoaderReached

    def loader(*args, **kwargs):
        return DataLoader(TensorDataset(torch.arange(128)), batch_size=8,
                          shuffle=kwargs['shuffle'],
                          generator=torch.Generator().manual_seed(kwargs['seed']))

    monkeypatch.setattr(fabric_module, 'Fabric', InspectFabric)
    monkeypatch.setattr(dataset, 'build_dataloader', loader)
    argv = ['--train-h5', str(tmp_path/'train.h5'), '--test-h5', str(tmp_path/'test.h5'),
            '--output-root', str(tmp_path/'output')]
    if stage == 'theorizer':
        argv += ['--method', 'neo', '--alpha', '0.33', '--seed', str(seed),
                 '--observation-checkpoint', str(tmp_path/'observation.pt')]
    runner = observation_runner if stage == 'observation' else theorizer_runner
    with pytest.raises(LoaderReached):
        runner.main(argv)
