"""Single-grid GridWorld observation pretraining: loader, objective, recipe and runner."""
import json
from dataclasses import asdict, replace

import h5py
import numpy as np
import pytest
import torch

from tasks.gridworld.data.dataset import (
    GridWorldHDF5Dataset,
    GridWorldObservationLoaderConfig,
    GridWorldSingleObservationLoader,
)
from tasks.gridworld.models.vae import VAE
from tasks.gridworld.observation_checkpoint import load_gridworld_observation_checkpoint
from tasks.gridworld.observation_pretraining import (
    GridWorldObservationObjective,
    GridWorldObservationPretrainingConfig,
)
from tasks.gridworld.observation_runner import (
    GridWorldObservationRunConfig,
    _config_from_arguments,
    build_parser,
    run_gridworld_observation_pretraining,
)
from tasks.gridworld.task import build_neo


def write_artifact(path, episodes, blank=False):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as file:
        file.attrs["dataset_length"] = episodes
        file.attrs["batch_size"] = 1
        file.attrs["shuffle"] = False
        for index in range(episodes):
            group = file.create_group(f"sample_{index}")
            grids = rng.integers(0, 9, (3, 10, 10))
            answer = rng.integers(0, 9, (10, 10))
            group["grids"] = (0 * grids if blank else grids).astype(np.int32)
            group["answer"] = (0 * answer if blank else answer).astype(np.int64)


def test_loader_pools_every_grid_once_per_epoch(tmp_path):
    path = tmp_path / "train.h5"
    write_artifact(path, 10)
    dataset = GridWorldHDF5Dataset(path)
    seen = []
    for rank in range(2):
        loader = GridWorldSingleObservationLoader(
            dataset, GridWorldObservationLoaderConfig(per_rank_batch_size=2, world_size=2, rank=rank, num_workers=0))
        loader.set_epoch(3)
        batches = list(loader)
        assert len(loader) == 3 and all(b.shape[1:] == (4, 10, 10) for b in batches)
        seen.append(torch.cat([b.reshape(-1, 10, 10) for b in batches]))
    pooled = torch.cat(seen)
    expected = loader.grids.long()
    assert pooled.shape == expected.shape
    key = lambda g: sorted(map(bytes, g.to(torch.uint8).numpy()))
    assert key(pooled) == key(expected)


def test_loader_breaks_episode_grouping(tmp_path):
    path = tmp_path / "train.h5"
    write_artifact(path, 64)
    loader = GridWorldSingleObservationLoader(
        GridWorldHDF5Dataset(path), GridWorldObservationLoaderConfig(per_rank_batch_size=16, num_workers=0))
    first = next(iter(loader))[0].reshape(-1, 10, 10).to(torch.uint8)
    episode_of = {bytes(g.numpy()): i // 4 for i, g in enumerate(loader.grids)}
    assert len({episode_of[bytes(g.numpy())] for g in first}) > 1


@pytest.mark.parametrize("episodes,world_size,batch", [(14, 5, 2), (23, 8, 1), (10, 2, 2)])
def test_every_rank_gets_the_same_number_of_full_batches(tmp_path, episodes, world_size, batch):
    path = tmp_path / "train.h5"
    write_artifact(path, episodes)
    dataset = GridWorldHDF5Dataset(path)
    counts = []
    for rank in range(world_size):
        loader = GridWorldSingleObservationLoader(dataset, GridWorldObservationLoaderConfig(
            per_rank_batch_size=batch, world_size=world_size, rank=rank, num_workers=0))
        batches = list(loader)
        assert len(batches) == len(loader)
        counts.append([b.shape[0] for b in batches])
    assert all(c == counts[0] for c in counts)


def test_objective_is_reconstruction_plus_weighted_kl():
    torch.manual_seed(0)
    model = VAE().eval()
    batch = torch.randint(0, 9, (3, 4, 10, 10))
    output = GridWorldObservationObjective(kl_weight=1e-5)(model, batch)
    assert output.num_observations == 12
    torch.testing.assert_close(output.loss, output.reconstruction_loss + 1e-5 * output.kl_loss)
    parameters = list(model.encoder.parameters())
    kl = torch.autograd.grad(output.kl_loss, parameters, retain_graph=True, allow_unused=True)
    assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in kl)


def test_cli_runs_the_release_recipe():
    arguments = build_parser().parse_args(
        ["--train-h5", "a.h5", "--test-h5", "b.h5", "--output-root", "out"])
    config = _config_from_arguments(arguments)
    assert config.pretraining == GridWorldObservationPretrainingConfig()
    recipe = asdict(config.pretraining)
    assert {key: recipe[key] for key in ("epochs", "learning_rate", "kl_weight", "warmup_ratio",
            "minimum_learning_rate_ratio", "weight_decay", "per_rank_batch_size",
            "effective_world_size")} == {
        "epochs": 50, "learning_rate": 0.0007, "kl_weight": 1e-5, "warmup_ratio": 0.1,
        "minimum_learning_rate_ratio": 0.005, "weight_decay": 0.01, "per_rank_batch_size": 512,
        "effective_world_size": 4}
    assert config.pretraining.total_steps == 12250
    for removed in (["--pretraining-profile", "appendix"], ["--observation-unit", "single"],
                    ["--development"], ["--wandb-mode", "disabled"]):
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["--train-h5", "a.h5", "--test-h5", "b.h5", "--output-root", "out", *removed])


def test_cpu_run_writes_the_handoff_records(tmp_path):
    train, test = tmp_path / "train.h5", tmp_path / "test.h5"
    write_artifact(train, 8, blank=True)
    write_artifact(test, 4, blank=True)
    pretraining = replace(
        GridWorldObservationPretrainingConfig(), epochs=3, train_episodes=8, test_episodes=4,
        per_rank_batch_size=2, effective_world_size=1, data_loader_workers=0,
        log_interval_steps=2, learning_rate=0.05, warmup_ratio=0.0)
    result = run_gridworld_observation_pretraining(GridWorldObservationRunConfig(
        train_h5=train, test_h5=test, output_root=tmp_path / "runs", wandb_mode="disabled",
        pretraining=pretraining))
    run = result.output_directory
    assert result.state.completed_epochs == 3 and result.state.global_step == 12
    status = json.loads((run / "status.json").read_text())
    assert status["status"] == "completed" and status["global_step"] == 12
    assert json.loads((run / "config.json").read_text())["pretraining"] == asdict(pretraining)
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in rows if "eval/grid_accuracy" in row] == [1, 2, 3]
    assert not any(key.startswith("legacy") for row in rows for key in row)
    selected = json.loads((run / "best_reconstruction.json").read_text())
    checkpoint = run / selected["checkpoint"]
    assert selected["checkpoint"].startswith("best_reconstruction/checkpoint_")
    assert checkpoint.is_file() and selected["score"] > 0
    model = build_neo("alpha-0.33")
    loaded = load_gridworld_observation_checkpoint(model, checkpoint)
    assert loaded == tuple(sorted(VAE().state_dict()))
