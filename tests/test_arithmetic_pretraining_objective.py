"""Observation pretraining cannot use transitions or paired targets."""
import json

import h5py
import pytest
import torch
from torch.nn import functional as F

from tasks.arithmetic_factorization.data.observations import generate_observations, build_observation_loader
from tasks.arithmetic_factorization.observation_pretraining import observation_pretraining_config
from tasks.arithmetic_factorization.observation import ArithmeticObservationModel
from tasks.arithmetic_factorization.observation_runner import select_reconstruction_checkpoint


def test_pretraining_is_only_digit_reconstruction():
    model = ArithmeticObservationModel(observation_pretraining_config())
    assert set(dict(model.named_children())) == {"encoder", "decoder"}
    assert len(model.state_dict()) == 3
    observations = torch.tensor([[[0, 1, 2, 3]], [[9, 8, 7, 6]]])
    output = model(observations)
    states, _ = model.encoder(observations)
    expected = F.cross_entropy(model.decoder(states).reshape(-1, 10), observations.reshape(-1))
    assert torch.equal(output["loss"], expected)
    actual_grad = torch.autograd.grad(output["loss"], tuple(model.parameters()))
    expected_grad = torch.autograd.grad(expected, tuple(model.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(actual_grad, expected_grad))
    with pytest.raises(ValueError, match="individual numbers"):
        model(observations.unsqueeze(1).repeat(1, 4, 1, 1))


def test_observations_have_no_programs_or_pairs(tmp_path):
    path = tmp_path / "numbers.h5"
    generate_observations(path)
    with h5py.File(path) as file:
        assert list(file) == ["observations"]
        assert file["observations"].shape == (10000, 1, 4)
    loader = build_observation_loader(path, batch_size=512, shuffle=False, seed=42)
    values = torch.cat(list(loader))
    numbers = (values[:, 0] * torch.tensor([1000, 100, 10, 1])).sum(1)
    assert torch.equal(numbers, torch.arange(10000))
    with pytest.raises(FileExistsError):
        generate_observations(path)


def test_paired_artifact_is_not_accepted_for_pretraining(tmp_path):
    path = tmp_path / "pairs.h5"
    with h5py.File(path, "w") as file:
        file.create_group("sample_0")
    with pytest.raises(KeyError):
        build_observation_loader(path, batch_size=8, shuffle=True, seed=42)


def test_reconstruction_selection_keeps_first_accuracy_maximum(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    records = []
    for step, accuracy, loss in ((0, 0.1, 2.0), (25, 0.7, 1.2), (50, 1.0, 0.8), (75, 1.0, 0.01)):
        if step:
            torch.save({"step": step}, checkpoints / f"checkpoint_{step}.pth")
        records.append({"global_step": step, "completed_epochs": step // 5,
                        "reconstruction/number_accuracy": accuracy, "reconstruction/loss": loss})
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in records))
    selected = select_reconstruction_checkpoint(tmp_path, 75)
    assert selected["checkpoint"] == "checkpoints/checkpoint_50.pth"
    assert selected["score"] == 1.0 and selected["global_step"] == 50
    assert torch.load(tmp_path / selected["checkpoint"], weights_only=True)["step"] == 50


def test_final_reconstruction_checkpoint_can_win_between_save_intervals(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    torch.save({"step": 25}, checkpoints / "checkpoint_25.pth")
    torch.save({"step": 30}, checkpoints / "checkpoint_final.pth")
    records = [{"global_step": step, "completed_epochs": step // 5,
                "reconstruction/number_accuracy": accuracy} for step, accuracy in ((25, 0.9), (30, 1.0))]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in records))
    selected = select_reconstruction_checkpoint(tmp_path, 30)
    assert selected["checkpoint"] == "checkpoints/checkpoint_final.pth"
    assert selected["score"] == 1.0 and selected["completed_epochs"] == 6
