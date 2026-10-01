"""Observation pretraining cannot use transitions or paired targets."""
import h5py
import pytest
import torch
from torch.nn import functional as F

from tasks.arithmetic_factorization.data.observations import generate_observations, build_observation_loader
from tasks.arithmetic_factorization.observation_pretraining import observation_pretraining_config
from tasks.arithmetic_factorization.observation import ArithmeticObservationModel


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
