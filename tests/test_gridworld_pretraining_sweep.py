import json
from dataclasses import asdict

import pytest

from tasks.gridworld.observation_runner import build_parser, _config_from_arguments


def parse(tmp_path, overrides=None):
    args = ["--train-h5", "train.h5", "--test-h5", "test.h5", "--output-root", str(tmp_path),
            "--pretraining-profile", "appendix"]
    if overrides is not None:
        path = tmp_path / "override.json"
        path.write_text(json.dumps(overrides))
        args += ["--pretraining-overrides", str(path)]
    return _config_from_arguments(build_parser().parse_args(args))


def test_sweep_preserves_topology_and_labels_nonpaper(tmp_path):
    baseline = parse(tmp_path)
    sweep = parse(tmp_path, {"kl_weight": 0, "epochs": 50})
    assert baseline.canonical and not baseline.pretraining_sweep
    assert not sweep.canonical and sweep.pretraining_sweep
    before, after = asdict(baseline.pretraining), asdict(sweep.pretraining)
    assert {key for key in before if before[key] != after[key]} == {"kl_weight", "epochs"}
    assert after["effective_world_size"] == 4
    assert after["train_episodes"] == 500000


@pytest.mark.parametrize("values", [{"effective_world_size": 1}, {"epochs": True},
    {"epochs": 1.5}, {"kl_weight": -1}, {"learning_rate": 0},
    {"kl_weight": float("nan")}, {"minimum_learning_rate_ratio": 2}, {}])
def test_invalid_overrides_rejected(tmp_path, values):
    with pytest.raises(ValueError):
        parse(tmp_path, values)


def test_sweep_handoff_cannot_be_mislabeled_as_paper(tmp_path):
    import torch
    from tasks.gridworld.observation_checkpoint import inspect_observation_checkpoint
    config = parse(tmp_path, {"kl_weight": 1e-7, "epochs": 50})
    payload = {"format_version": 1, "model_state_dict": {"x": torch.ones(1)},
        "optimizer_state_dict": {}, "scheduler_state_dict": {},
        "training_state": {"completed_epochs": 49, "global_step": 12005},
        "optimization": asdict(config.pretraining.optimization()),
        "metadata": {"domain": "gridworld", "stage": "observation_pretraining",
            "position": "after_eval_before_train", "evaluated_epoch": 49,
            "historical_checkpoint_number": 12006, "source_commit": "a" * 40,
            "canonical": False, "pretraining_sweep": True,
            "pretraining_profile": "appendix", "kl_weight": 1e-7}}
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    evidence = inspect_observation_checkpoint(path, require_canonical=False)
    assert evidence.kl_weight == 1e-7
    assert not evidence.matches_paper_pretraining_record
    assert not evidence.matches_appendix_pretraining_record
    with pytest.raises(ValueError, match="canonical theorizer"):
        inspect_observation_checkpoint(path, require_canonical=True)
    payload["metadata"]["canonical"] = True
    torch.save(payload, path)
    with pytest.raises(ValueError, match="sweep metadata"):
        inspect_observation_checkpoint(path, require_canonical=False)


def test_sensitivity_run_records_and_checks_canonical_data_hashes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tasks.gridworld import observation_runner as runner
    train, test = tmp_path / 'train.h5', tmp_path / 'test.h5'
    train.write_bytes(b'train'); test.write_bytes(b'test')
    monkeypatch.setattr(runner, 'PAPER_TRAIN_ARTIFACT_SHA256', runner.sha256_file(train))
    monkeypatch.setattr(runner, 'PAPER_TEST_ARTIFACT_SHA256', runner.sha256_file(test))
    monkeypatch.setattr(runner, 'broadcast_object', lambda *args: None)
    config = SimpleNamespace(train_h5=train, test_h5=test, canonical=False, pretraining_sweep=True)
    context = SimpleNamespace(is_global_zero=True)
    result = runner._prepare_artifact_evidence(config, context)
    assert result['train']['sha256'] == runner.sha256_file(train)
    assert result['test']['sha256'] == runner.sha256_file(test)
    train.write_bytes(b'different')
    with pytest.raises(RuntimeError, match='training HDF5'):
        runner._prepare_artifact_evidence(config, context)
