from dataclasses import asdict, replace

import pytest
import torch
from lightning.fabric import Fabric

from tasks.gridworld.models.observation_pretraining import (
    GridWorldObservationPretrainingModel, GridWorldObservationPretrainingObjective,
)
from tasks.gridworld.observation_checkpoint import _inspect_release_v1
from tasks.gridworld.observation_pretraining import GridWorldObservationPretrainingConfig
from tasks.gridworld.observation_runner import build_parser, _config_from_arguments
from tasks.arithmetic_factorization.observation_pretraining import observation_pretraining_config
from tasks.arithmetic_factorization.observation_runner import pretraining_parameters
from tasks.arithmetic_factorization.task import ArithmeticObservationModel, build_neo
from tasks.arithmetic_factorization.theorizer_runner import load_observation_model
from tasks.arithmetic_factorization.fabric_output import fabric_output_hook


def test_gridworld_kl_reaches_encoder_gradient_and_zero_preserves_output():
    torch.manual_seed(42)
    model = GridWorldObservationPretrainingModel().eval()
    data = torch.randint(0, 8, (2, 4, 10, 10))
    raw = model(data)
    class Cached(torch.nn.Module):
        def forward(self, unused):
            return raw
    assert GridWorldObservationPretrainingObjective()(Cached(), data) is raw
    adjusted = GridWorldObservationPretrainingObjective(kl_weight=1e-5)(Cached(), data)
    torch.testing.assert_close(adjusted.loss, raw.loss + 1e-5 * raw.kl_loss)
    parameters = list(model.encoder.parameters())
    actual = torch.autograd.grad(adjusted.loss - raw.loss, parameters, retain_graph=True, allow_unused=True)
    expected = torch.autograd.grad(raw.kl_loss, parameters, allow_unused=True)
    nonzero = False
    for a, e in zip(actual, expected):
        if e is not None:
            torch.testing.assert_close(a, e * 1e-5, atol=1e-8, rtol=1e-4)
            nonzero |= bool(a.abs().sum() > 0)
    assert nonzero


@pytest.mark.parametrize('profile,weight', [('recovered', 0.0), ('appendix', 1e-5)])
def test_gridworld_profile_cli_and_handoff_are_explicit(profile, weight):
    args = build_parser().parse_args(['--train-h5', '/train', '--test-h5', '/test',
        '--output-root', '/output', '--pretraining-profile', profile])
    config = _config_from_arguments(args)
    assert config.pretraining.kl_weight == weight
    payload = {'model_state_dict': {'x': torch.ones(1)}, 'optimizer_state_dict': {},
        'scheduler_state_dict': {}, 'training_state': {'completed_epochs': 499, 'global_step': 122255},
        'optimization': asdict(GridWorldObservationPretrainingConfig().optimization()),
        'metadata': {'domain': 'gridworld', 'stage': 'observation_pretraining',
            'position': 'after_eval_before_train', 'evaluated_epoch': 499,
            'historical_checkpoint_number': 122256, 'source_commit': 'a' * 40,
            'canonical': True, 'pretraining_profile': profile, 'kl_weight': weight}}
    result = _inspect_release_v1(payload, 'digest')
    assert result.matches_paper_pretraining_record == (profile == 'recovered')
    assert result.matches_appendix_pretraining_record == (profile == 'appendix')
    payload['metadata']['kl_weight'] = 0.123
    with pytest.raises(ValueError, match='KL weight disagree'):
        _inspect_release_v1(payload, 'digest')


def test_arithmetic_appendix_embeddings_train_and_transfer_under_fabric(tmp_path):
    config = observation_pretraining_config('appendix')
    original = observation_pretraining_config()
    assert config.epochs == 500 and config.minimum_learning_rate_ratio == 0.005
    assert replace(config, epochs=50, minimum_learning_rate_ratio=0.1) == original
    params = pretraining_parameters(config, steps_per_epoch=5)
    assert not params.use_vae and params.total_steps == 2500
    core = ArithmeticObservationModel(params, config=config)
    core.register_forward_hook(fabric_output_hook)
    before = core.encoder.state_dict()['embedding_encoder.embedding.weight'].clone()
    fabric = Fabric(accelerator='cpu', devices=1, precision='bf16-mixed')
    optimizer = torch.optim.AdamW(core.parameters(), lr=config.learning_rate)
    model, optimizer = fabric.setup(core, optimizer)
    data = torch.randint(0, 10, (2, 4, 1, 4))
    from tasks.arithmetic_factorization.theorizer_runner import _evaluate
    metrics = _evaluate(model, [(data[:, :3], data[:, 3], [])], fabric)
    assert 0 <= metrics["transfer_accuracy"] <= 1
    model.train()
    result = model(data)
    assert torch.isfinite(result.loss)
    fabric.backward(result.loss)
    optimizer.step()
    assert not torch.equal(before, core.encoder.state_dict()['embedding_encoder.embedding.weight'])
    path = tmp_path / 'observation.pth'
    torch.save({'model_state_dict': core.state_dict()}, path)
    downstream = build_neo('neo', '0.33')
    loaded = load_observation_model(downstream, path)
    assert set(loaded) == {k for k in core.state_dict() if k.startswith(('encoder.', 'decoder.'))}
    assert all(torch.equal(v, downstream.state_dict()[k]) for k, v in loaded.items())
    assert all(not p.requires_grad for name, p in downstream.named_parameters() if name in loaded)
