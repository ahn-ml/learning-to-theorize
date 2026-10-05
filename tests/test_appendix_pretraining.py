import torch
from lightning.fabric import Fabric

from tasks.arithmetic_factorization.observation_pretraining import ArithmeticObservationPretrainingConfig
from tasks.arithmetic_factorization.observation_runner import evaluate_reconstruction, pretraining_parameters
from tasks.arithmetic_factorization.task import ArithmeticObservationModel, build_neo
from tasks.arithmetic_factorization.theorizer_runner import load_observation_model
from tasks.arithmetic_factorization.fabric_output import fabric_output_hook


def test_arithmetic_embeddings_train_and_transfer_under_fabric(tmp_path):
    config = ArithmeticObservationPretrainingConfig()
    params = pretraining_parameters(config, steps_per_epoch=5)
    assert params.total_steps == 2500
    core = ArithmeticObservationModel(params, config)
    core.register_forward_hook(fabric_output_hook)
    before = core.encoder.state_dict()['embedding_encoder.embedding.weight'].clone()
    fabric = Fabric(accelerator='cpu', devices=1, precision='bf16-mixed')
    optimizer = torch.optim.AdamW(core.parameters(), lr=config.learning_rate)
    model, optimizer = fabric.setup(core, optimizer)
    data = torch.randint(0, 10, (2, 4, 1, 4))
    rng = torch.get_rng_state()
    metrics = evaluate_reconstruction(model, [(data[:, :3], data[:, 3], [])], fabric)
    assert torch.equal(rng, torch.get_rng_state())
    assert set(metrics) == {'reconstruction_loss', 'digit_accuracy', 'number_accuracy'}
    assert 0 <= metrics['number_accuracy'] <= metrics['digit_accuracy'] <= 1
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
