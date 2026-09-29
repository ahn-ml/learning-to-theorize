from dataclasses import dataclass

import torch
from lightning.fabric import Fabric

from tasks.arithmetic_factorization.fabric_output import fabric_output_hook


@dataclass(frozen=True, slots=True)
class FrozenResult:
    loss: torch.Tensor
    nested: tuple


class FrozenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor([1.5, -2.0]))

    def forward(self, ignored):
        loss = self.value.to(torch.bfloat16).square().sum()
        self.original = FrozenResult(loss, (
            FrozenResult(loss, (torch.tensor([1, 2]), None)), [loss],
        ))
        return self.original


def test_fabric_forward_and_backward_preserve_frozen_output_values_and_gradients():
    original = FrozenModel()
    keys = set(original.state_dict())
    original.register_forward_hook(fabric_output_hook)
    fabric = Fabric(accelerator="cpu", devices=1, precision="bf16-mixed")
    model = fabric.setup(original)
    result = model(torch.zeros(1))

    assert original.original.loss.dtype == torch.bfloat16
    assert result.loss.dtype == result.nested[0].loss.dtype == torch.float32
    assert result.nested[1][0].dtype == torch.float32
    assert result.nested[0].nested[0].dtype == torch.int64
    assert result.nested[0].nested[1] is None
    assert float(result.loss) == float(original.original.loss)
    assert set(original.state_dict()) == keys
    fabric.backward(result.loss)
    assert torch.equal(original.value.grad, torch.tensor([3.0, -4.0]))


def test_real_training_setup_has_unique_parameters_and_updates_model():
    from tasks.arithmetic_factorization.experiment import get_experiment
    from tasks.arithmetic_factorization.task import build_neo
    from tasks.arithmetic_factorization.theorizer_runner import setup_training

    torch.manual_seed(42)
    original = build_neo("neo", "0.33")
    original.freeze_observation_model()
    fabric = Fabric(accelerator="cpu", devices=1, precision="bf16-mixed")
    model, optimizer = setup_training(fabric, original, get_experiment("neo", "0.33"))
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters})
    before = [p.detach().clone() for p in parameters]
    output = model(torch.randint(0, 10, (2, 4, 1, 6)))
    assert torch.isfinite(output.loss)
    fabric.backward(output.loss)
    optimizer.step()
    assert any(not torch.equal(old, new) for old, new in zip(before, parameters))
