import torch

from models.neo import NEO
from tasks.arithmetic_factorization.task import ArithmeticRollout, build_neo


def test_grounding_is_independent_of_explanation_length_and_matches_gridworld():
    model = build_neo("neo", "0.66")
    distances = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], requires_grad=True)
    short = torch.tensor([1, 1])
    mixed = torch.tensor([1, 3])
    loss = model.reduce_consistency_loss(distances, mixed)
    assert torch.equal(loss, torch.tensor(10.5))
    assert torch.equal(loss, model.reduce_consistency_loss(distances, short))
    assert torch.equal(loss, NEO.reduce_consistency_loss(model, distances, mixed))
    loss.backward()
    # Later rollout states are still grounded, with equal per-episode weight.
    assert torch.equal(distances.grad, torch.full_like(distances, 0.5))


def test_one_step_arithmetic_objective_is_unchanged():
    distances = torch.tensor([[2., 4.]], requires_grad=True)
    lengths = torch.ones(2, dtype=torch.long)
    loss = ArithmeticRollout.reduce_consistency_loss(None, distances, lengths)
    assert torch.equal(loss, (distances / lengths.unsqueeze(0)).sum(0).mean())
    loss.backward()
    assert torch.equal(distances.grad, torch.full_like(distances, 0.5))
