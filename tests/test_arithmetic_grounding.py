import torch

from models.neo import NEO
from tasks.arithmetic_factorization.task import ArithmeticRollout, build_neo


def test_grounding_divides_each_episode_by_its_selected_length():
    model = build_neo("neo", "0.66")
    distances = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], requires_grad=True)
    short = torch.tensor([1, 1])
    mixed = torch.tensor([1, 3])
    loss = model.reduce_consistency_loss(distances, mixed)
    # Episode 0: (1 + 3 + 5) / 1; episode 1: (2 + 4 + 6) / 3; batch mean.
    assert torch.equal(loss, torch.tensor(6.5))
    # With every program of length one the published and shared reductions agree.
    assert torch.equal(model.reduce_consistency_loss(distances, short),
                       NEO.reduce_consistency_loss(model, distances, short))
    loss.backward()
    expected = torch.tensor([[.5, 1 / 6]] * 3)
    torch.testing.assert_close(distances.grad, expected)


def test_one_step_arithmetic_objective_is_unchanged():
    distances = torch.tensor([[2., 4.]], requires_grad=True)
    lengths = torch.ones(2, dtype=torch.long)
    loss = ArithmeticRollout.reduce_consistency_loss(None, distances, lengths)
    assert torch.equal(loss, (distances / lengths.unsqueeze(0)).sum(0).mean())
    loss.backward()
    assert torch.equal(distances.grad, torch.full_like(distances, 0.5))
