"""The per-quantization EMA codebook update of the Arithmetic quantizer."""
import pytest
import torch
import torch.nn.functional as F

from tasks.arithmetic_factorization.experiment import get_experiment
from tasks.arithmetic_factorization.models.quantizer import VectorQuantizer
from tasks.arithmetic_factorization.task import build_neo
from tasks.arithmetic_factorization.theorizer_runner import build_optimizer


def make():
    # A near-zero temperature makes training-mode sampling pick the nearest code.
    q = VectorQuantizer(codebook_size=4, embedding_dim=2, use_ema=True, ema_decay=.99,
                        tau_start=1e-6, tau_end=1e-6, tau_steps=1)
    with torch.no_grad():
        q.embedding.weight.copy_(torch.tensor([[1., .2], [.3, 1.], [-1., .1], [-.4, -.8]]))
        q.ema_cluster_size.fill_(2)
        q.ema_weight.copy_(q.embedding.weight * 2)
    return q


def reference_update(q, z_flattened, indices):
    """The EMA update, written out term by term."""
    one_hot = F.one_hot(indices, 4).float()
    size = q.ema_cluster_size * q.ema_decay + (1 - q.ema_decay) * one_hot.sum(0)
    weight = q.ema_weight * q.ema_decay + (1 - q.ema_decay) * torch.matmul(one_hot.t(), z_flattened)
    n = torch.sum(size.view(-1))
    updated = (size + q.ema_epsilon) / (n + 4 * q.ema_epsilon) * n
    return size, weight, weight / updated.unsqueeze(1)


def test_update_happens_before_quantizing_and_carries_no_graph():
    torch.manual_seed(0)
    q = make()
    z = torch.randn(16, 1, 2, requires_grad=True)
    flat = F.normalize(z.detach().flatten(end_dim=-2), dim=-1)
    distances = (flat.square().sum(1, keepdim=True) + F.normalize(q.embedding.weight, dim=-1).square().sum(1)
                 - 2 * flat @ F.normalize(q.embedding.weight, dim=-1).T)
    indices = distances.argmin(1)
    size, weight, embedding = reference_update(q, flat, indices)
    values, out = q(z)
    assert torch.equal(out['min_encoding_indices'], indices)
    assert torch.equal(q.ema_cluster_size, size) and torch.equal(q.ema_weight, weight)
    assert torch.equal(q.embedding.weight, embedding)
    # The quantized forward value comes from the refreshed codebook.
    torch.testing.assert_close(values.detach().flatten(end_dim=-2), F.normalize(embedding[indices], dim=-1))
    assert not q.ema_weight.requires_grad and q.ema_weight.grad_fn is None


def test_eval_and_step_leave_codebook_unchanged():
    q = make()
    q(torch.randn(8, 1, 2))
    state = {k: v.clone() for k, v in q.state_dict().items()}
    q.step()
    assert all(torch.equal(q.state_dict()[k], v) for k, v in state.items())
    q.eval()
    q(torch.randn(8, 1, 2))
    assert all(torch.equal(q.state_dict()[k], v) for k, v in state.items())


@pytest.mark.parametrize('alpha', ['0.33', '0.66', '1.00'])
def test_ema_codebook_is_not_optimized(alpha):
    model = build_neo('neo', alpha)
    q = model.quantizer.quantizer
    assert q.use_ema and not q.embedding.weight.requires_grad
    optimizer = build_optimizer(model, get_experiment('neo', alpha))
    assert not any(p is q.embedding.weight for g in optimizer.param_groups for p in g['params'])
