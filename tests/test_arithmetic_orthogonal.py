"""Behavior of the active regularizer and its persistent EMA correction."""
import copy

import pytest
import torch
import torch.nn.functional as F

from tasks.arithmetic_factorization.experiment import get_experiment
from tasks.arithmetic_factorization.models.quantizer import VectorQuantizer
from tasks.arithmetic_factorization.task import build_neo
from tasks.arithmetic_factorization.theorizer_runner import build_optimizer


@pytest.mark.parametrize('alpha', ['0.33', '0.66', '1.00'])
def test_paper_weight_reaches_quantizer_and_optimizer(alpha):
    config = get_experiment('neo', alpha)
    model = build_neo('neo', alpha)
    q = model.quantizer.quantizer
    assert q.orthogonal_reg_weight == config.quantizer.orthogonal_regularization_weight == 10
    assert q.orthogonal_reg_max_codes == 16
    assert q.embedding.weight.requires_grad
    optimizer = build_optimizer(model, config)
    matches = [g for g in optimizer.param_groups if any(p is q.embedding.weight for p in g['params'])]
    assert len(matches) == 1
    assert matches[0]['lr'] == config.optimization.learning_rate


def make_quantizer(weight=10):
    q = VectorQuantizer(codebook_size=4, embedding_dim=2, use_ema=True,
                        ema_decay=.99, orthogonal_reg_weight=weight,
                        orthogonal_reg_max_codes=4)
    with torch.no_grad():
        q.embedding.weight.copy_(torch.tensor([[1., .2], [.3, 1.], [-1., .1], [-.4, -.8]]))
        q.ema_cluster_size.fill_(2)
        q.ema_weight.copy_(q.embedding.weight * 2)
    return q


def test_regularizer_gradient_is_nonzero_and_reduces_gram_loss():
    q = make_quantizer()
    optimizer = torch.optim.SGD([q.embedding.weight], lr=.01)
    before = q.orthogonal_loss().detach().clone()
    (10*q.orthogonal_loss()).backward()
    assert torch.isfinite(q.embedding.weight.grad).all()
    assert q.embedding.weight.grad.norm() > 0
    optimizer.step()
    assert q.orthogonal_loss() < before


def test_ema_keeps_optimizer_correction_and_carries_it_to_next_step():
    torch.manual_seed(3)
    q = make_quantizer()
    optimizer = torch.optim.SGD([q.embedding.weight], lr=.01)
    for _ in range(2):
        before = q.embedding.weight.detach().clone()
        old_counts = q.ema_cluster_size.clone()
        old_sums = q.ema_weight.clone()
        for _ in range(3):
            _, result = q(torch.randn(16, 1, 2, requires_grad=True))
        counts = q._ema.counts.clone(); sums = q._ema.sums.clone()
        result['quantizer_loss'].backward()
        assert torch.equal(before, q.embedding.weight)
        optimizer.step(); optimizer.zero_grad()
        correction = q.embedding.weight.detach().clone()-before
        assert correction.norm() > 0
        expected_counts = .99*old_counts + .01*counts
        size = (expected_counts+q.ema_epsilon)/(expected_counts.sum()+4*q.ema_epsilon)*expected_counts.sum()
        raw_centroid = (.99*old_sums+.01*sums)/size[:, None]
        expected = raw_centroid+correction
        q.step()
        torch.testing.assert_close(q.embedding.weight, expected)
        assert not torch.allclose(q.embedding.weight, raw_centroid)
        torch.testing.assert_close(q.ema_weight, expected*size[:, None])
        assert q._ema.counts is None and q._embedding_before_optimizer is None


def test_zero_weight_preserves_plain_ema_update():
    q = make_quantizer(0)
    assert not q.embedding.weight.requires_grad
    old_counts=q.ema_cluster_size.clone(); old_sums=q.ema_weight.clone()
    _, result = q(torch.randn(16, 1, 2, requires_grad=True))
    counts=.99*old_counts+.01*q._ema.counts
    sums=.99*old_sums+.01*q._ema.sums
    size=(counts+q.ema_epsilon)/(counts.sum()+4*q.ema_epsilon)*counts.sum()
    result['quantizer_loss'].backward();q.step()
    torch.testing.assert_close(q.embedding.weight, sums/size[:, None])
    assert result['orthogonal_loss'].item()==0


def test_neo_regularizer_is_counted_once_and_does_not_reach_programmer():
    torch.manual_seed(42)
    model = build_neo('neo', '0.66').train()
    output = model(torch.randint(0, 10, (8, 4, 1, 6)))
    q=model.quantizer.quantizer
    actual=torch.autograd.grad(output.action_vq_loss, q.embedding.weight, retain_graph=True)[0]
    expected=torch.autograd.grad(10*q.orthogonal_loss(), q.embedding.weight, retain_graph=True)[0]
    torch.testing.assert_close(actual, expected)
    # Reconstruction's STE must continue to update the programmer, not codes.
    grad=torch.autograd.grad(output.reconstruction_loss, q.embedding.weight, allow_unused=True)[0]
    assert grad is None or grad.count_nonzero()==0


def test_eval_is_read_only_and_checkpoint_continues_same_ema_update():
    q=make_quantizer(); opt=torch.optim.AdamW([q.embedding.weight],lr=.0015,weight_decay=.01)
    _,result=q(torch.randn(16,1,2,requires_grad=True));result['quantizer_loss'].backward()
    opt.step();opt.zero_grad();q.step()
    state=copy.deepcopy(q.state_dict());saved_opt=copy.deepcopy(opt.state_dict())
    # Same persistent checkpoint keys as before orthogonal regularization.
    assert set(state)==set(make_quantizer(0).state_dict())
    restored=make_quantizer();restored.load_state_dict(state)
    restored_opt=torch.optim.AdamW([restored.embedding.weight],lr=.0015,weight_decay=.01)
    restored_opt.load_state_dict(saved_opt)
    q.eval();q.orthogonal_reg_max_codes=2
    rng=torch.get_rng_state().clone()
    q(torch.ones(16,1,2))
    assert torch.equal(rng,torch.get_rng_state())
    assert q._ema.counts is None and q._embedding_before_optimizer is None
    assert all(torch.equal(q.state_dict()[k],v) for k,v in state.items())
    q.orthogonal_reg_max_codes=4
    z=torch.randn(16,1,2)
    for model,optimizer in [(q,opt),(restored,restored_opt)]:
        model.train();_,r=model(z.clone().requires_grad_());r['quantizer_loss'].backward()
        optimizer.step();optimizer.zero_grad();model.step()
    for k,v in q.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[k])
