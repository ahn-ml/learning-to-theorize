"""EMA is a step-level update over all rollout positions and all ranks."""
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from tasks.image_editing.models.quantizer import ActionQuantizer, ActionQuantizerConfig


def make(task):
    return ActionQuantizer(ActionQuantizerConfig(codebook_size=4, action_dim=4,
                                                ema_decay=.9, entropy_weight=0))


def forward(q, task, x, training_mode=True):
    out=q(x,training_mode=training_mode)
    return out.indices,out.loss_per_sample


@pytest.mark.parametrize('task',['image'])
def test_rollout_is_frozen_then_one_ema_update(task):
    torch.manual_seed(5);q=make(task).train();before={k:v.clone() for k,v in q.state_dict().items()}
    counts=torch.zeros(4);sums=torch.zeros(4,4);loss=0
    for _ in range(3):
        x=torch.randn(7,1,4,requires_grad=True);idx,l=forward(q,task,x)
        one=F.one_hot(idx,4).float();counts+=one.sum(0);sums+=one.t()@F.normalize(x.detach().flatten(0,1),dim=-1)
        loss=loss+l.mean()
        assert torch.equal(q.embedding.weight,before['embedding.weight'])
        assert torch.equal(q.ema_cluster_size,before['ema_cluster_size'])
    loss.backward();q.step()
    expected_counts=.1*counts;expected_sums=.9*before['ema_weight']+.1*sums
    total=expected_counts.sum();denom=(expected_counts+1e-5)/(total+4e-5)*total
    torch.testing.assert_close(q.ema_cluster_size,expected_counts)
    torch.testing.assert_close(q.embedding.weight,expected_sums/denom[:,None])
    assert not q.ema_weight.requires_grad
    assert set(q.state_dict())==set(before)  # checkpoint keys unchanged
    after=q.embedding.weight.clone();q.step();assert torch.equal(after,q.embedding.weight)
    forward(q,task,torch.randn(3,1,4),training_mode=False);q.step()
    assert torch.equal(after,q.embedding.weight)  # explicit eval cannot queue updates
    q.eval();forward(q,task,torch.randn(3,1,4));q.step()
    assert torch.equal(after,q.embedding.weight)


def distributed_worker(rank,path,task):
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    try:
        torch.manual_seed(8);q=make(task).train();reference=make(task).train();reference.load_state_dict(q.state_dict())
        # Each rank sees different assignments. Reference receives their union.
        batches=[torch.randn(3,1,4),torch.randn(5,1,4)]
        forward(q,task,batches[rank]);q.step()
        merged=torch.cat(batches);forward(reference,task,merged)
        # Analytical global update without calling a second collective.
        counts=reference._ema.counts; sums=reference._ema.sums
        c=.1*counts;w=.9*reference.ema_weight+.1*sums;n=c.sum();denom=(c+1e-5)/(n+4e-5)*n
        torch.testing.assert_close(q.embedding.weight,w/denom[:,None])
        torch.testing.assert_close(q.ema_cluster_size,c)
        other=q.embedding.weight.clone();dist.broadcast(other,src=0)
        torch.testing.assert_close(q.embedding.weight,other,rtol=0,atol=0)
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('task',['image'])
def test_two_rank_ema_matches_global_batch(tmp_path,task):
    mp.spawn(distributed_worker,args=(str(tmp_path/'rendezvous'),task),nprocs=2,join=True)
