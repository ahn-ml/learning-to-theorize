from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tasks.arithmetic_factorization.validation import released_transfer_batch


class Encoder(nn.Module):
    def forward(self, x):
        return x.float(), None


class Decoder(nn.Module):
    def forward(self, x):
        return F.one_hot(x.long().remainder(10), 10).float()


class Programmer(nn.Module):
    def forward(self, current, goal):
        return torch.full_like(current, 1.6)


class Executor(nn.Module):
    def forward(self, state, action):
        return state + action


class Quantizer(nn.Module):
    def forward(self, actions, **kwargs):
        return SimpleNamespace(values=actions, indices=torch.zeros(len(actions), dtype=torch.long))


def model():
    return SimpleNamespace(encoder=Encoder(), decoder=Decoder(), theory_programmer=Programmer(),
                           program_executor=Executor(), quantizer=Quantizer())


@pytest.mark.parametrize('horizon',[3,6])
def test_batched_validation_matches_released_stopping_and_predictions(horizon):
    m=model()
    data=torch.tensor([[0,1,3,4],[0,2,3,5],[0,9,3,6]]).reshape(3,4,1,1)
    result=released_transfer_batch(m,data,max_steps=horizon)
    assert result['selected_lengths'].tolist()==[1,2,horizon]
    assert result['query_prediction'].flatten().tolist() == [4, 5, (3 + horizon) % 10]
    assert result['query_solved'].tolist() == [True, True, horizon == 3]
    changed=data.clone();changed[:,3]=(changed[:,3]+2)%10
    again=released_transfer_batch(m,changed,max_steps=horizon)
    assert torch.equal(result['query_prediction'],again['query_prediction'])
    assert torch.equal(result['selected_lengths'],again['selected_lengths'])


def test_fabric_validation_preserves_weights_rng_and_ema():
    from lightning.fabric import Fabric
    from tasks.arithmetic_factorization.experiment import get_experiment
    from tasks.arithmetic_factorization.task import build_neo
    from tasks.arithmetic_factorization.theorizer_runner import setup_training, _evaluate

    torch.manual_seed(42)
    original=build_neo('neo','0.33')
    original.freeze_observation_model()
    fabric=Fabric(accelerator='cpu',devices=1,precision='bf16-mixed')
    wrapped,_=setup_training(fabric,original,get_experiment('neo','0.33'))
    data=torch.randint(0,10,(3,4,1,6))
    batches=[(data[:2,:3],data[:2,3],[]),(data[2:,:3],data[2:,3],[])]
    before={k:v.clone() for k,v in original.state_dict().items()}
    rng=torch.get_rng_state()
    metrics=_evaluate(wrapped,batches,fabric)
    assert all(key in metrics for key in ('latent_transfer_accuracy','paper_transfer_accuracy','transfer_accuracy'))
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,original.state_dict()[k]) for k,v in before.items())
    assert original.quantizer.quantizer._ema.counts is None
    with torch.no_grad(), fabric.autocast():
        expected = released_transfer_batch(original, data, max_steps=3)['query_solved'].tolist()
    assert metrics['transfer_accuracy']==pytest.approx(sum(expected)/len(expected))
