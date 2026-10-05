"""Grounding changes state recurrence, never the query-based stopping rule."""
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from test_transfer_contract import make_model, Objective
from tasks.arithmetic_factorization.task import build_neo
from tasks.arithmetic_factorization.latent_inference import latent_candidates
from tasks.gridworld.theorizer_evaluation import transfer_rollout


class FractionalExecutor(nn.Module):
    def forward(self, state, action):
        return state + .6


class RoundedObjective(Objective):
    def decode_prediction(self, prediction):
        return prediction.round()


class RoundedGridDecoder(nn.Module):
    def forward(self, state):
        return F.one_hot(state.round().long(), 9).float()


def test_shared_hard_grounding_changes_only_eval_recurrence():
    model = make_model().eval()
    model.program_executor = FractionalExecutor()
    model.objective = RoundedObjective()
    batch=torch.tensor([0.,2.,3.,5.]).reshape(1,4,1,1)
    latent=model(batch,is_eval=True)
    hard=model(batch,is_eval=True,hard_grounding=True)
    assert latent.selected_lengths.tolist()==[3,3]
    assert hard.selected_lengths.tolist()==[2,2]
    batch[:,3]=8
    changed=model(batch,is_eval=True,hard_grounding=True)
    assert torch.equal(hard.predictions,changed.predictions)
    assert torch.equal(hard.selected_lengths,changed.selected_lengths)
    with pytest.raises(ValueError,match='evaluation-only'):
        model(batch,hard_grounding=True)


@pytest.mark.parametrize('hard',[False,True])
@pytest.mark.parametrize('horizon',[3,6])
def test_arithmetic_both_modes_match_shared_greedy(hard,horizon):
    torch.manual_seed(12)
    model=build_neo('neo','0.33').eval()
    data=torch.randint(0,10,(2,4,1,6))
    for episode in data:
        with torch.no_grad():out=model(episode[None],is_eval=True,num_transitions=horizon,hard_grounding=hard)
        result=latent_candidates(model,episode,max_steps=horizon,greedy=True,hard_grounding=hard)
        assert torch.equal(result.query_prediction,out.predictions.argmax(-1)[1::2])
        assert torch.equal(result.support_prediction,out.predictions.argmax(-1)[::2])
        assert result.lengths.item()==out.selected_lengths[0].item()


@pytest.mark.parametrize('hard',[False,True])
def test_arithmetic_sampling_does_not_use_query_labels(hard):
    torch.manual_seed(12);model=build_neo('neo','0.33').eval()
    episode=torch.randint(0,10,(4,1,6));calls=[]
    hook=model.encoder.register_forward_hook(lambda *a:calls.append(1))
    torch.manual_seed(42)
    original=latent_candidates(model,episode,max_steps=3,candidates=8,hard_grounding=hard)
    assert len(calls)==(5 if hard else 2)
    hook.remove();episode[3]=(episode[3]+3)%10
    torch.manual_seed(42)
    changed=latent_candidates(model,episode,max_steps=3,candidates=8,hard_grounding=hard)
    assert torch.equal(original.indices,changed.indices)
    assert torch.equal(original.lengths,changed.lengths)
    assert torch.equal(original.query_prediction,changed.query_prediction)


def test_gridworld_greedy_switch_controls_support_and_query():
    model=make_model().eval();model.program_executor=FractionalExecutor();model.decoder=RoundedGridDecoder()
    args=[model,torch.tensor([[0]]),torch.tensor([[2]]),torch.tensor([[3]]),torch.tensor([[5]])]
    for hard,expected_steps in [(True,2),(False,3)]:
        calls=[];hook=model.encoder.register_forward_hook(lambda *a:calls.append(1))
        out=transfer_rollout(*args,max_steps=3,device=torch.device('cpu'),hard_grounding=hard)
        hook.remove()
        assert out.num_extracted_steps==out.num_applied_steps==expected_steps
        assert len(calls)==(3+2*expected_steps if hard else 3)
        args[-1]=torch.tensor([[8]])
        changed=transfer_rollout(*args,max_steps=3,device=torch.device('cpu'),hard_grounding=hard)
        assert torch.equal(out.final_prediction,changed.final_prediction)
        args[-1]=torch.tensor([[5]])


def test_gridworld_sampling_grounding_switch_and_default_parity():
    from tasks.gridworld.task import build_neo as grid_model
    from tasks.gridworld.theorizer_scaling import sample_gridworld_theories,apply_gridworld_theories
    torch.manual_seed(12);model=grid_model('alpha-0.33').eval()
    grids=torch.randint(0,9,(4,10,10));args=dict(max_steps=3,num_theories=4,sample_temperature=.3,device=torch.device('cpu'))
    torch.manual_seed(42);default=sample_gridworld_theories(model,grids[0],grids[1],**args)
    torch.manual_seed(42);explicit=sample_gridworld_theories(model,grids[0],grids[1],**args,hard_grounding=True)
    assert default==explicit
    for hard in [False,True]:
        calls=[];hook=model.encoder.register_forward_hook(lambda *a:calls.append(1))
        theories=sample_gridworld_theories(model,grids[0],grids[1],**args,hard_grounding=hard)
        apply_gridworld_theories(model,grids[2],grids[3],theories,device=torch.device('cpu'),hard_grounding=hard)
        hook.remove();assert len(calls)==(9 if hard else 3)


def test_image_uses_existing_8bit_projection_for_hard_grounding():
    from tasks.image_editing.task import build_neo as image_model
    torch.manual_seed(12);model=image_model('0.33').eval()
    data=torch.randint(0,256,(1,4,32,32,3),dtype=torch.uint8)
    states=[];inputs=[]
    h1=model.program_executor.register_forward_hook(lambda m,a,y:states.append(y.detach().clone()))
    h2=model.program_executor.register_forward_pre_hook(lambda m,a:inputs.append(a[0].detach().clone()))
    with torch.no_grad():
        out=model(data,is_eval=True,hard_grounding=True)
        for step in range(1,len(inputs)):
            decoded=model.objective.decode_prediction(model.decoder(states[step-1]))
            expected,_=model.encoder(decoded)
            torch.testing.assert_close(inputs[step],expected,rtol=0,atol=0)
    h1.remove();h2.remove()
    changed=data.clone();changed[:,3]=0
    with torch.no_grad():other=model(changed,is_eval=True,hard_grounding=True)
    assert torch.equal(out.predictions,other.predictions)
    assert torch.equal(out.selected_lengths,other.selected_lengths)


def test_matched_arithmetic_defaults_to_hard_grounding():
    torch.manual_seed(12)
    model = build_neo('neo', '0.33').eval()
    episode = torch.randint(0, 10, (4, 1, 6))
    calls = []
    hook = model.encoder.register_forward_hook(lambda *args: calls.append(1))
    default = latent_candidates(model, episode, max_steps=3, greedy=True)
    hook.remove()
    assert len(calls) == 5
    explicit = latent_candidates(model, episode, max_steps=3, greedy=True, hard_grounding=True)
    assert torch.equal(default.query_prediction, explicit.query_prediction)
    assert torch.equal(default.lengths, explicit.lengths)


def test_matched_scaling_default_matches_explicit_hard():
    from tasks.arithmetic_factorization.theorizer_scaling import evaluate_test_time_scaling
    torch.manual_seed(12)
    model = build_neo('neo', '0.33').eval()
    data = torch.randint(0, 10, (2, 4, 1, 6))
    args = dict(max_steps=3, budgets=(1, 4), device='cpu')
    batches = [(data[:, :3], data[:, 3], [])]
    torch.manual_seed(42)
    default = evaluate_test_time_scaling(model, batches, **args)
    torch.manual_seed(42)
    explicit = evaluate_test_time_scaling(model, batches, **args, hard_grounding=True)
    assert default.legacy_dict() == explicit.legacy_dict()
