
import pytest
import torch

from tasks.arithmetic_factorization.latent_inference import latent_candidates
from tasks.arithmetic_factorization.task import build_neo
from tasks.arithmetic_factorization.theorizer_scaling import SampledArithmeticTheories, budget_outcome


@pytest.mark.parametrize('horizon', [3, 6])
@pytest.mark.parametrize('hard_grounding', [False, True])
def test_greedy_matches_shared_neo_and_preserves_state(horizon, hard_grounding):
    torch.manual_seed(12)
    model = build_neo('neo', '0.33').eval()
    data = torch.randint(0, 10, (4, 4, 1, 6))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state()
    for episode in data:
        with torch.no_grad():
            reference = model(episode.unsqueeze(0), is_eval=True, num_transitions=horizon,
                              hard_grounding=hard_grounding)
        result = latent_candidates(model, episode, max_steps=horizon, greedy=True, hard_grounding=hard_grounding)
        assert result.query_correct.item() == bool(reference.query_metrics.number_accuracy)
        assert result.support_correct.item() == bool(reference.metrics.number_accuracy)
        assert result.lengths.item() == reference.mean_explanation_length.item()
        assert torch.equal(result.support_prediction, reference.predictions.argmax(-1)[::2])
        assert torch.equal(result.query_prediction, reference.predictions.argmax(-1)[1::2])
        changed = episode.clone(); changed[3] = (changed[3] + 1) % 10
        again = latent_candidates(model, changed, max_steps=horizon, greedy=True, hard_grounding=hard_grounding)
        assert torch.equal(result.indices, again.indices)
        assert torch.equal(result.lengths, again.lengths)
        assert torch.equal(result.query_prediction, again.query_prediction)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_sampled_programs_do_not_use_query_labels_or_reencode():
    torch.manual_seed(12)
    model = build_neo('neo', '0.33').eval()
    episode = torch.randint(0, 10, (4, 1, 6))
    calls = []
    hook = model.encoder.register_forward_hook(lambda *args: calls.append(1))
    torch.manual_seed(42)
    original = latent_candidates(model, episode, max_steps=6, candidates=16, hard_grounding=False)
    # Only the initial input and target encode, never intermediate predictions.
    assert len(calls) == 2
    hook.remove()
    episode[3] = (episode[3] + 3) % 10
    torch.manual_seed(42)
    changed = latent_candidates(model, episode, max_steps=6, candidates=16, hard_grounding=False)
    assert torch.equal(original.indices, changed.indices)
    assert torch.equal(original.lengths, changed.lengths)
    assert torch.equal(original.query_prediction, changed.query_prediction)


def test_vote_uses_solved_support_prefix_only():
    theories = SampledArithmeticTheories(
        action_indices=(((1,), (2,)), ((1,), (9,)), ((3,), (4,)), ((3,), (4,))),
        support_solved_at_step=(1, 1, 2, -1),
    )
    # First two reinforce the same truncated program; failed fourth is ignored.
    result = budget_outcome(theories, [False, False, True, True], 4)
    assert result[3] is False and result[4] == 3 and result[5] == 2


def test_scaling_evaluator_latent_route():
    from tasks.arithmetic_factorization.theorizer_scaling import evaluate_test_time_scaling
    model = build_neo('neo', '0.33').eval()
    data = torch.randint(0, 10, (2, 4, 1, 6))
    result = evaluate_test_time_scaling(model, [(data[:, :3], data[:, 3], [])],
                                      max_steps=3, budgets=(1, 4), device='cpu', hard_grounding=False)
    assert [m.budget for m in result.budgets] == [1, 4]
    assert all(m.total == 2 for m in result.budgets)
