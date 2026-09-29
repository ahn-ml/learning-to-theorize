import pytest
import torch

from tasks.checkpoint_selection import choose_best, score_id
from tasks.arithmetic_factorization.latent_inference import latent_candidates
from tasks.arithmetic_factorization.task import build_neo


def record(score, step, direction='max'):
    return dict(score=score, global_step=step, checkpoint=f'{step}.pth',
                direction=direction, metric='query', episodes=12,
                split='id', hard_grounding=True)


def test_selects_query_metric_and_latest_exact_tie():
    assert choose_best([record(.8, 3), record(.9, 2), record(.9, 4)])['global_step'] == 4
    assert choose_best([record(.2, 3, 'min'), record(.1, 1, 'min')])['global_step'] == 1


def test_image_selection_uses_unprojected_query_loss():
    from types import SimpleNamespace

    calls = []
    def model(batch, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(query_metrics=SimpleNamespace(l1=0.25))

    score = score_id(model, 'image', torch.zeros(3, 4, 3, 32, 32, dtype=torch.uint8),
                     device='cpu', coefficient=0.99, batch_size=2)
    assert score['hard_grounding'] is False
    assert score['score'] == 0.25 and score['direction'] == 'min'
    assert all(call['hard_grounding'] is False for call in calls)
    assert all(call['length_control_coefficient'] == 0.99 for call in calls)
    records = [{**record(loss, step, 'min'), 'hard_grounding': False}
               for loss, step in [(0.2, 1), (0.1, 2), (0.1, 3)]]
    assert choose_best(records)['global_step'] == 3


@pytest.mark.parametrize('change', [dict(split='length-ood'), dict(hard_grounding=False),
                                   dict(score=float('nan')), dict(episodes=13)])
def test_rejects_invalid_selection_records(change):
    with pytest.raises(ValueError):
        choose_best([record(.2, 1), {**record(.3, 2), **change}])


def test_batched_arithmetic_selection_matches_final_greedy():
    torch.manual_seed(12)
    model = build_neo('neo', '0.33').eval()
    episodes = torch.randint(0, 10, (7, 4, 1, 6), dtype=torch.uint8)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    score = score_id(model, 'arithmetic', episodes, device='cpu', batch_size=3)
    correct = sum(int(latent_candidates(model, episode.long(), max_steps=3,
                  greedy=True, hard_grounding=True).query_correct.item()) for episode in episodes)
    assert score['score'] == correct / len(episodes)
    assert all(torch.equal(before[k], value) for k, value in model.state_dict().items())
    assert score['episodes'] == 7 and score['hard_grounding']
