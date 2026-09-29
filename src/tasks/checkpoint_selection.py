"""Select saved NEO checkpoints using task-specific ID query transfer.

ID is explicitly selection data, not an independent held-out test. OOD data
are not accepted by this module; the caller records the ID artifact identity.
"""
from __future__ import annotations

from pathlib import Path
import math

import h5py
import numpy as np
import torch

from training.runtime import sha256_file as checkpoint_hash

HARD_GROUNDING = {"gridworld": True, "arithmetic": True, "image": False}


def read_episodes(path: Path, domain: str, limit: int | None = None) -> torch.Tensor:
    """Read the same support/query layout as the task loaders, once per sweep."""
    with h5py.File(path, 'r') as file:
        count = int(file.attrs['dataset_length'])
        if limit is not None:
            count = min(count, limit)
        first = file['sample_0']['grids'][:]
        shape = (count, 4, *first.shape[1:])
        episodes = np.empty(shape, dtype=np.uint8)
        for index in range(count):
            group = file[f'sample_{index}']
            grids = group['grids'][:]
            if domain == 'image':
                episode = grids
            else:
                episode = np.concatenate([grids, group['answer'][:][None]], axis=0)
                if domain == 'arithmetic':
                    episode = np.where(episode == 255, 0, episode)
            if episode.min() < 0 or episode.max() > 255:
                raise ValueError('observation is outside the stored byte range')
            episodes[index] = episode
    return torch.from_numpy(episodes)


def build_model(domain: str, alpha: str, device: str):
    if domain == 'gridworld':
        from tasks.gridworld.task import build_neo
        model = build_neo(f'alpha-{alpha}')
    elif domain == 'arithmetic':
        from tasks.arithmetic_factorization.task import build_neo
        model = build_neo('neo', alpha)
    elif domain == 'image':
        from tasks.image_editing.task import build_neo
        model = build_neo(('neo', alpha))
    else:
        raise ValueError(f'unknown domain {domain}')
    return model.to(device).eval()


def load_candidate(model, domain: str, alpha: str, seed: int, checkpoint: Path) -> dict:
    if domain == 'gridworld':
        from tasks.gridworld.theorizer_evaluation import load_theorizer_checkpoint
        payload = load_theorizer_checkpoint(model, checkpoint,
            expected_experiment=f'alpha-{alpha}', expected_method='neo', expected_seed=seed)
        coefficient = None
    elif domain == 'image':
        from tasks.image_editing.evaluation import load_model_checkpoint, checkpoint_length_control
        metadata = {}
        load_model_checkpoint(model, checkpoint, metadata=metadata)
        payload = metadata
        coefficient = checkpoint_length_control(metadata)
    else:
        from tasks.arithmetic_factorization.experiment import get_experiment
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(payload['model_state_dict'], strict=True)
        model.update_length_control(payload['global_step'], get_experiment('neo', alpha).total_steps)
        coefficient = model.length_control_coefficient
    if domain in ('arithmetic', 'image'):
        config = payload['config']
        if (config.get('method'), str(config.get('alpha')), config.get('seed')) != ('neo', alpha, seed):
            raise ValueError(f'checkpoint metadata does not match NEO alpha={alpha}, seed={seed}')
    return {'global_step': int(payload['global_step']), 'coefficient': coefficient}


@torch.no_grad()
def score_id(model, domain: str, episodes: torch.Tensor, *, device: str,
             coefficient: float | None = None, batch_size: int | None = None) -> dict:
    """Keep each task's final-evaluator stopping, projection and precision."""
    hard_grounding = HARD_GROUNDING[domain]
    count = len(episodes)
    if count == 0:
        raise ValueError('ID selection requires at least one episode')
    total = 0.0
    if domain == 'gridworld':
        from tasks.gridworld.theorizer_evaluation import transfer_rollout
        for episode in episodes:
            result = transfer_rollout(model, *episode, max_steps=4,
                                      device=torch.device(device), hard_grounding=hard_grounding)
            total += int(result.solved)
        metric, direction = 'query_grid_accuracy', 'max'
    elif domain == 'arithmetic':
        batch_size = batch_size or 128
        for start in range(0, count, batch_size):
            batch = episodes[start:start + batch_size].to(device).long()
            with torch.autocast(device_type=torch.device(device).type, enabled=False):
                output = model(batch, is_eval=True, num_transitions=3, hard_grounding=hard_grounding)
            predicted = output.predictions.argmax(-1)[1::2]
            correct = model.objective.exact_match(predicted, batch[:, 3])
            total += int(correct.sum())
        metric, direction = 'query_number_accuracy', 'max'
    elif domain == 'image':
        from training.runtime import bfloat16_autocast
        batch_size = batch_size or 64
        for start in range(0, count, batch_size):
            batch = episodes[start:start + batch_size].to(device)
            with bfloat16_autocast(torch.device(device)):
                output = model(batch, is_eval=True, hard_grounding=hard_grounding,
                               length_control_coefficient=coefficient)
            total += float(output.query_metrics.l1) * len(batch)
        metric, direction = 'query_l1', 'min'
    else:
        raise ValueError(f'unknown domain {domain}')
    value = total / count
    if not math.isfinite(value):
        raise ValueError('non-finite ID selection score')
    return {'score': value, 'metric': metric, 'direction': direction,
            'episodes': count, 'hard_grounding': hard_grounding, 'split': 'id',
            'id_is_selection_data': True}


def choose_best(records: list[dict]) -> dict:
    """Select by ID score only; latest optimizer step resolves exact ties."""
    if not records:
        raise ValueError('no checkpoint scores')
    contract = {(r['metric'], r['direction'], r['episodes'], r['hard_grounding']) for r in records}
    if len(contract) != 1:
        raise ValueError('all candidates require the same ID metric, episodes and grounding setting')
    for r in records:
        if r['split'] != 'id' or not isinstance(r['hard_grounding'], bool) or not math.isfinite(r['score']):
            raise ValueError('selection accepts finite ID scores with a boolean grounding setting only')
        if r['direction'] not in ('max', 'min'):
            raise ValueError('unknown optimization direction')
    sign = 1 if records[0]['direction'] == 'max' else -1
    return max(records, key=lambda r: (sign * r['score'], r['global_step'], r['checkpoint']))
