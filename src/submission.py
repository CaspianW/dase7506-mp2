"""NumPy group ranking with learned question-number arithmetic evidence."""

import json
from pathlib import Path

import numpy as np

from question_operation_features import VARIANTS, candidate_values, feature_names
from submission_answer_groups import candidate_scores as base_scores, representations
from submission_ensemble import choose_scores


def candidate_scores(bank, arrays, weights, variant):
    result = base_scores(bank, arrays, 'mean', weights[:1028])
    values = candidate_values(bank, variant)
    for q, groups in enumerate(representations(bank, arrays, 'mean')):
        for _, indices in groups:
            result[q, indices] += values[q, indices[0]] @ weights[1028:]
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite question arithmetic score')
    return result


def load_checkpoint(checkpoint):
    directory = Path(checkpoint)
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    if config.get('model') != 'answer_group_question_operations' or config.get('schema_version') != 1:
        raise ValueError('Unsupported question arithmetic checkpoint')
    names = feature_names(config['variant'])
    if config.get('feature_names') != names:
        raise ValueError('Question arithmetic feature order changed')
    with np.load(directory / 'weights.npz', allow_pickle=False) as saved:
        if saved.files != ['w']:
            raise ValueError('Unexpected inference assets')
        weights = saved['w'].copy()
    if weights.shape != (1028 + len(names),) or not np.isfinite(weights).all():
        raise ValueError('Invalid question arithmetic coefficients')
    return config, weights


def predict(bank, arrays, checkpoint):
    config, weights = load_checkpoint(checkpoint)
    return choose_scores(candidate_scores(bank, arrays, weights, config['variant']), bank, arrays)
