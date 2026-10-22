"""Linear preferences over numeric-answer groups, using NumPy inference."""

import json
from pathlib import Path

import numpy as np

from selection_utils import candidate_answers
from submission_ensemble import choose_scores


MODES = ('mean', 'mean_max', 'candidate')


def representations(bank, arrays, mode):
    if mode not in MODES:
        raise ValueError('Unknown pooling mode')
    embeddings = np.asarray(arrays['embeddings'], dtype=np.float64)
    if embeddings.shape != (len(bank), 4, 1024) or not np.isfinite(embeddings).all():
        raise ValueError('Expected finite [N, 4, 1024] embeddings')
    result = []
    for row, values, parsed in zip(bank, embeddings, candidate_answers(bank)):
        if mode == 'candidate':
            result.append([(values[i], [i]) for i in range(4)])
            continue
        members = {}
        for i, answer in enumerate(parsed):
            if answer is not None:
                members.setdefault(answer, []).append(i)
        groups = []
        for answer in sorted(members):
            # Stable summation order also makes pooled floats invariant to permutation.
            indices = sorted(members[answer], key=lambda i: (
                row['candidates'][i]['text'], values[i].tobytes()))
            block = values[indices]
            mean = block.mean(axis=0)
            support = len(indices)
            extra = np.asarray([support, support / 4, len(members),
                                np.square(block - mean).mean()], dtype=np.float64)
            feature = np.concatenate([mean, block.max(axis=0), extra]) if mode == 'mean_max' \
                else np.concatenate([mean, extra])
            groups.append((feature, indices))
        result.append(groups)
    return result


def candidate_scores(bank, arrays, mode, weights):
    result = np.full((len(bank), 4), -1e100, dtype=np.float64)
    parsed_answers = candidate_answers(bank)
    for row_index, groups in enumerate(representations(bank, arrays, mode)):
        parsed = parsed_answers[row_index]
        if not any(answer is not None for answer in parsed):
            result[row_index] = 0
            continue
        for feature, indices in groups:
            result[row_index, indices] = feature @ weights
        for i, answer in enumerate(parsed):
            if answer is None:
                result[row_index, i] = -1e100
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite ranking scores')
    return result


def load_checkpoint(checkpoint):
    directory = Path(checkpoint)
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    if config.get('model') != 'numeric_answer_group_linear_pairwise' or \
            config.get('schema_version') != 1 or config.get('mode') not in MODES:
        raise ValueError('Unsupported answer-group checkpoint')
    dimension = {'mean': 1028, 'mean_max': 2052, 'candidate': 1024}[config['mode']]
    with np.load(directory / 'weights.npz', allow_pickle=False) as values:
        weights = values['w'].copy()
    if weights.shape != (dimension,) or not np.isfinite(weights).all():
        raise ValueError('Invalid ranking coefficients')
    return config, weights


def predict(bank, arrays, checkpoint):
    config, weights = load_checkpoint(checkpoint)
    scores = candidate_scores(bank, arrays, config['mode'], weights)
    return choose_scores(scores, bank, arrays)
