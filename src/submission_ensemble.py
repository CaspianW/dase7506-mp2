"""Average three frozen ranking models with NumPy CPU inference."""

import json
from pathlib import Path

import numpy as np

from selection_utils import candidate_answers


PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def load_checkpoint(checkpoint):
    directory = Path(checkpoint).resolve()
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    if config.get('model') != 'equal_raw_score_ensemble' or config.get('schema_version') != 1:
        raise ValueError('Unsupported ensemble checkpoint')
    members = config.get('members')
    if not isinstance(members, list) or len(members) != 3 or config.get('filter_invalid') is not True:
        raise ValueError('Expected three equally weighted models and invalid-answer filtering')
    loaded = []
    for relative in members:
        member = (directory / relative).resolve()
        if Path(relative).is_absolute() or not member.is_relative_to(directory):
            raise ValueError('Member assets must be inside the ensemble checkpoint')
        model = json.loads((member / 'config.json').read_text(encoding='utf-8'))
        kind = model.get('model')
        if kind not in ('embedding_mlp', 'antisymmetric_pairwise_mlp') or \
                model.get('schema_version') != 1 or model.get('hidden_dimension') != 64:
            raise ValueError('Unsupported ensemble member')
        shapes = {'w1': (1024, 64), 'b1': (64,), 'w2': (64,)}
        if kind == 'embedding_mlp':
            shapes['b2'] = ()
        with np.load(member / 'weights.npz', allow_pickle=False) as values:
            weights = {name: values[name].copy() for name in shapes}
        if any(weights[name].shape != shape or not np.isfinite(weights[name]).all()
               for name, shape in shapes.items()):
            raise ValueError('Invalid member weights')
        loaded.append((kind, weights))
    if len({kind for kind, _ in loaded}) != 1:
        raise ValueError('All members must belong to the same ranking family')
    return config, loaded


def member_scores(embeddings, kind, weights):
    if kind == 'embedding_mlp':
        hidden = np.maximum(embeddings.reshape(-1, 1024) @ weights['w1'] + weights['b1'], 0)
        return (hidden @ weights['w2'] + weights['b2']).reshape(-1, 4)
    result = np.zeros((len(embeddings), 4), dtype=np.float64)
    left, right = np.asarray(PAIRS).T
    for start in range(0, len(embeddings), 256):
        stop = min(start + 256, len(embeddings))
        differences = (embeddings[start:stop, left] - embeddings[start:stop, right]).reshape(-1, 1024)
        projected = differences @ weights['w1']
        plus = np.maximum(projected + weights['b1'], 0)
        minus = np.maximum(-projected + weights['b1'], 0)
        preference = (.5 * ((plus - minus) @ weights['w2'])).reshape(-1, 6)
        for pair, (i, j) in enumerate(PAIRS):
            result[start:stop, i] += preference[:, pair] / 3
            result[start:stop, j] -= preference[:, pair] / 3
    return result


def candidate_scores(bank, arrays, members):
    embeddings = np.asarray(arrays['embeddings'], dtype=np.float64)
    if embeddings.shape != (len(bank), 4, 1024) or not np.isfinite(embeddings).all():
        raise ValueError('Expected finite embeddings with shape [N, 4, 1024]')
    result = np.zeros((len(bank), 4), dtype=np.float64)
    for kind, weights in members:
        result += member_scores(embeddings, kind, weights) / 3
    return result


def choose_scores(scores, bank, arrays):
    answers = candidate_answers(bank)
    embeddings = np.asarray(arrays['embeddings'], dtype=np.float64)
    choices = []
    for row_index, (row, parsed) in enumerate(zip(bank, answers)):
        usable = [i for i, value in enumerate(parsed) if value is not None] or list(range(4))
        maximum = max(scores[row_index, i] for i in usable)
        tied = [i for i in usable if scores[row_index, i] == maximum]
        # Content resolves exact score ties without depending on IDs or positions.
        choice = min(tied, key=lambda i: (parsed[i] or '', row['candidates'][i]['text'],
                                        embeddings[row_index, i].tobytes()))
        choices.append(int(choice))
    return choices


def predict(bank, arrays, checkpoint):
    _, members = load_checkpoint(checkpoint)
    return choose_scores(candidate_scores(bank, arrays, members), bank, arrays)
