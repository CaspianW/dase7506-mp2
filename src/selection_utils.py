"""Candidate parsing, selection rules, and development metrics."""

from collections import Counter

import numpy as np

from common import parse_answer


def candidate_answers(bank):
    return [[parse_answer(c['text']) for c in row['candidates']] for row in bank]


def make_targets(bank, references):
    """Label every correct candidate, including multiple positives in a group."""
    if len(bank) != len(references):
        raise ValueError('Reference and bank lengths differ')
    labels = []
    for row, ref in zip(bank, references):
        if row['id'] != ref['id'] or row['question'] != ref['question']:
            raise ValueError('Reference and bank alignment differs')
        gold = parse_answer(ref['answer'])
        if gold is None:
            raise ValueError('A reference has no valid numeric answer')
        labels.append([parse_answer(c['text']) == gold for c in row['candidates']])
    labels = np.asarray(labels, dtype=bool).reshape(-1, 4)
    if not labels.any(axis=1).all():
        raise ValueError('A development group has no correct candidate')
    return labels


def choose_scores(scores, answers, filter_invalid=True):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(answers), 4) or not np.isfinite(scores).all():
        raise ValueError('Expected four finite scores per question')
    if len(answers) == 0:
        return []
    if filter_invalid:
        valid = np.asarray([[a is not None for a in row] for row in answers])
        # An audit bank might contain a group with no parseable answers.
        usable = valid | ~valid.any(axis=1, keepdims=True)
        scores = np.where(usable, scores, -np.inf)
    return [int(i) for i in scores.argmax(axis=1)]


def baseline_choices(answers, method):
    choices = []
    for row in answers:
        if method == 'candidate0':
            choice = 0
        elif method == 'first_valid':
            choice = next((i for i, a in enumerate(row) if a is not None), 0)
        elif method == 'majority':
            counts = Counter(a for a in row if a is not None)
            choice = max(range(4), key=lambda i: counts.get(row[i], 0)
                         if row[i] is not None else -1)
        else:
            raise ValueError(f'Unknown baseline: {method}')
        choices.append(choice)
    return choices


def disagreement_mask(answers):
    return np.asarray([len({a for a in row if a is not None}) >= 2
                       for row in answers], dtype=bool)


def selection_metrics(labels, choices, answers):
    choices = np.asarray(choices, dtype=np.int64)
    correct = labels[np.arange(len(labels)), choices]
    hard = disagreement_mask(answers)

    def summarize(mask):
        n = int(mask.sum())
        count = int(correct[mask].sum())
        return {'correct': count, 'n': n,
                'accuracy': 100 * count / n if n else None}

    return {'overall': summarize(np.ones(len(labels), dtype=bool)),
            'disagreement': summarize(hard)}


def paired_comparison(labels, choices, baseline, answers):
    selected = labels[np.arange(len(labels)), choices]
    previous = labels[np.arange(len(labels)), baseline]
    result = {}
    for name, mask in [('overall', np.ones(len(labels), dtype=bool)),
                       ('disagreement', disagreement_mask(answers))]:
        fixed = int((mask & selected & ~previous).sum())
        broken = int((mask & previous & ~selected).sum())
        result[name] = {'fixed': fixed, 'broken': broken, 'net_correct': fixed - broken}
    return result
