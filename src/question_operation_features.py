"""Bounded exact arithmetic features from question text and parsed answers."""

from collections import Counter
from fractions import Fraction
import math
import re

import numpy as np

from selection_utils import candidate_answers


VARIANTS = ('numeric_surface', 'operation_support')
SURFACE_NAMES = ('answer_sign', 'log1p_clipped_absolute_answer', 'answer_is_integer',
    'answer_equals_retained_question_number', 'log1p_retained_question_number_count',
    'log1p_answer_to_question_max_magnitude')
OP_NAMES = ('sum_has_match', 'difference_has_match', 'product_has_match', 'quotient_has_match',
    'log1p_sum_matches', 'log1p_difference_matches', 'log1p_product_matches', 'log1p_quotient_matches')
LIMIT = 10**12
PART = r'(?:\d[\d,]*(?:\.\d+)?|\.\d+)'
TOKEN = re.compile(r'(?<![\w.,/])[-+]?' + PART + r'(?:/[-+]?' + PART + r')?,?(?![\w,/]|\.\d)')
STRICT = re.compile(r'[-+]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?|[-+]?\.\d+')


def feature_names(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown question operation variant')
    return list(SURFACE_NAMES + (OP_NAMES if variant == 'operation_support' else ()))


def question_numbers(text):
    result = []
    for match in TOKEN.finditer(str(text)[:20000]):
        token = match.group()
        if len(token) > 80:
            continue
        if token.endswith(','):
            token = token[:-1]
        parts = token.split('/')
        if any(STRICT.fullmatch(part) is None for part in parts):
            continue
        try:
            value = Fraction(parts[0].replace(',', ''))
            if len(parts) == 2:
                value /= Fraction(parts[1].replace(',', ''))
        except (ValueError, ZeroDivisionError):
            continue
        if abs(value) > LIMIT or value.denominator > LIMIT:
            continue
        result.append(value)
        if len(result) == 16:
            break
    return result


def operation_counts(numbers):
    counts = [Counter() for _ in range(4)]
    for i, a in enumerate(numbers):
        for b in numbers[i + 1:]:
            counts[0][a + b] += 1
            counts[2][a * b] += 1
            counts[1][a - b] += 1
            counts[1][b - a] += 1
            if b:
                counts[3][a / b] += 1
            if a:
                counts[3][b / a] += 1
    return counts


def answer_features(answer, numbers, variant, counts=None):
    feature_names(variant)
    answer = Fraction(answer)
    magnitude = float(min(abs(answer), LIMIT))
    maximum = max((abs(value) for value in numbers), default=Fraction(0))
    divisor = float(max(maximum, Fraction(1, LIMIT))) if maximum else 1.
    features = [float((answer > 0) - (answer < 0)), math.log1p(magnitude),
        float(answer.denominator == 1), float(answer in numbers), math.log1p(len(numbers)),
        math.log1p(magnitude / divisor)]
    if variant == 'operation_support':
        if counts is None:
            raise ValueError('Compute operation counts once per question')
        matches = [counter[answer] for counter in counts]
        features += [float(value > 0) for value in matches] + [math.log1p(value) for value in matches]
    return np.asarray(features, dtype=np.float64)


def candidate_values(bank, variant):
    values = np.zeros((len(bank), 4, len(feature_names(variant))), dtype=np.float64)
    for q, (row, answers) in enumerate(zip(bank, candidate_answers(bank))):
        numbers = question_numbers(row['question'])
        counts = operation_counts(numbers) if variant == 'operation_support' else None
        by_answer = {answer: answer_features(answer, numbers, variant, counts)
                     for answer in set(answers) if answer is not None}
        for i, answer in enumerate(answers):
            if answer is not None:
                values[q, i] = by_answer[answer]
    return values
