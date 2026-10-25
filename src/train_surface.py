"""Train the fixed numeric-surface selector using supplied training rows."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '4'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from common import check_data, load_inputs
from question_operation_features import candidate_values, feature_names
from selection_utils import make_targets
from submission_answer_groups import representations

RECIPES = ('full_uniform', 'full_equal_question', 'subsets_equal_question')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def load_training_rows(data_dir, indices=None):
    data_dir = Path(data_dir)
    check_data(data_dir, ('train',))
    bank, arrays = load_inputs('train', data_dir)
    indices = list(range(len(bank))) if indices is None else list(indices)
    wanted, references = set(indices), {}
    with (data_dir / 'gsm8k_train.jsonl').open(encoding='utf-8') as stream:
        for number, line in enumerate(stream):
            if number in wanted:
                references[number] = json.loads(line)
    selected = [bank[i] for i in indices]
    labels = make_targets(selected, [references[i] for i in indices])
    return selected, {key: value[indices] for key, value in arrays.items()}, labels

def subset_representations(values, full_groups):
    """Enumerate nonempty subsets inside each existing numeric-answer group."""
    result = []
    for _, indices in full_groups:
        subsets = []
        for size in range(1, len(indices) + 1):
            for chosen in combinations(indices, size):
                block = values[list(chosen)]
                mean = block.mean(axis=0)
                feature = np.concatenate([mean, [size, size / 4, len(full_groups),
                                               np.square(block - mean).mean()]])
                subsets.append((feature, list(chosen)))
        assert len(subsets) == 2 ** len(indices) - 1
        result.append(subsets)
    return result


def build_pairs(bank, arrays, labels, recipe, source_indices=None):
    if recipe not in RECIPES:
        raise ValueError('Unknown fixed recipe')
    embeddings = np.asarray(arrays['embeddings'], dtype=np.float64)
    full = representations(bank, arrays, 'mean')
    features = np.stack([feature for groups in full for feature, _ in groups])
    # All recipes share this scaler; augmented subsets never fit preprocessing.
    scaler = StandardScaler().fit(features)
    differences, questions, positive_masks, negative_masks, raw_weights = [], [], [], [], []
    full_pair_counts, actual_pair_counts, subset_counts = [], [], []
    informative = []
    for row_index, (groups, targets) in enumerate(zip(full, labels)):
        expanded = subset_representations(embeddings[row_index], groups) \
            if recipe == 'subsets_equal_question' else [[group] for group in groups]
        positive, negative = [], []
        num_positive_groups, num_negative_groups = 0, 0
        for (_, full_indices), subsets in zip(groups, expanded):
            truth = targets[full_indices]
            if not np.all(truth == truth[0]):
                raise AssertionError('Equivalent numeric answers must share their label')
            if truth[0]:
                num_positive_groups += 1
                positive.extend(subsets)
            else:
                num_negative_groups += 1
                negative.extend(subsets)
        full_count = num_positive_groups * num_negative_groups
        count = len(positive) * len(negative)
        full_pair_counts.append(full_count)
        actual_pair_counts.append(count)
        subset_counts.append(sum(len(group) for group in expanded))
        if not count:
            continue
        informative.append(row_index)
        question_weight = 1. if recipe == 'full_uniform' else 1 / (2 * count)
        for positive_feature, positive_indices in positive:
            for negative_feature, negative_indices in negative:
                assert targets[positive_indices].all() and not targets[negative_indices].any()
                positive_mask = sum(1 << i for i in positive_indices)
                negative_mask = sum(1 << i for i in negative_indices)
                assert positive_mask & negative_mask == 0
                differences.append((positive_feature - negative_feature) / scaler.scale_)
                questions.append(row_index)
                positive_masks.append(positive_mask)
                negative_masks.append(negative_mask)
                raw_weights.append(question_weight)
    if not differences:
        raise ValueError('At least one question must have correct and incorrect valid groups')
    positive_differences = np.stack(differences)
    X = np.concatenate([positive_differences, -positive_differences])
    y = np.concatenate([np.ones(len(differences)), np.zeros(len(differences))])
    sample_weight = np.asarray(raw_weights + raw_weights, dtype=np.float64)
    sample_weight *= len(sample_weight) / sample_weight.sum()
    question_array = np.asarray(questions + questions, dtype=np.int64)
    totals = np.bincount(question_array, weights=sample_weight, minlength=len(bank))
    assert np.isclose(sample_weight.sum(), len(sample_weight), rtol=1e-14)
    assert np.all(totals[informative] > 0)
    if recipe != 'full_uniform':
        np.testing.assert_allclose(totals[informative], len(sample_weight) / len(informative),
                                   rtol=1e-13, atol=1e-13)
    source = np.arange(len(bank)) if source_indices is None else np.asarray(source_indices)
    if source.shape != (len(bank),) or len(set(source.tolist())) != len(bank):
        raise ValueError('Supply one distinct provenance index per fitting question')
    signature = [{'source_index': int(source[i]), 'id': bank[i]['id']} for i in informative]
    record = {'recipe': recipe, 'lambda': .01, 'original_fit_questions': len(bank),
              'informative_fit_questions': len(informative),
              'augmentation_fit_questions': len(informative) if recipe == 'subsets_equal_question' else 0,
              'original_fit_groups': len(features), 'training_subset_groups': sum(subset_counts),
              'original_unordered_pairs': sum(full_pair_counts),
              'training_unordered_pairs': len(differences), 'training_ordered_pairs': len(y),
              'sample_weight_sum': float(sample_weight.sum()), 'sample_weight_mean': float(sample_weight.mean()),
              'question_total_weight_min': float(totals[informative].min()),
              'question_total_weight_max': float(totals[informative].max()),
              'sample_weight_sha256': hashlib.sha256(sample_weight.tobytes()).hexdigest(),
              'question_totals_sha256': hashlib.sha256(totals.tobytes()).hexdigest(),
              'effective_question_id_index_sha256': hashlib.sha256(
                  json.dumps(signature, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
              'scaler_fit_only_original_full_groups': True,
              'scaler_mean_sha256': hashlib.sha256(scaler.mean_.tobytes()).hexdigest(),
              'scaler_scale_sha256': hashlib.sha256(scaler.scale_.tobytes()).hexdigest(),
              'same_question_pairs_only': True, 'positive_negative_labels_checked': True}
    audit = {'unordered_pair_question_indices': np.asarray(questions, dtype=np.int64),
             'unordered_positive_subset_masks': np.asarray(positive_masks, dtype=np.int8),
             'unordered_negative_subset_masks': np.asarray(negative_masks, dtype=np.int8),
             'sample_weight': sample_weight, 'question_total_weight': totals,
             'source_indices': source, 'full_unordered_pairs_per_question': np.asarray(full_pair_counts),
             'actual_unordered_pairs_per_question': np.asarray(actual_pair_counts)}
    return X, y, sample_weight, scaler, record, audit


def objective(w, X, y, sample_weight):
    sign = 2 * y - 1
    margin = sign * (X @ w)
    normalized = sample_weight / sample_weight.sum()
    value = float(normalized @ np.logaddexp(0, -margin) + .005 * (w @ w))
    gradient = X.T @ (-normalized * expit(-margin) * sign) + .01 * w
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise FloatingPointError('Nonfinite objective or gradient')
    return value, gradient


def joint_pairs(bank, arrays, labels, variant, source_indices=None):
    feature_names(variant)
    X, y, weights, base_scaler, record, audit = build_pairs(
        bank, arrays, labels, 'subsets_equal_question', source_indices)
    full = representations(bank, arrays, 'mean')
    values = candidate_values(bank, variant)
    original = np.stack([values[q, indices[0]]
                         for q, groups in enumerate(full) for _, indices in groups])
    auxiliary_scaler = StandardScaler().fit(original)
    differences = []
    for q, pm, nm in zip(audit['unordered_pair_question_indices'],
            audit['unordered_positive_subset_masks'], audit['unordered_negative_subset_masks']):
        positive = next(i for i in range(4) if int(pm) & (1 << i))
        negative = next(i for i in range(4) if int(nm) & (1 << i))
        differences.append((values[q, positive] - values[q, negative]) / auxiliary_scaler.scale_)
    delta = np.stack(differences)
    Z = np.concatenate([delta, -delta])
    design = np.concatenate([X, Z], axis=1)
    np.testing.assert_array_equal(design[:, :1028], X)
    return design, y, weights, {'base': base_scaler, 'auxiliary': auxiliary_scaler}, record, audit, original


def export_checkpoint(path, weights, variant):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / 'weights.npz', w=weights)
    write_json(path / 'config.json', {'schema_version': 1, 'model': 'answer_group_question_operations',
        'variant': variant, 'feature_names': feature_names(variant),
        'base_recipe': 'subsets_equal_question', 'lambda': .01, 'no_intercept': True,
        'jointly_fitted': True, 'inference_uses_full_original_groups': True,
        'scale_folded_into_weights': True})


def fit_question_operations(bank, arrays, labels, variant, source_indices=None, output=None):
    """Fit only caller-supplied rows; return model, scalers, record and pair audit."""
    X, y, weights, scalers, record, audit, original = joint_pairs(
        bank, arrays, labels, variant, source_indices)
    classifier = LogisticRegression(C=1 / (.01 * len(y)), fit_intercept=False,
                                    solver='lbfgs', max_iter=2000, tol=1e-6)
    with warnings.catch_warnings(), threadpool_limits(limits=4):
        warnings.simplefilter('error', ConvergenceWarning)
        classifier.fit(X, y, sample_weight=weights)
    standard_w = classifier.coef_[0].copy()
    if not np.isfinite(standard_w).all():
        raise FloatingPointError('Nonfinite joint coefficients')
    scales = np.concatenate([scalers['base'].scale_, scalers['auxiliary'].scale_])
    raw_w = standard_w / scales
    reference = X @ standard_w
    exported = (X * scales) @ raw_w
    np.testing.assert_allclose(reference, exported, atol=1e-10, rtol=1e-11)
    value, gradient = objective(standard_w, X, y, weights)
    record.update({'variant': variant, 'dimension': X.shape[1], 'jointly_fitted': True,
        'C': classifier.C, 'iterations': int(classifier.n_iter_[0]), 'success': True,
        'solver': 'sklearn LogisticRegression lbfgs', 'max_iter': 2000, 'tol': 1e-6,
        'warm_start': False, 'zero_initialization': True, 'objective': value,
        'gradient_inf_norm': float(np.abs(gradient).max()), 'original_auxiliary_groups': len(original),
        'auxiliary_scaler_includes_G1': True, 'auxiliary_scaler_fit_augmented_subsets': False,
        'auxiliary_scaler_mean_sha256': hashlib.sha256(scalers['auxiliary'].mean_.tobytes()).hexdigest(),
        'auxiliary_scaler_scale_sha256': hashlib.sha256(scalers['auxiliary'].scale_.tobytes()).hexdigest(),
        'export_pair_score_max_abs_error': float(np.abs(reference - exported).max())})
    model = {'raw_w': raw_w, 'standard_w': standard_w, 'variant': variant}
    if output is not None:
        output = Path(output)
        export_checkpoint(output / 'checkpoint', raw_w, variant)
        np.savez(output / 'fit_weights.npz', raw_w=raw_w, standard_w=standard_w,
                 objective_gradient=gradient)
        np.savez(output / 'fit_scalers.npz', base_mean=scalers['base'].mean_,
            base_scale=scalers['base'].scale_, auxiliary_mean=scalers['auxiliary'].mean_,
            auxiliary_scale=scalers['auxiliary'].scale_)
        np.savez(output / 'training_pair_audit.npz', **audit)
        write_json(output / 'training.json', record)
    return model, scalers, record, audit



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory.')
    bank, arrays, labels = load_training_rows(args.data_dir)
    start = time.perf_counter()
    model, scalers, record, audit = fit_question_operations(bank, arrays, labels, 'numeric_surface', list(range(len(bank))), args.output)
    record.update({'training_questions': len(bank), 'fit_seconds': time.perf_counter() - start,
                   'validation_used_for_fitting': False, 'test_used_for_fitting': False})
    write_json(args.output / 'training.json', record)
    print(json.dumps({'training_questions': len(bank), 'dimension': len(model['raw_w']), 'fit_seconds': record['fit_seconds']}))


if __name__ == '__main__':
    main()
