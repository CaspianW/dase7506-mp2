"""Reproduce the fixed surface anchor and local-arithmetic residual."""

import os
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '4'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

from quantity_flow_features import cached_values, group_features, variant_values
from submission_answer_groups import representations
from submission_ensemble import choose_scores
from submission_question_operations import candidate_scores as surface_scores
from train_surface import fit_question_operations, joint_pairs, load_training_rows, write_json

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
OPTIONS = {'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-6, 'maxls': 40}
VARIANT = 'local_arithmetic'
VARIANTS = (VARIANT,)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_fixed_sources():
    provenance = json.loads((ROOT / 'results/source_provenance.json').read_text(encoding='utf-8'))
    for entry in provenance['mapping'].values():
        if entry['destination'].startswith('src/') and entry['byte_exact']:
            if sha(ROOT / entry['destination']) != entry['sha256']:
                raise ValueError('Frozen inference or feature source changed')
    if sha(CODE / 'train_surface.py') != provenance['retained_base_S_trainer_sha256']:
        raise ValueError('Frozen surface trainer changed')


def build_full_pairs(bank, arrays, labels, source, anchor_raw):
    full = representations(bank, arrays, 'mean')
    anchor_scores = surface_scores(bank, arrays, anchor_raw, 'numeric_surface')
    q_indices, positives, negatives, pmasks, nmasks, margins, raw_weights = [], [], [], [], [], [], []
    pair_counts = np.zeros(len(bank), dtype=np.int64)
    for q, groups in enumerate(full):
        good, bad = [], []
        for g, (_, indices) in enumerate(groups):
            truth = labels[q, indices]
            if not np.all(truth == truth[0]):
                raise ValueError('Numeric group labels disagree')
            (good if truth[0] else bad).append(g)
        pair_counts[q] = len(good) * len(bad)
        if not pair_counts[q]:
            continue
        for p in good:
            for n in bad:
                pi, ni = groups[p][1], groups[n][1]
                q_indices.append(q)
                positives.append(p)
                negatives.append(n)
                pmasks.append(sum(1 << i for i in pi))
                nmasks.append(sum(1 << i for i in ni))
                margins.append(anchor_scores[q, pi[0]] - anchor_scores[q, ni[0]])
                raw_weights.append(1. / pair_counts[q])
    if not q_indices:
        raise ValueError('Full-group residual training requires informative questions')
    q_indices = np.asarray(q_indices, dtype=np.int64)
    margins = np.asarray(margins)
    weights = np.tile(raw_weights, 2).astype(np.float64)
    weights *= len(weights) / weights.sum()
    totals = np.bincount(np.tile(q_indices, 2), weights=weights, minlength=len(bank))
    informative = pair_counts > 0
    np.testing.assert_allclose(totals[informative], len(weights) / informative.sum(), atol=1e-12, rtol=1e-12)
    audit = {'unordered_pair_question_indices': q_indices,
        'unordered_pair_source_indices': np.asarray(source)[q_indices],
        'positive_group_indices': np.asarray(positives), 'negative_group_indices': np.asarray(negatives),
        'positive_subset_masks': np.asarray(pmasks, dtype=np.int8),
        'negative_subset_masks': np.asarray(nmasks, dtype=np.int8), 'source_indices': np.asarray(source),
        'ordered_sample_weight': weights, 'question_total_weight': totals,
        'full_pairs_per_question': pair_counts, 'anchor_margin': np.r_[margins, -margins],
        'y': np.r_[np.ones(len(margins)), np.zeros(len(margins))]}
    return audit


def fingerprint(bank, arrays, labels, source):
    digest = hashlib.sha256(json.dumps(bank, sort_keys=True, ensure_ascii=False,
                                      separators=(',', ':')).encode('utf-8'))
    for value in (np.asarray(arrays['embeddings']), np.asarray(labels), np.asarray(source)):
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def prepare_local(bank, arrays, labels, source_indices=None, anchor_fit=None, cache=None):
    source = np.arange(len(bank)) if source_indices is None else np.asarray(source_indices)
    if anchor_fit is None:
        with threadpool_limits(limits=4):
            anchor_fit = fit_question_operations(bank, arrays, labels, 'numeric_surface', source)
    anchor, scalers, record, anchor_audit = anchor_fit
    X, _, _, expected_scalers, _, expected_audit, _ = joint_pairs(bank, arrays, labels, 'numeric_surface', source)
    for name in scalers:
        np.testing.assert_array_equal(scalers[name].mean_, expected_scalers[name].mean_)
        np.testing.assert_array_equal(scalers[name].scale_, expected_scalers[name].scale_)
    for name in expected_audit:
        np.testing.assert_array_equal(anchor_audit[name], expected_audit[name])
    del X
    table = cached_values(bank, arrays, source, cache)
    full_audit = build_full_pairs(bank, arrays, labels, source, anchor['raw_w'])
    return {'table': table, 'anchor_raw': anchor['raw_w'].copy(), 'full_audit': full_audit,
        'fingerprint': fingerprint(bank, arrays, labels, source), 'source_indices': source}


def residual_design(bank, arrays, prepared, variant):
    groups = group_features(bank, arrays, variant_values(prepared['table'], variant))
    original = np.stack([feature for row in groups for feature, _ in row])
    scaler = StandardScaler().fit(original)
    audit = prepared['full_audit']
    delta = np.stack([(groups[q][p][0] - groups[q][n][0]) / scaler.scale_
        for q, p, n in zip(audit['unordered_pair_question_indices'],
            audit['positive_group_indices'], audit['negative_group_indices'])])
    return np.r_[delta, -delta], scaler, original


def objective(v, X, anchor_margin, y, weights):
    sign = 2 * y - 1
    margin = anchor_margin + X @ v
    normalized = weights / weights.sum()
    loss = float(normalized @ np.logaddexp(0., -sign * margin) + .005 * (v @ v))
    gradient = X.T @ (-normalized * sign * expit(-sign * margin)) + .01 * v
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise FloatingPointError('Nonfinite quantity-flow objective or gradient')
    return loss, gradient


def export_checkpoint(path, model):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / 'weights.npz', **{name: model[name] for name in ('anchor_raw', 'raw_v')})
    write_json(path / 'config.json', {'schema_version': 1, 'model': 'surface_quantity_flow',
        'variant': model['variant'],
        'lambda': .01, 'anchor_frozen': True, 'inference_uses_original_full_groups': True})


def fit_residual(bank, arrays, labels, variant, source_indices=None, output=None,
                        prepared=None, anchor_fit=None, cache=None):
    """Fit the fixed residual on caller rows only; never loads a data split."""
    if variant not in VARIANTS:
        raise ValueError('Unknown fixed residual variant')
    source = np.arange(len(bank)) if source_indices is None else np.asarray(source_indices)
    if prepared is None:
        prepared = prepare_local(bank, arrays, labels, source, anchor_fit, cache)
    if prepared['fingerprint'] != fingerprint(bank, arrays, labels, source):
        raise ValueError('Prepared caller rows differ')
    X, scaler, original = residual_design(bank, arrays, prepared, variant)
    audit = prepared['full_audit']
    arguments = (X, audit['anchor_margin'], audit['y'], audit['ordered_sample_weight'])
    initial = np.zeros(X.shape[1])
    initial_objective, initial_gradient = objective(initial, *arguments)
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez(output / 'initial.npz', standard_v=initial, objective=np.asarray(initial_objective),
                 gradient=initial_gradient)
        np.savez(output / 'training_pair_audit.npz', **audit)
        np.savez(output / 'fit_scalers.npz', residual_mean=scaler.mean_, residual_scale=scaler.scale_,
                 residual_var=scaler.var_, n_samples=np.asarray(scaler.n_samples_seen_))
        np.savez(output / 'original_group_features.npz', features=original)
    tick = time.perf_counter()
    try:
        with threadpool_limits(limits=4):
            result = minimize(objective, initial, args=arguments, jac=True, method='L-BFGS-B', options=OPTIONS)
    except Exception as error:
        if output is not None:
            write_json(output / 'failure.json', {'exception': type(error).__name__, 'message': str(error)})
        raise
    raw_v = result.x / scaler.scale_
    record = {'variant': variant, 'dimension': X.shape[1], 'success': bool(result.success),
        'status': int(result.status), 'message': str(result.message), 'iterations': int(result.nit),
        'evaluations': int(result.nfev), 'gradient_evaluations': int(result.njev),
        'initial_objective': initial_objective, 'objective': float(result.fun),
        'gradient_inf_norm': float(np.abs(result.jac).max()), 'options': OPTIONS,
        'seconds': time.perf_counter() - tick, 'original_full_group_scaler_rows': len(original),
        'full_ordered_pairs': len(audit['y']), 'anchor_frozen': True,
        'fitting_rows_sha256': prepared['fingerprint'], 'lambda': .01,
        'pair_loss_distinct_from_S_subset_augmentation': True}
    if output is not None:
        np.savez(output / 'residual_coefficients.npz', standard_v=result.x, raw_v=raw_v,
            initial_gradient=initial_gradient, gradient=result.jac,
            initial_objective=np.asarray(initial_objective), objective=np.asarray(result.fun))
        write_json(output / 'optimizer.json', record)
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.jac).all() \
            or not np.isfinite(result.fun) or not np.isfinite(raw_v).all():
        raise RuntimeError(f'{variant} solver failed: {result.message}')
    np.testing.assert_allclose(X @ result.x, (X * scaler.scale_) @ raw_v, atol=1e-11, rtol=1e-11)
    model = {'anchor_raw': prepared['anchor_raw'].copy(), 'raw_v': raw_v, 'variant': variant}
    if output is not None:
        export_checkpoint(output / 'checkpoint', model)
    return model, scaler, record, audit


def native_scores(bank, arrays, model, prepared, scaler, standard_v, table):
    result = surface_scores(bank, arrays, model['anchor_raw'], 'numeric_surface')
    for q, groups in enumerate(group_features(bank, arrays, variant_values(table, model['variant']))):
        for feature, indices in groups:
            result[q, indices] += ((feature - scaler.mean_) / scaler.scale_) @ standard_v
    return result


def margin_parity(bank, arrays, native, exported):
    maximum = 0.
    for q, groups in enumerate(representations(bank, arrays, 'mean')):
        indices = [idx[0] for _, idx in groups]
        if indices:
            n, e = native[q, indices], exported[q, indices]
            maximum = max(maximum, float(np.abs((n - n[0]) - (e - e[0])).max()))
    if maximum >= 1e-10 or choose_scores(native, bank, arrays) != choose_scores(exported, bank, arrays):
        raise AssertionError(f'Quantity-flow margin/choice export failed: {maximum}')
    return {'max_abs_group_margin_error': maximum, 'exact_choices': True}


def fit_local(bank, arrays, labels, source_indices=None, output=None):
    """Fit one S anchor and one residual using only caller-provided training rows."""
    source = np.arange(len(bank)) if source_indices is None else np.asarray(source_indices)
    if source.shape != (len(bank),) or len(set(source.tolist())) != len(bank):
        raise ValueError('Supply one distinct provenance index per fitting question')
    verify_fixed_sources()
    if output is not None:
        output = Path(output)
        if output.exists(): raise FileExistsError('Use a fresh fitting output directory')
        output.mkdir(parents=True)
    started = time.perf_counter()
    with threadpool_limits(limits=4):
        anchor_fit = fit_question_operations(bank, arrays, labels, 'numeric_surface', source,
            output / 'S' if output is not None else None)
        prepared = prepare_local(bank, arrays, labels, source, anchor_fit=anchor_fit)
        model, scaler, record, audit = fit_residual(bank, arrays, labels, VARIANT, source,
            output / VARIANT if output is not None else None, prepared=prepared)
        pools = threadpool_info()
        assert all(pool['num_threads'] <= 4 for pool in pools)
    if output is not None:
        export_checkpoint(output / 'checkpoint', model)
        questions, masks = [], []
        for q, groups in enumerate(representations(bank, arrays, 'mean')):
            for _, indices in groups: questions.append(q); masks.append(sum(1 << i for i in indices))
        np.savez(output / 'fitting_provenance.npz', source_indices=source, labels=labels,
            group_local_question_indices=np.asarray(questions), group_source_indices=source[questions],
            group_subset_masks=np.asarray(masks, dtype=np.int8))
        np.savez(output / 'candidate_feature_table.npz', table=prepared['table'], source_indices=source)
        X, _, _ = residual_design(bank, arrays, prepared, VARIANT)
        np.savez(output / VARIANT / 'training_design.npz', X=X)
        receipt = {'variant': VARIANT, 'training_questions': len(bank), 'fits': {'S': 1, 'residual': 1},
            'fit_seconds': time.perf_counter()-started, 'fitting_rows_sha256': prepared['fingerprint'],
            'source_sha256': {p.name: sha(p) for p in sorted(CODE.glob('*.py'))},
            'numeric_threadpools': pools, 'surface': anchor_fit[2], 'residual': record,
            'validation_used_for_fitting': False, 'test_used_for_fitting': False,
            'checkpoint_sha256': {p.name: sha(p) for p in sorted((output/'checkpoint').iterdir())}}
        write_json(output / 'training.json', receipt)
    return model, scaler, record, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a fresh output directory.')
    verify_fixed_sources()
    bank, arrays, labels = load_training_rows(args.data_dir)
    fit_local(bank, arrays, labels, np.arange(len(bank)), args.output)
    print(json.dumps({'training_questions': len(bank), 'variant': VARIANT,
        'checkpoint': str(args.output/'checkpoint'), 'fits': {'S': 1, 'residual': 1}}), flush=True)


if __name__ == '__main__': main()
