"""Reproduce the fixed five-fold comparison using training data only."""

import argparse
import hashlib
import json
from pathlib import Path

from train import (build_pairs, fit_question_operations, load_training_rows, write_json,
                   LogisticRegression, threadpool_limits)
import numpy as np
from submission import candidate_scores
from submission_ensemble import choose_scores
from selection_utils import candidate_answers, selection_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).resolve().parent.parent / 'results/protocol.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--without-numeric-surface', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory.')
    args.output.mkdir(parents=True)
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    manifest = json.loads((args.data_dir / 'manifest.json').read_text(encoding='utf-8'))
    assert all(manifest['sha256'][key] == value for key, value in protocol['source_sha256'].items())
    indices = np.asarray(protocol['development_indices'])
    bank, arrays, labels = load_training_rows(args.data_dir, indices.tolist())
    scores, seen, folds = np.full((len(bank), 4), np.nan), np.zeros(len(bank), dtype=int), []
    for fold in protocol['folds']:
        fit, valid = np.asarray(fold['fit']), np.asarray(fold['valid'])
        assert not set(fit) & set(valid)
        fit_bank, valid_bank = [bank[i] for i in fit], [bank[i] for i in valid]
        fit_arrays, valid_arrays = ({key: value[idx] for key, value in arrays.items()} for idx in (fit, valid))
        if args.without_numeric_surface:
            X, y, weights, scaler, _, _ = build_pairs(fit_bank, fit_arrays, labels[fit],
                'subsets_equal_question', indices[fit])
            classifier = LogisticRegression(C=1 / (.01 * len(y)), fit_intercept=False,
                solver='lbfgs', max_iter=2000, tol=1e-6)
            with threadpool_limits(limits=4):
                classifier.fit(X, y, sample_weight=weights)
            model = {'raw_w': np.r_[classifier.coef_[0] / scaler.scale_, np.zeros(6)]}
        else:
            model, _, _, _ = fit_question_operations(fit_bank, fit_arrays, labels[fit],
                'numeric_surface', indices[fit], args.output / f'fold{fold["fold"]}')
        scores[valid] = candidate_scores(valid_bank, valid_arrays, model['raw_w'], 'numeric_surface')
        seen[valid] += 1
        metrics = selection_metrics(labels[valid], choose_scores(scores[valid], valid_bank, valid_arrays), candidate_answers(valid_bank))
        folds.append({'fold': fold['fold'], 'metrics': metrics})
    assert (seen == 1).all()
    result = {'pooled': selection_metrics(labels, choose_scores(scores, bank, arrays), candidate_answers(bank)),
        'folds': folds, 'training_fits': 5,
        'numeric_surface_features': not args.without_numeric_surface,
        'protocol_sha256': hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        'coordinator_holdout_used': False, 'official_validation_used': False, 'official_test_used': False}
    np.savez(args.output / 'oof_scores.npz', scores=scores)
    write_json(args.output / 'summary.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
