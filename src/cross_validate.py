"""Reproduce fixed S and L five-fold results on training development rows."""

import argparse
import hashlib
import json
from pathlib import Path

from train import fit_local, load_training_rows, write_json
import numpy as np
from submission import candidate_scores
from submission_question_operations import candidate_scores as surface_scores
from submission_ensemble import choose_scores
from selection_utils import candidate_answers, selection_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).resolve().parent.parent / 'results/protocol.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a fresh output directory.')
    args.output.mkdir(parents=True)
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    manifest = json.loads((args.data_dir / 'manifest.json').read_text(encoding='utf-8'))
    assert all(manifest['sha256'][key] == value for key, value in protocol['source_sha256'].items())
    indices = np.asarray(protocol['development_indices'])
    bank, arrays, labels = load_training_rows(args.data_dir, indices.tolist())
    scores = {name: np.full((len(bank),4),np.nan) for name in ('S','local_arithmetic')}
    seen, folds = np.zeros(len(bank),dtype=int), {name: [] for name in scores}
    for fold in protocol['folds']:
        fit, valid = np.asarray(fold['fit']), np.asarray(fold['valid'])
        assert not set(fit) & set(valid)
        fit_bank, valid_bank = [bank[i] for i in fit], [bank[i] for i in valid]
        fit_arrays, valid_arrays = ({key:value[idx] for key,value in arrays.items()} for idx in (fit,valid))
        model,_,_,_=fit_local(fit_bank,fit_arrays,labels[fit],indices[fit],args.output/f'fold{fold["fold"]}')
        scores['S'][valid]=surface_scores(valid_bank,valid_arrays,model['anchor_raw'],'numeric_surface')
        scores['local_arithmetic'][valid]=candidate_scores(valid_bank,valid_arrays,model)
        seen[valid]+=1
        for name in scores:
            metrics=selection_metrics(labels[valid],choose_scores(scores[name][valid],valid_bank,valid_arrays),candidate_answers(valid_bank))
            folds[name].append({'fold':fold['fold'],'metrics':metrics})
    assert (seen==1).all()
    results={name:{'pooled':selection_metrics(labels,choose_scores(value,bank,arrays),candidate_answers(bank)),
        'folds':folds[name]} for name,value in scores.items()}
    receipt={'results':results,'training_fits':{'S':5,'residual':5},
        'protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        'coordinator_holdout_used':False,'official_validation_used':False,'official_test_used':False}
    for name,value in scores.items():np.savez(args.output/f'{name}_oof_scores.npz',scores=value)
    write_json(args.output/'summary.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
