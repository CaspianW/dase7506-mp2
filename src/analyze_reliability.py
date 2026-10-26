"""Reproduce descriptive reliability diagnostics from fixed development OOF scores."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='4'
os.environ['OMP_NUM_THREADS']='1'
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from selection_utils import candidate_answers, disagreement_mask
from submission_ensemble import choose_scores
from train_surface import load_training_rows, write_json
EDGES=(0.,.25,.5,1.,2.,4.,8.,np.inf)
COVERAGES=(100,90,75,50,25,10)

def gaps_and_counts(scores, answers):
    gaps = np.full(len(answers), np.nan)
    counts = np.zeros(len(answers), dtype=int)
    for q, row in enumerate(answers):
        groups = {}
        for i, answer in enumerate(row):
            if answer is not None:
                groups.setdefault(answer, []).append(i)
        values = []
        for members in groups.values():
            block = np.asarray(scores[q, members])
            np.testing.assert_array_equal(block, np.full(len(members), block[0]))
            values.append(float(block[0]))
        counts[q] = len(values)
        if len(values) >= 2:
            ordered = sorted(values, reverse=True)
            gaps[q] = ordered[0]-ordered[1]
    return gaps, counts

def auc_by_ranks(correct, values):
    correct = np.asarray(correct, dtype=bool)
    positive = int(correct.sum()); negative = len(correct)-positive
    if not positive or not negative:
        return None
    ranks = rankdata(values, method='average')
    return float((ranks[correct].sum()-positive*(positive+1)/2)/(positive*negative))

def count_accuracy(correct, selected):
    n = int(np.sum(selected)); count = int(np.sum(correct[selected]))
    return dict(n=n, correct=count, wrong=n-count, accuracy=100*count/n if n else None)

def confidence_summary(correct, gaps):
    correct, gaps = np.asarray(correct, dtype=bool), np.asarray(gaps, dtype=float)
    assert np.isfinite(gaps).all() and np.all(gaps >= 0)
    result = dict(count_accuracy(correct, np.ones(len(correct), dtype=bool)),
        gap_AUROC_for_correctness=auc_by_ranks(correct, gaps), bins=[], selective_coverage=[])
    if len(gaps):
        result['gap_percentiles'] = {str(p):float(np.percentile(gaps,p)) for p in (0,10,25,50,75,90,100)}
    else:
        result['gap_percentiles'] = {}
    for lo, hi in zip(EDGES, EDGES[1:]):
        selected = (gaps >= lo)&(gaps < hi)
        result['bins'].append(dict(left=lo, right=hi if np.isfinite(hi) else 'infinity',
            **count_accuracy(correct, selected)))
    assert sum(b['n'] for b in result['bins']) == len(correct)
    for target in COVERAGES:
        if not len(gaps):
            result['selective_coverage'].append(dict(target_percent=target, cutoff=None,
                actual_percent=None, **count_accuracy(correct, np.zeros(0,dtype=bool))))
            continue
        wanted = int(np.ceil(len(gaps)*target/100))
        cutoff = float(np.sort(gaps)[-wanted])
        selected = gaps >= cutoff
        result['selective_coverage'].append(dict(target_percent=target, cutoff=cutoff,
            actual_percent=100*int(selected.sum())/len(gaps), **count_accuracy(correct,selected)))
    if np.any(correct) and np.any(~correct):
        independent = float(roc_auc_score(correct,gaps))
        np.testing.assert_allclose(result['gap_AUROC_for_correctness'],independent,atol=1e-14,rtol=0)
        result['AUC_independent_sklearn_check_passed'] = True
    else:
        result['AUC_independent_sklearn_check_passed'] = None
    return result

def agreement_summary(labels, choices, answers, indices):
    indices = np.asarray(indices, dtype=int)
    selected_S, selected_L = choices['S'][indices], choices['local_arithmetic'][indices]
    S = labels[indices, selected_S]; L = labels[indices, selected_L]
    agree = np.asarray([answers[q][a] == answers[q][b] for q,a,b in zip(indices,selected_S,selected_L)],dtype=bool)
    result = {}
    for name, mask in (('agree',agree), ('disagree',~agree)):
        result[name] = dict(n=int(mask.sum()), both_correct=int(np.sum(mask&S&L)),
            S_only_correct=int(np.sum(mask&S&~L)), L_only_correct=int(np.sum(mask&~S&L)),
            both_wrong=int(np.sum(mask&~S&~L)),
            S_accuracy=100*int(np.sum(mask&S))/int(mask.sum()) if mask.sum() else None,
            L_accuracy=100*int(np.sum(mask&L))/int(mask.sum()) if mask.sum() else None)
        assert result[name]['n'] == sum(result[name][key] for key in
            ('both_correct','S_only_correct','L_only_correct','both_wrong'))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--oof-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh output directory.')
    protocol=json.loads((Path(__file__).resolve().parent.parent/'results/protocol.json').read_text(encoding='utf-8'))
    bank,arrays,labels=load_training_rows(args.data_dir,protocol['development_indices'])
    answers=candidate_answers(bank);hard=disagreement_mask(answers)
    results={};choices={}
    for name in ('S','local_arithmetic'):
        with np.load(args.oof_dir/f'{name}_oof_scores.npz') as saved:scores=saved['scores']
        assert scores.shape==(len(bank),4)
        choices[name]=np.asarray(choose_scores(scores,bank,arrays))
        correct=labels[np.arange(len(bank)),choices[name]]
        gaps,_=gaps_and_counts(scores,answers)
        results[name]=confidence_summary(correct[hard],gaps[hard])
    result=dict(hard_confidence=results,agreement=agreement_summary(labels,choices,answers,np.flatnonzero(hard)),
        training_fits=0,official_test_used=False,interpretation='Post-selection descriptive OOF analysis. Gaps are not calibrated probabilities. Selective accuracy is not a full-bank official score.')
    args.output.mkdir(parents=True);write_json(args.output/'summary.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
