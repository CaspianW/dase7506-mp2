"""Reproduce fixed pre-test C1028/S/format ablations, without model selection."""

import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='4'
os.environ['OMP_NUM_THREADS']='1'

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from train_surface import build_pairs, fit_question_operations, load_training_rows, write_json
from train import prepare_local, residual_design, objective, OPTIONS
from submission_answer_groups import candidate_scores as group_scores
from submission_question_operations import candidate_scores as surface_scores
from submission import candidate_scores as residual_scores
from submission_ensemble import choose_scores
from selection_utils import candidate_answers, selection_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=Path(__file__).resolve().parent.parent/'results/protocol.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh output directory.')
    protocol=json.loads(args.protocol.read_text(encoding='utf-8'))
    source=np.asarray(protocol['development_indices'])
    bank,arrays,labels=load_training_rows(args.data_dir,source.tolist())
    scores={name:np.full((len(bank),4),np.nan) for name in ('group_only_C1028','S','format_only')}
    args.output.mkdir(parents=True)
    folds=[]
    with threadpool_limits(limits=4):
        for fold in protocol['folds']:
            fit,valid=np.asarray(fold['fit']),np.asarray(fold['valid'])
            fb,vb=[bank[i] for i in fit],[bank[i] for i in valid]
            fa,va=({key:value[idx] for key,value in arrays.items()} for idx in (fit,valid))
            X,y,weight,scaler,_,_=build_pairs(fb,fa,labels[fit],'subsets_equal_question',source[fit])
            C=LogisticRegression(C=1/(.01*len(y)),fit_intercept=False,solver='lbfgs',max_iter=2000,tol=1e-6)
            C.fit(X,y,sample_weight=weight)
            scores['group_only_C1028'][valid]=group_scores(vb,va,'mean',C.coef_[0]/scaler.scale_)
            Sfit=fit_question_operations(fb,fa,labels[fit],'numeric_surface',source[fit])
            anchor=Sfit[0]['raw_w']
            scores['S'][valid]=surface_scores(vb,va,anchor,'numeric_surface')
            prepared=prepare_local(fb,fa,labels[fit],source[fit],anchor_fit=Sfit)
            X,scaler,_=residual_design(fb,fa,prepared,'format_only');audit=prepared['full_audit']
            result=minimize(objective,np.zeros(6),args=(X,audit['anchor_margin'],audit['y'],audit['ordered_sample_weight']),
                jac=True,method='L-BFGS-B',options=OPTIONS)
            if not result.success:raise RuntimeError(result.message)
            model=dict(anchor_raw=anchor,raw_v=result.x/scaler.scale_,variant='format_only')
            scores['format_only'][valid]=residual_scores(vb,va,model)
            folds.append(dict(fold=fold['fold'],results={name:selection_metrics(labels[valid],
                choose_scores(value[valid],vb,va),candidate_answers(vb)) for name,value in scores.items()}))
    summary=dict(purpose='Reproduction of fixed pre-test ablations; no tuning or selection',
        results={name:selection_metrics(labels,choose_scores(value,bank,arrays),candidate_answers(bank)) for name,value in scores.items()},
        folds=folds,model_fits=dict(group_only_C1028=5,S=5,format_only=5),official_test_used=False)
    for name,value in scores.items():np.savez(args.output/f'{name}_oof_scores.npz',scores=value)
    write_json(args.output/'summary.json',summary)
    print(json.dumps(summary['results']))


if __name__=='__main__':main()
