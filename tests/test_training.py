"""Synthetic reproduction checks; no supplied data or actual optimization."""

import argparse
import copy
from itertools import permutations
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'src'))

import numpy as np
from submission import candidate_scores, predict, load_checkpoint
from submission_answer_groups import representations
from submission_ensemble import choose_scores
from submission_question_operations import candidate_scores as surface_scores
from selection_utils import candidate_answers
from train import (OPTIONS, VARIANT, verify_fixed_sources, fingerprint, prepare_local,
    residual_design, objective, fit_residual, fit_local, export_checkpoint, native_scores, margin_parity, sha)
from train_surface import joint_pairs


class ReproductionTests(unittest.TestCase):
    mocked_residual_calls=0

    @classmethod
    def setUpClass(cls):
        texts=[['2+3=5\n#### 5','2+3=6\n#### 5','3+3=6\n#### 6','3+3=7\n#### 6'],
            ['1+1=2\n#### 2','1+2=3\n#### 3','2+2=4\n#### 4','invalid'],
            ['1+2=3\n#### 3','1+2=4\n#### 3','2+1=3\n#### 3','3+0=3\n#### 3'],
            ['1+0=1\n#### 1','1+1=2\n#### 2','1+2=3\n#### 3','1+3=4\n#### 4']]
        cls.bank=[{'id':f'unit-{q}','question':'Synthetic inputs 2 and 3.','candidates':[
            {'index':i,'text':text} for i,text in enumerate(row)]} for q,row in enumerate(texts)]
        cls.arrays={'embeddings':np.random.default_rng(763).normal(size=(4,4,1024))}
        cls.labels=np.asarray([[1,1,0,0],[1,0,0,0],[1,1,1,1],[1,0,0,0]],dtype=bool)
        cls.source=np.asarray([17,26,35,49])
        _,_,_,scalers,record,audit,_=joint_pairs(cls.bank,cls.arrays,cls.labels,'numeric_surface',cls.source)
        standard=np.random.default_rng(638).normal(size=1034)*.01
        cls.anchor={'raw_w':standard/np.r_[scalers['base'].scale_,scalers['auxiliary'].scale_],
            'standard_w':standard,'variant':'numeric_surface'}
        cls.anchor_fit=(cls.anchor,scalers,record,audit)
        cls.prepared=prepare_local(cls.bank,cls.arrays,cls.labels,cls.source,anchor_fit=cls.anchor_fit)

    def test_frozen_inference_features_base_trainer_and_dependency_pins(self):
        verify_fixed_sources()
        provenance=json.loads((ROOT/'results/source_provenance.json').read_text())
        self.assertEqual(provenance['snapshot_files_verified'],15)
        self.assertEqual(sum(p.stat().st_size for p in (ROOT/'checkpoint').iterdir()),9157)
        manifest=json.loads((ROOT/'results/candidate_snapshot_manifest.json').read_text())
        for name in ('config.json','weights.npz'):
            self.assertEqual(sha(ROOT/'checkpoint'/name),manifest['files']['checkpoint/'+name]['sha256'])
        self.assertEqual((ROOT/'requirements.txt').read_text().strip(),'numpy==2.3.5')
        for pin in ('scipy==1.16.3','scikit-learn==1.7.2','threadpoolctl==3.6.0'):
            self.assertIn(pin,(ROOT/'requirements-train.txt').read_text())

    def test_original_group_scaler_and_fullpair_provenance(self):
        X,scaler,features=residual_design(self.bank,self.arrays,self.prepared,VARIANT)
        self.assertEqual(features.shape,(10,22));self.assertEqual(X.shape,(12,22))
        self.assertEqual(scaler.n_samples_seen_,10)
        np.testing.assert_allclose(scaler.mean_,features.mean(axis=0),atol=1e-14)
        np.testing.assert_array_equal(X[6:],-X[:6])
        audit=self.prepared['full_audit']
        np.testing.assert_array_equal(audit['full_pairs_per_question'],[1,2,0,3])
        np.testing.assert_allclose(audit['question_total_weight'],[4.,4.,0.,4.])
        groups=representations(self.bank,self.arrays,'mean')
        S=surface_scores(self.bank,self.arrays,self.anchor['raw_w'],'numeric_surface')
        expected=np.asarray([S[q,groups[q][p][1][0]]-S[q,groups[q][n][1][0]] for q,p,n in
            zip(audit['unordered_pair_question_indices'],audit['positive_group_indices'],audit['negative_group_indices'])])
        np.testing.assert_array_equal(audit['anchor_margin'],np.r_[expected,-expected])
        np.testing.assert_array_equal(audit['source_indices'],self.source)
        np.testing.assert_array_equal(audit['unordered_pair_source_indices'],self.source[audit['unordered_pair_question_indices']])

    def test_objective_gradient_and_weight_rescaling(self):
        X,_,_=residual_design(self.bank,self.arrays,self.prepared,VARIANT)
        a=self.prepared['full_audit'];v=np.random.default_rng(29).normal(size=22)*.03
        args=(X,a['anchor_margin'],a['y'],a['ordered_sample_weight'])
        value,gradient=objective(v,*args)
        other=objective(v,X,a['anchor_margin'],a['y'],a['ordered_sample_weight']*7)
        self.assertAlmostEqual(value,other[0],places=14);np.testing.assert_allclose(gradient,other[1],atol=1e-14)
        for i in range(22):
            direction=np.zeros(22);direction[i]=1.
            numeric=(objective(v+1e-6*direction,*args)[0]-objective(v-1e-6*direction,*args)[0])/2e-6
            self.assertAlmostEqual(numeric,gradient[i],places=8)

    def test_mocked_residual_artifacts_and_native_export_parity(self):
        def optimizer(function,initial,args,jac,method,options):
            ReproductionTests.mocked_residual_calls+=1
            self.assertEqual(method,'L-BFGS-B');self.assertTrue(jac);self.assertEqual(options,OPTIONS)
            np.testing.assert_array_equal(initial,np.zeros(22))
            v=np.arange(22)*.001;value,gradient=function(v,*args)
            return SimpleNamespace(x=v,fun=value,jac=gradient,success=True,status=0,
                message='Explicit synthetic mock without optimization',nit=0,nfev=1,njev=1)
        with tempfile.TemporaryDirectory() as temporary:
            with patch('train.minimize',side_effect=optimizer):
                model,scaler,_,_=fit_residual(self.bank,self.arrays,self.labels,VARIANT,self.source,
                    temporary,prepared=self.prepared)
            with np.load(Path(temporary)/'residual_coefficients.npz') as saved:
                np.testing.assert_array_equal(saved['raw_v'],saved['standard_v']/scaler.scale_)
                native=native_scores(self.bank,self.arrays,model,self.prepared,scaler,saved['standard_v'],self.prepared['table'])
            margin_parity(self.bank,self.arrays,native,candidate_scores(self.bank,self.arrays,model))
            self.assertEqual((Path(temporary)/'checkpoint/config.json').read_bytes(),(ROOT/'checkpoint/config.json').read_bytes())
            _,reloaded=load_checkpoint(Path(temporary)/'checkpoint')
            for key in ('anchor_raw','raw_v'):np.testing.assert_array_equal(reloaded[key],model[key])

    def test_fit_local_wrapper_calls_fixed_stages_once_and_preserves_outputs(self):
        def optimizer(function,initial,args,jac,method,options):
            ReproductionTests.mocked_residual_calls+=1
            value,gradient=function(initial,*args)
            return SimpleNamespace(x=initial,fun=value,jac=gradient,success=True,status=0,
                message='Explicit zero synthetic mock without optimization',nit=0,nfev=1,njev=1)
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'fresh'
            with patch('train.fit_question_operations',return_value=self.anchor_fit) as Sfit,patch('train.minimize',side_effect=optimizer):
                model,_,_,_=fit_local(self.bank,self.arrays,self.labels,self.source,output)
            self.assertEqual(Sfit.call_count,1);self.assertEqual(Sfit.call_args.args[3],'numeric_surface')
            self.assertEqual(Sfit.call_args.args[-1],output/'S')
            self.assertTrue((output/'local_arithmetic/training_design.npz').exists())
            np.testing.assert_array_equal(candidate_scores(self.bank,self.arrays,model),
                surface_scores(self.bank,self.arrays,self.anchor['raw_w'],'numeric_surface'))
            with self.assertRaises(FileExistsError):fit_local(self.bank,self.arrays,self.labels,self.source,output)
            with self.assertRaises(ValueError):fit_residual(self.bank,self.arrays,self.labels,VARIANT,self.source+1,prepared=self.prepared)

    def test_checkpoint_permutations_new_IDs_empty_and_invalid_inputs(self):
        model={'anchor_raw':self.anchor['raw_w'],'raw_v':np.linspace(-.02,.02,22),'variant':VARIANT}
        selected=choose_scores(candidate_scores(self.bank,self.arrays,model),self.bank,self.arrays)
        expected=[a[i] for a,i in zip(candidate_answers(self.bank),selected)]
        with tempfile.TemporaryDirectory() as temporary:
            export_checkpoint(temporary,model)
            for order in permutations(range(4)):
                bank=copy.deepcopy(self.bank)
                for row in bank:
                    row['id']='new-unseen-id';row['candidates']=[row['candidates'][i] for i in order]
                    for i,candidate in enumerate(row['candidates']):candidate['index']=i
                arrays={'embeddings':self.arrays['embeddings'][:,order]}
                chosen=predict(bank,arrays,temporary)
                self.assertEqual([a[i] for a,i in zip(candidate_answers(bank),chosen)],expected)
            self.assertEqual(predict([],{'embeddings':np.empty((0,4,1024))},temporary),[])
            invalid=copy.deepcopy(self.bank[:1])
            for candidate in invalid[0]['candidates']:candidate['text']='invalid'
            np.testing.assert_array_equal(candidate_scores(invalid,{'embeddings':self.arrays['embeddings'][:1]},model),[[0.]*4])


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/checks/prefit_checks.json',
                        help='Write this run receipt outside the archived results directory.')
    args = parser.parse_args()
    with (patch('sklearn.linear_model.LogisticRegression.fit',side_effect=AssertionError('No classifier fits')),
          patch('train.minimize',side_effect=AssertionError('No optimizer calls')),
          patch('train.load_training_rows',side_effect=AssertionError('No supplied data')),
          patch('train_surface.load_training_rows',side_effect=AssertionError('No supplied data')),
          patch('common.load_inputs',side_effect=AssertionError('No input loads'))):
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReproductionTests))
    if result.wasSuccessful():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'passed':True,'tests':result.testsRun,
            'explicit_mock_optimizer_calls':ReproductionTests.mocked_residual_calls,'actual_model_fits':0,
            'supplied_inputs_loaded':False,'reference_rows_decoded':0,
            'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src').glob('*.py'))},
            'test_sha256':sha(Path(__file__))},indent=2)+'\n',encoding='utf-8',newline='\n')
    raise SystemExit(not result.wasSuccessful())
