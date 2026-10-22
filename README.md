# DASE7506 MP2: Selecting Reliable LLM Reasoning

This repository contains the current frozen numeric-answer-group selector, its training code, checkpoint, and evaluation records. The official test score is **82.65524625267666% (386/467)**. The course report is pending; this repository has not been submitted to the course leaderboard.

The selector groups candidates with equivalent parsed numeric answers, averages the supplied representations inside each group, and scores groups with a linear model plus six question/final-answer features. Training uses within-answer subset augmentation and equal total pair weight per informative question. Inference uses the original complete groups and one 1,034-element coefficient vector. It requires no model download or LLM call.

## Files

```text
src/          Inference, training, cross-validation, and official evaluation tools
checkpoint/   Frozen weights and configuration: 9,089 uncompressed bytes
results/      Scores, compact experiment records, split protocol, and hashes
licenses/     Supplied data license and attribution
```

Exploratory notebooks, failed-run checkpoints, environments, and the research workspace are kept outside this repository. `main` holds reviewed reproducible versions. Future experimental code stays local until it has passed the development checks.

## Installation and data

Use Python 3.12. Run these commands from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
git clone https://github.com/xudongwu-0/DASE7506-MP2.git upstream
git -C upstream checkout bb801f4f2e7629d270df765ab12c333b2dc616bb
```

On Linux or macOS, replace `.venv\Scripts\python.exe` with `.venv/bin/python`. The official data are in `upstream/code/data`; they are not duplicated in this repository. The official loader checks the input hashes. An existing unmodified copy of the course data can also be passed through `--data-dir`.

## Evaluate the checkpoint

Validation:

```powershell
.venv\Scripts\python.exe -B src/evaluate.py --data-dir upstream/code/data --split validation --output runs/validation
```

Final test scoring or peer reproduction:

```powershell
.venv\Scripts\python.exe -B src/evaluate.py --data-dir upstream/code/data --split test --output runs/test
```

Use a fresh output directory each time. The wrapper runs the unchanged official `predict.py` and independent `score_predictions.py` in separate fresh processes. It records complete commands, source/asset hashes, the official score, and external wall time including process startup, imports, loading, preprocessing, and scoring. Numeric libraries are limited to four threads and GPU visibility is disabled.

For an instructor audit bank, use `--split audit --data-dir PATH_TO_AUDIT_INPUTS`. When audit references are unavailable, run prediction alone:

```powershell
.venv\Scripts\python.exe -B src/predict.py --method src/submission.py --checkpoint checkpoint --split audit --data-dir PATH_TO_AUDIT_INPUTS --output runs/audit-predictions.json
```

## Reproduce training

Install training dependencies, then fit only the supplied training split:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.venv\Scripts\python.exe -B src/train.py --data-dir upstream/code/data --output runs/retrained
```

The checkpoint will be in `runs/retrained/checkpoint`. The fixed recipe uses standardization fitted on original training groups, logistic pair loss, mean-loss L2 coefficient 0.01, no intercept, `lbfgs`, `max_iter=2000`, and `tol=1e-6`. Augmented subsets never fit preprocessing. No validation/test targets enter fitting or stopping. The CPU fit is deterministic in the recorded environment; a full refit using this cleaned training entrypoint reproduced the frozen weights bit-for-bit.

Reproduce the original development five-fold score:

```powershell
.venv\Scripts\python.exe -B src/cross_validate.py --data-dir upstream/code/data --output runs/cv
```

The supplied protocol uses 3,804 training questions, including 1,691 with conflicting parsed answers. The remaining 951 training questions formed a coordinator holdout; they are excluded from development CV. All final checkpoint fitting uses the full 4,755-question training split. Each CV fold refits preprocessing and weights from scratch.

The matched ablation removes the six numeric-surface features while retaining the same embeddings, grouping, subsets, question weights, solver, and folds:

```powershell
.venv\Scripts\python.exe -B src/cross_validate.py --data-dir upstream/code/data --without-numeric-surface --output runs/cv-without-numeric
```

This control records 1,370/1,691 conflicting-answer questions correct, compared with 1,392 for the selected model. Both variants include the original group-support and variance features. This comparison does not establish that every numeric feature contributes independently.

## Recorded results and limits

| Evaluation | Correct | Accuracy |
|---|---:|---:|
| Development five-fold, conflicting answers | 1,392/1,691 | 82.3182% |
| Coordinator training holdout, conflicting answers | 346/423 | 81.7967% |
| Official validation | 746/798 | 93.4837% |
| Official test | 386/467 | 82.6552% |

Validation includes easier questions, so its overall accuracy should not be compared directly with the conflicting-answer-only scores. Published accuracy grades the final numeric answer, not every reasoning step.

The first frozen test evaluation took 0.5359309 seconds for fresh prediction plus independent scoring on an Intel Core i5-14600KF under Windows 11, with four numeric threads. Inference assets total 9,089 bytes, with uncompressed NPZ storage. Reference eligibility is determined on the course Xeon Platinum 8457C; local timing does not establish reference-machine eligibility.

Development involved repeated CV, training holdout, and validation comparisons. An earlier V1 test aggregate was already known. A question-number punctuation parsing repair was made during development, and the earlier buggy attempt was excluded from qualification. This snapshot was selected before its V2 test score was opened. Neither the V2 test score nor per-question test errors were used to tune it. Test references must not be used for later training, tuning, selection, or lookups. These historical exposures prevent claims of a pristine independent test or global optimality.

## Attribution and remaining submission work

The bank, numeric parser, prediction runner, and scorer come from the [official course repository](https://github.com/xudongwu-0/DASE7506-MP2). Original questions and references derive from [GSM8K](https://github.com/openai/grade-school-math); the course candidates and supplied representations were prepared with Qwen3-0.6B. Data notices are retained in `licenses/`. No pretrained model weights or external training datasets are included.

Codex provided substantive assistance with research planning, implementation, training execution, audits, packaging, and documentation. The student must understand and review the implementation. A report of at most ten pages remains to be added after the experimental work is settled. The course website's GitHub issue submission is a separate step and has not been performed.
