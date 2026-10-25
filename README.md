# DASE7506 MP2: Selecting Reliable LLM Reasoning

This local candidate contains the qualified **local-arithmetic selector L**, its frozen checkpoint and standalone reproduction code. Its recorded official validation accuracy is **94.11027568922306% (751/798)**. Its final test score is pending. This branch has not been pushed, and the course report and leaderboard issue remain pending.

L first scores canonical numeric-answer groups with the surface selector S. A linear residual then uses bounded arithmetic checks and answer-format features from the candidate text. Equivalent parsed answers share a group; inference scores the original complete groups. The checkpoint contains a 1,034-element S anchor and a 22-element residual. It needs NumPy and the supplied embeddings, with no model download or LLM call.

```text
src/          Inference, standalone training, fixed cross-validation and evaluation
checkpoint/   Frozen L weights and configuration, 9,157 bytes
results/      Candidate recipe, source provenance and frozen split protocol
              baseline_s/ contains historical S records and its checkpoint
tests/        Synthetic checks without supplied data or actual fitting
licenses/     Supplied data license and attribution
```

The historical S test result, 386/467, belongs to S and is stored under `results/baseline_s/`. It is not an L score. Experimental worktrees and rejected models are excluded from this package.

## Installation and data

Use Python 3.12 from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
git clone https://github.com/xudongwu-0/DASE7506-MP2.git upstream
git -C upstream checkout bb801f4f2e7629d270df765ab12c333b2dc616bb
```

On Linux or macOS, replace `.venv\Scripts\python.exe` with `.venv/bin/python`. The data are in `upstream/code/data`; they are not duplicated here. You can also pass an existing unmodified copy through `--data-dir`. The loader checks the official input hashes.

## Evaluate the frozen checkpoint

```powershell
.venv\Scripts\python.exe -B src/evaluate.py --data-dir upstream/code/data --split validation --output runs/validation
```

The wrapper runs the official prediction and independent scoring tools in separate fresh processes. It records commands, source and asset hashes, score and external elapsed time, including startup, imports, input loading, preprocessing and scoring. The existing public tools retain their `--data-dir` support; numeric parsing and scoring semantics are unchanged.

After the final candidate is frozen, the final test evaluation command is:

```powershell
.venv\Scripts\python.exe -B src/evaluate.py --data-dir upstream/code/data --split test --output runs/final-test
```

The test command has not been run for L. Use a fresh output directory for every evaluation. For an instructor audit bank without references, run prediction alone:

```powershell
.venv\Scripts\python.exe -B src/predict.py --method src/submission.py --checkpoint checkpoint --split audit --data-dir PATH_TO_AUDIT_INPUTS --output runs/audit-predictions.json
```

Numeric libraries are limited to four threads and GPU visibility is disabled. The active L checkpoint is 9,157 bytes; all packaged checkpoints total 18,246 bytes when the historical S copy is also counted. Both are below the 16 MiB asset limit. The course requires prediction plus scoring within 30 seconds on its reference Xeon Platinum 8457C. Local CPU timings do not establish reference-machine eligibility.

## Reproduce training

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.venv\Scripts\python.exe -B tests/test_training.py
.venv\Scripts\python.exe -B src/train.py --data-dir upstream/code/data --output runs/retrained
```

Training uses only the 4,755 supplied training questions. Stage S retains the original fixed numeric-surface trainer, including within-answer subset augmentation and equal total weight per informative question. The residual uses only original full-group correct-versus-incorrect pairs. Its old11 candidate features are pooled by group mean and maximum; the scaler includes every original group, including questions with one numeric answer. The S score is frozen as an offset.

The residual minimizes weighted mean logistic loss plus `0.005 * ||v||²`, starts from zero and uses L-BFGS-B with `maxiter=2000`, `ftol=1e-12`, `gtol=1e-6` and `maxls=40`. No intercept, validation fitting, parameter search or test fitting occurs. Dependency versions are pinned to NumPy 2.3.5, SciPy 1.16.3, scikit-learn 1.7.2 and threadpoolctl 3.6.0.

The final model is in `runs/retrained/checkpoint`. The output also retains the S and residual pair audits, scalers, coefficients, initial/final objectives and gradients, feature table and fitting provenance. A cold run of this standalone trainer reproduced the supplied checkpoint byte for byte. Its five-fold S/L scores also match every archived score exactly. An independent audit reconstructed the full-data pairs, scalers, objective and gradients. A clean Git archive in a fresh NumPy-only environment reproduced all 798 validation choices and the 751/798 score. Compact receipts, including the independent checks, are in `results/`.

To reproduce the fixed five-fold S/L comparison on the training development pool:

```powershell
.venv\Scripts\python.exe -B src/cross_validate.py --data-dir upstream/code/data --output runs/cross-validation
```

This command uses `results/protocol.json` and leaves the excluded training holdout and official validation/test splits unused. It evaluates both fixed models without selecting another recipe.

## Recorded results and limits

| Evaluation | L correct | Scope |
| --- | ---: | --- |
| Original five-fold development | 1,410/1,691 | Conflicting numeric answers only |
| Grouped five-fold development | 1,405/1,691 | Conflicting numeric answers only |
| Excluded training holdout | 347/423 | Conflicting numeric answers only |
| Official validation | 751/798 | All validation questions |
| Official test | Pending | Final L evaluation has not run |

Overall validation accuracy includes questions whose candidates agree, so it is not directly comparable with the hard-question scores. The holdout gain over S was one question. Repeated development and validation comparisons, and the known historical S test score, limit claims of independent generalization. The metric grades the selected final numeric answer; it does not certify every reasoning step.

## Attribution and disclosure

The bank, numeric parser, prediction runner and scorer come from the [official course repository](https://github.com/xudongwu-0/DASE7506-MP2). Questions and references derive from [GSM8K](https://github.com/openai/grade-school-math); course candidates and supplied representations were prepared with Qwen3-0.6B. Data notices remain in `licenses/`. No external training datasets or additional pretrained weights are included.

Codex provided substantive assistance with planning, implementation, experiment execution, audits, packaging and documentation. The student must understand and review the code. A report of at most ten pages and the course website's GitHub issue submission remain to be completed.
