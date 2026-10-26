# DASE7506 MP2: Selecting Reliable LLM Reasoning

**Wu Houlin | Student ID: 3036863317**

The final method is the local-arithmetic selector **L**: a numeric-answer-group surface model followed by a small arithmetic-feature residual. Its official test accuracy is **84.79657387580299% (396/467)**. The method and checkpoint were selected before this test score was read. No test-based tuning or model switching followed.

The [seven-page report](report/MP2_Report.pdf) explains the method, controls, failed extensions, reliability findings, costs and limitations. [Markdown source](report/report.md) and [PDF build script](report/build_report.py) are included. The course leaderboard issue is created separately through the MP2 website; publishing this repository alone does not submit the assignment.

## Files

```text
src/          NumPy inference, fixed training/CV and core analysis entry points
checkpoint/   Final L weights and configuration (9,157 bytes)
report/       Seven-page report, its source and build script
results/      Aggregate scores, split protocol, hashes and compact audit records
              baseline_s/ preserves historical S results and checkpoint
licenses/     Supplied dataset license and attribution
```

Research worktrees, raw training outputs, cached test selections and environments are excluded. The historical S test score, 386/467, belongs to the earlier baseline and remains under `results/baseline_s/`; the matching original release is preserved.

## Install and obtain the supplied data

Use Python 3.12. Run these PowerShell commands from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
git clone https://github.com/xudongwu-0/DASE7506-MP2.git upstream
git -C upstream checkout bb801f4f2e7629d270df765ab12c333b2dc616bb
```

On Linux or macOS, use `.venv/bin/python`. Data are in `upstream/code/data` and are not duplicated here. An existing unmodified copy can be supplied with `--data-dir`. The official loader checks dataset hashes.

## Evaluate the frozen checkpoint without retraining

```powershell
.venv\Scripts\python.exe -B src/evaluate.py --data-dir upstream/code/data --split test --output runs/test
```

This runs the official predictor and independent scorer in separate fresh processes, then records the exact score, commands, hashes and external wall time. It includes startup, imports, asset/input loading, preprocessing and scoring. The frozen result is in [results/test_score.json](results/test_score.json); measured local resources are in [results/test_measurement.json](results/test_measurement.json).

Validation reproduction:

```powershell
.venv\Scripts\python.exe -B src/evaluate.py --data-dir upstream/code/data --split validation --output runs/validation
```

For an instructor audit bank without references:

```powershell
.venv\Scripts\python.exe -B src/predict.py --method src/submission.py --checkpoint checkpoint --split audit --data-dir PATH_TO_AUDIT_INPUTS --output runs/audit-predictions.json
```

Use a fresh output directory each time. Inference accepts the caller's bank and embeddings, uses no answer lookup and makes no LLM/API calls. Numeric libraries are limited to four threads and GPU visibility is disabled. The active checkpoint is 9,157 bytes; counting the unused historical S checkpoint too gives 18,246 bytes, below 16 MiB. Complete final test prediction plus scoring took **1.383 seconds on an Intel Core i5-14600KF**. The reference Xeon Platinum 8457C was not measured, so local timing does not certify reference-machine eligibility.

## Reproduce training and the main experiments

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.venv\Scripts\python.exe -B tests/test_training.py
.venv\Scripts\python.exe -B src/train.py --data-dir upstream/code/data --output runs/retrained
.venv\Scripts\python.exe -B src/cross_validate.py --data-dir upstream/code/data --output runs/cv
.venv\Scripts\python.exe -B src/reproduce_controls.py --data-dir upstream/code/data --output runs/controls
.venv\Scripts\python.exe -B src/analyze_reliability.py --data-dir upstream/code/data --oof-dir runs/cv --output runs/reliability
```

Full training uses only the 4,755 supplied training questions. S is a no-intercept logistic preference model on 1,034 group/numeric-surface features, with within-answer subset augmentation and equal total weight per informative question. Scalers fit original full groups. L freezes S and learns a 22-coordinate mean/max residual from three format and eight bounded numeric-check features, using original full-group pairs.

The residual minimizes weighted mean logistic loss plus `0.005 * ||v||^2`, starts at zero and uses L-BFGS-B with `maxiter=2000`, `ftol=1e-12`, `gtol=1e-6` and `maxls=40`. Dependencies are pinned: NumPy 2.3.5, SciPy 1.16.3, scikit-learn 1.7.2 and threadpoolctl 3.6.0. Split/CV seeds are 20261021/20261022; the split protocol is saved in `results/protocol.json`.

The retrained model is in `runs/retrained/checkpoint`. Pair audits, scalers, initial/final coefficients, objectives, gradients and fitting provenance are retained in that run. A clean full training run reproduced both final checkpoint files byte for byte. All S/L five-fold scores and 50 associated fold-array artifacts matched the original records exactly. The C1028/S/format-only utilities also reproduced every archived OOF score exactly; the reliability utility reproduced its audited statistics.

`results/research_registry.json` preserves compact historical recipes/results and their source hashes. Other exploratory extensions are documented there and in the report; their large local worktrees are not copied into this repository. These utilities reproduce fixed prior methods and do not conduct another model search.

## Recorded results

| Evaluation | L correct | Population |
|---|---:|---|
| Ordinary five-fold development | 1,410/1,691 | Conflicting numeric answers |
| Lexical-grouped five-fold development | 1,405/1,691 | Conflicting numeric answers |
| Previously queried training holdout | 347/423 | Conflicting numeric answers |
| Official validation | 751/798 | Complete validation bank |
| Official test | **396/467** | Complete test bank |

The holdout gain over S was one question. Repeated development/validation comparisons and historical aggregate test exposure limit independent-generalization claims. Score-gap subset accuracy is descriptive and cannot replace complete-bank evaluation. The task grades the selected final number, not the validity of every reasoning step.

## Rebuild the report

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-report.txt
.venv\Scripts\python.exe report/build_report.py
```

Report dependencies are optional and are not imported by inference. The supplied PDF has seven pages, below the ten-page limit.

## Attribution and AI disclosure

The bank, numeric parser, predictor and scorer come from the [official course repository](https://github.com/xudongwu-0/DASE7506-MP2). Questions/references derive from GSM8K; candidate solutions and supplied representations were prepared with Qwen3-0.6B. Notices remain in `licenses/`. No external training datasets or additional pretrained weights are used.

OpenAI Codex provided substantive assistance with planning, implementation, experiment execution, audits, packaging and report drafting. Local Scientific Agent Skills workflows supported the checks and are acknowledged in the report. The student remains responsible for understanding and reviewing the work. The final freeze and exact asset hashes are recorded in `results/final_selection.json` and the release manifest.
