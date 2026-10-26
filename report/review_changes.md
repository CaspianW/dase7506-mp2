# Review and revisions, 27 October 2026

This review covers the published report, README, inference path, training objectives, evaluation wrapper and reproduction scripts at `c197a77518ef65c052b1497dd474648a3ad85438`. It was performed with OpenAI Codex assistance. It does not constitute an independent human review.

The review found no discrepancy in the reported final score, core development counts or training objective. It found two supporting-script problems and several places where the documentation could mislead a reader. The selected model, feature extraction, official parser/scorer and checkpoint are unchanged.

## Code and reproduction instructions

| Finding | Revision |
|---|---|
| `evaluate.py` captured a failed child process's output without displaying the actual diagnostic. A missing input or dependency could appear only as a generic subprocess failure. | Display the failed stage, exit code, stdout and stderr, and stop with the child's exit code. The successful prediction and scoring commands are unchanged. |
| Running `tests/test_training.py` overwrote the archived `results/prefit_checks.json`. | Write new receipts under `runs/checks/` by default; support an explicit `--output` path. Preserve the historical record. |
| The report trained a checkpoint and then evaluated the default distributed checkpoint. It did not explicitly evaluate the newly trained weights. | Add `--checkpoint runs/retrained/checkpoint` and explain the default in both the report and README. |
| The PDF builder could insert line breaks into long shell commands without a continuation character. | Preserve source line breaks; use explicit PowerShell continuation in the report. |
| Historical JSON records still contain stage-specific fields such as `report_generated: false`. | Explain their historical scope in the README. Keep their original values and use `SUBMISSION.json` for current artifact links. |

## Report wording and interpretation

The revisions replace experiment-management language with descriptions of what was tested. For example, “admission decision” became “checking the result with a different split”; “declared temporal primary” became “the planned temporal-order method.” The repeated terms “gates,” “promotion” and “numerical locality” were replaced where they obscured the experiment.

Several changes improve precision:

- “Independently reconstructed” now identifies a separate checking script. The report explicitly says that the checks were automated and developed with Codex assistance.
- The hypothesis and format-only comparison are stated directly. The arithmetic block includes final-answer compatibility and more coefficients, so its gain cannot be attributed solely to detecting wrong intermediate calculations.
- The relative answer-magnitude feature uses the largest retained **absolute** question number, with a denominator of one when no nonzero number is retained.
- Selective coverage can exceed its target because the requested count is rounded up and cutoff ties are retained. The original explanation mentioned ties alone.
- Pooled OOF margins come from separately fitted models, so their scales need not be equally calibrated. The reported AUROC and subset accuracies remain descriptive.
- The package reproduces the selected method, core controls and reliability analysis. The report now explicitly states that the other extensions cannot all be rerun from the public package alone.
- Prior validation reuse and aggregate test exposure remain disclosed. The 170/170 high-gap subset is not presented as a guarantee of perfect performance.

The abstract and discussion were edited for direct, factual language. No AI-detection score was used, and these edits do not conceal AI assistance. The disclosure now includes the review and language editing.

## Verification

Six existing synthetic training/inference tests and two new subprocess tests pass. The checks cover objective gradients, sample weighting, checkpoint export, candidate permutation, empty input and subprocess diagnostics. The test run leaves the historical receipt untouched.

A fresh run of the unchanged training procedure on all 4,755 supplied training questions reproduces both checkpoint files byte for byte. This review adds two verification fits, with no hyperparameter search. The retrained checkpoint scores 751/798 on validation. The distributed checkpoint still scores 396/467 on the complete official test bank, with every selected index matching the original frozen evaluation. No test-error analysis or model selection was performed.

All frozen model, feature and training source files retain their original hashes. Among files listed in the pre-test source freeze, only the evaluation wrapper changed. Aggregate measurements and hashes for this review are in `results/review.json`.

The regenerated report remains seven pages. All pages were rendered for visual inspection, and text bounds were checked. The references were checked against the course guide, arXiv and PMLR pages. Local CPU timing remains distinct from measurement on the official reference CPU.

The reviewed version is released as `mp2-final-v2.1`; the previous release remains available. The course submission issue has not been created by this review.
