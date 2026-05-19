# Evaluation Notes

Use a consistent evaluation pass when reporting AES model quality.

## Metrics

Track at least the following metrics for the overall score and each rubric:

- Quadratic weighted kappa.
- Mean absolute error.
- Root mean squared error.
- Exact score accuracy when useful for diagnostics.
- Pearson correlation for trend agreement.

## Split Discipline

Evaluate CE, NTL, WNTL, SAL, and mixed-loss runs on the same held-out split. When reporting `test_14_1`, `test_14_2`, or `test_14_3`, include the split name in the artifact filename.

## Reporting

Summaries should include the base model, loss configuration, max sequence length, LoRA settings, and checkpoint selection rule. Keep raw prediction exports so per-rubric error patterns can be checked later.
