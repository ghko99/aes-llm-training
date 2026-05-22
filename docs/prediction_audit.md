# Prediction Audit

Audit raw predictions before promoting a metric table.

## Required Files

- Raw model generations or logits export.
- Parsed score output.
- Gold labels for the evaluated split.
- Metric output file.
- Parse failure log, even when empty.

## Checks

- Confirm every example has all required rubric scores.
- Count out-of-range or missing scores.
- Compare parsed score distributions with gold distributions.
- Inspect a few high-error examples per rubric.
- Verify the evaluated split name appears in the output filename or metadata.

## Notes

Keep raw predictions outside Git unless they are tiny examples. Commit only the audit process or curated result summaries.
