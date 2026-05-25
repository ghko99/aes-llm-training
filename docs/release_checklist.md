# Release Checklist

Use this checklist before promoting an AES LLM checkpoint or adapter as a reusable artifact.

## Required Metadata

- Base model name and revision.
- Adapter or checkpoint path.
- Dataset split version.
- Training command and Git commit.
- Loss configuration and LoRA settings.
- Evaluation script and metric output.

## Validation

- Confirm all rubric scores are parsed for the held-out split.
- Review parse failure counts.
- Compare overall and per-rubric metrics against the previous candidate.
- Inspect a small set of high-error predictions.

## Packaging

Keep tokenizer files, adapter config, adapter weights, and model card notes together. Do not publish raw essays or non-synthetic prediction examples.
