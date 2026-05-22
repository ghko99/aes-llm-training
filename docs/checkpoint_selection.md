# Checkpoint Selection

Use the same selection rule when comparing CE, NTL, WNTL, SAL, and mixed-loss runs.

## Selection Rule

Choose one primary metric before training starts. Common choices are overall QWK, average per-rubric QWK, or validation loss. Avoid selecting a checkpoint after inspecting the test split.

## Metadata To Save

- Selected checkpoint path.
- Selection metric and value.
- Validation split name.
- Epoch and global step.
- Loss configuration.
- Whether Unsloth or multi-GPU mode was used.

## Reporting

When a table reports test performance, include the checkpoint selection rule in the surrounding notes. This keeps comparison fair across different loss configurations.
