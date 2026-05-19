# Run Tracking

This project supports multiple training paths, so each experiment should carry enough metadata to reproduce the launch conditions.

## Minimum Run Record

- Git commit and branch.
- `MODEL_PATH` value and model revision.
- Training entrypoint: `train.sh`, `train_multi_gpu.sh`, or direct `python train.py`.
- GPU count, CUDA version, and whether Unsloth was enabled.
- Dataset split revision from `aes_datasets/`.
- Loss flags such as `--no_ntl`, `--use_sal`, and `--no_weighted_ntl`.

## Output Bundle

Keep these artifacts together after each run:

- Training log and final command line.
- Best checkpoint path.
- Evaluation JSON or table export.
- W&B run URL when logging is enabled.
- Notes about any interrupted or resumed job.

This makes CE, NTL, WNTL, and SAL variants easier to compare without relying on memory or shell history.
