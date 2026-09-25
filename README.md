# MIRROR

Training, evaluation, and dataset preparation code for LLM personalization.

## Layout

- train/: ContextSFT, PerCE, NextQuill, MIRROR, and MIRROR-F training programs.
- eval/: Base, RAG, last-K, TRSE, G-Eval, Expert, and OOD evaluation programs.
- data/: Dataset download and preparation programs.
- pyproject.toml and uv.lock: Python and uv environment definitions.

The MIRROR entry point is train/train_mirror.py. The MIRROR-F entry point is train/train_mirrorf.py.

Model weights, checkpoints, raw datasets, generated results, logs, credentials, and third-party repositories are intentionally excluded.
