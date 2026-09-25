# MIRROR: From Imitation to Internalization in LLM Personalization

Training, evaluation, and dataset preparation code for LLM personalization.

## Layout

- train/: ContextSFT, PerCE, NextQuill, MIRROR, and MIRROR-F training programs.
- eval/: Base, RAG, last-K, TRSE, G-Eval, Expert, and OOD evaluation programs.
- data/: Dataset download and preparation programs.
- pyproject.toml and uv.lock: Python and uv environment definitions.

The MIRROR entry point is train/train_mirror.py. The MIRROR-F entry point is train/train_mirrorf.py.

Model weights, checkpoints, raw datasets, generated results, logs, credentials, and third-party repositories are intentionally excluded.

## Run code

### 1. Install the environment

Python 3.10 or 3.11 is required. Install uv, then create the locked environment:

```bash
uv sync
```


### 2. Prepare local directories

The programs expect the following directories under the repository root:

```text
models/
TrainModels/
TestResults/
eval/data/
logs/
figs/
```

Model checkpoints are not included in this repository. Place base models under `models/` and trained models under `TrainModels/`.

### 3. Download and prepare data

Run dataset preparation programs from the `data/` directory:

```bash
cd data
python download_one.py
python prepare_amazon.py
python fetch_general_evals.py
python prepare_summary.py
cd ..
```

Each program reads its input and output locations from the environment or its source-level defaults. Dataset files should remain local and must not be committed to the repository.

### 4. Train models

Run formal training programs from the `train/` directory without additional command-line arguments:

```bash
cd train
python train_contextsft.py
python train_PerCE.py
python train_NextQuill.py
python train_mirror.py
python train_mirrorf.py
cd ..
```

The MIRROR and MIRROR-F programs are the public names for the two on-policy training implementations. Configure the base model, teacher model, task set, GPU selection, output name, and training settings through the environment variables defined in each program.

For example:

```bash
export TRAIN_TASKS=longlamp
export CUDA_VISIBLE_DEVICES=0
python train/train_mirror.py
```

### 5. Run standard evaluation

Run evaluation programs from the `eval/` directory:

```bash
cd eval
python eval_lamp_base.py
python eval_lamp_sparse.py
python eval_ood.py
python eval_quality.py
cd ..
```

Run retrieval-based evaluations from `eval/rag-methods/`:

```bash
cd eval/rag-methods
python eval_RAG.py
python eval_lastK.py
python eval_TRSE.py
cd ../..
```

Run Expert and general-capability evaluations from `eval/quality/`:

```bash
cd eval/quality
python eval_expert.py
python eval_disaster.py
cd ../..
```

The OOD entry points are located in `eval/OODeval/`:

```bash
cd eval/OODeval
python longlampOOD.py
python amazonOOD.py
cd ../..
```

### 6. Configure evaluation models and tasks

Common evaluation settings are controlled through environment variables:

```bash
export EVAL_MODEL_NAMES=Qwen3-1.7B
export EVAL_TASKS=longlamp
export EVAL_GPU_IDS=0
```

Use `EVAL_TASKS=amazon` for the three Amazon tasks and `EVAL_TASKS=lamp` for the LaMP tasks supported by the evaluation code. Evaluation outputs are written to `TestResults/`, which is ignored by Git.

### 7. Output locations

- Trained models: `TrainModels/`
- Evaluation summaries and details: `TestResults/`
- Training logs: `logs/`
- Training figures: `figs/`

These output directories are intentionally excluded from version control. The repository contains source code and environment definitions only.
