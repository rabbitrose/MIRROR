# MIRROR: From Imitation to Internalization in LLM Personalization

Training and evaluation code for LLM personalization.

Model weights, checkpoints, raw datasets, generated results, logs, credentials, and third-party repositories are excluded.

## Project tree

```text
MIRROR/
├── train/
│   ├── train_sft.py
│   ├── train_contextsft.py
│   ├── train_PerCE.py
│   ├── train_NextQuill.py
│   ├── train_mirror.py
│   └── train_mirrorf.py
├── eval/
│   ├── eval_common.py
│   ├── eval_lamp_base.py
│   ├── eval_quality.py
│   ├── eval_ood.py
│   ├── rag-methods/
│   │   ├── eval_RAG.py
│   │   ├── eval_lastK.py
│   │   └── eval_TRSE.py
│   ├── quality/
│   │   ├── eval_expert.py
│   │   └── eval_disaster.py
│   └── OODeval/
│       ├── longlampOOD.py
│       └── amazonOOD.py
├── data/
│   ├── download_one.py
│   ├── fetch_general_evals.py
│   ├── prepare_amazon.py
│   └── prepare_summary.py
├── pyproject.toml
├── uv.lock
└── .gitignore
```

## Train

Install the locked environment from the repository root:

```bash
uv sync
```

Place base models in `models/` and trained models in `TrainModels/`. Prepare the required datasets with the programs in `data/`.

Run formal training programs without additional command-line arguments:

```bash
python train/train_contextsft.py
python train/train_PerCE.py
python train/train_NextQuill.py
python train/train_mirror.py
python train/train_mirrorf.py
```

Use environment variables to configure the task set, model paths, GPU selection, output name, and training settings. For example:

```bash
export TRAIN_TASKS=longlamp
export CUDA_VISIBLE_DEVICES=0
python train/train_mirror.py
```

The public training entry points are `train_mirror.py` for MIRROR and `train_mirrorf.py` for MIRROR-F.

## Eval

Place evaluation datasets under `eval/data/` and model checkpoints under `models/` or `TrainModels/`. Configure the model, task set, and GPU with environment variables:

```bash
export EVAL_MODEL_NAMES=Qwen3-1.7B
export EVAL_TASKS=longlamp
export EVAL_GPU_IDS=0
```

Run the full-context evaluation:

```bash
python eval/eval_lamp_base.py
```

Run quality, OOD, retrieval, and general-capability evaluations:

```bash
python eval/eval_quality.py
python eval/eval_ood.py
python eval/rag-methods/eval_RAG.py
python eval/rag-methods/eval_lastK.py
python eval/rag-methods/eval_TRSE.py
python eval/quality/eval_expert.py
python eval/quality/eval_disaster.py
python eval/OODeval/longlampOOD.py
python eval/OODeval/amazonOOD.py
```

Evaluation summaries and details are written to `TestResults/`. Training logs are written to `logs/`, and training figures are written to `figs/`. These output directories are ignored by Git.
