# AFS 文件索引：GitHub `MIRROR` ↔ `afs_laihuayi02`

`afs_laihuayi02` 是扁平存储，不支持目录树。本文件是 GitHub 远程仓库文件树与 `afs_laihuayi02` 中散装文件名的一一对应索引。

## 基本信息

- GitHub 远程：`git@github.com:rabbitrose/MIRROR.git`（private，分支 `main`）
- 本次同步对应 commit：`be9e809`
- AFS 落地目录：`/root/paddlejob/workspace/afs_laihuayi02`
- 本地工作副本：`/root/paddlejob/workspace/env_run/MIRROR`
- 已同步文件数：50

## 命名规则

GitHub 中的路径 `a/b/c.py` → AFS 中的文件名 `MIRROR__a__b__c.py`。

- 前缀 `MIRROR__` 标识来源仓库（与同机 `afs_laihuayi` 中 `RoleACL__...` 的约定一致）
- 路径分隔符 `/` 一律替换为双下划线 `__`
- 文件基名保持原样（含空格与中文），以保证与 GitHub 路径严格一一对应
- 反向还原：去掉 `MIRROR__` 前缀，再把 `__` 换回 `/`

## 不进 GitHub 的内容

- **模型权重**：`models/`、`TrainModels/`、`checkpoints/`、`*.safetensors`、`*.bin`、`*.pt`、`*.pth` 只存放在 `afs_laihuayi02`，已由 `.gitignore` 拦截
- **uv 虚拟环境**：`.venv/` 不上传；环境版本信息通过 `pyproject.toml` 与 `uv.lock` 入库，用 `uv sync` 即可复现
- **训练/评测产物**：`TestResults/`、`logs/`、`*.json`、`*.jsonl`、`*.parquet`、`*.log` 同样只存 AFS

新增这类文件到 AFS 后，请按上面的命名规则补进下面的索引表。

## 索引表

### `(repo root)`

| GitHub 路径 | afs_laihuayi02 文件名 | 大小 |
| --- | --- | --- |
| `.gitignore` | `MIRROR__.gitignore` | 167 B |
| `README.md` | `MIRROR__README.md` | 3.1 KB |
| `pyproject.toml` | `MIRROR__pyproject.toml` | 382 B |
| `uv.lock` | `MIRROR__uv.lock` | 615.3 KB |

### `ablation`

| GitHub 路径 | afs_laihuayi02 文件名 | 大小 |
| --- | --- | --- |
| `ablation/train_mirror.py` | `MIRROR__ablation__train_mirror.py` | 26.2 KB |

### `data`

| GitHub 路径 | afs_laihuayi02 文件名 | 大小 |
| --- | --- | --- |
| `data/download_one.py` | `MIRROR__data__download_one.py` | 9.1 KB |
| `data/fetch_general_evals.py` | `MIRROR__data__fetch_general_evals.py` | 2.1 KB |
| `data/prepare_amazon.py` | `MIRROR__data__prepare_amazon.py` | 10.3 KB |
| `data/prepare_summary.py` | `MIRROR__data__prepare_summary.py` | 11.0 KB |

### `eval/OODeval`

| GitHub 路径 | afs_laihuayi02 文件名 | 大小 |
| --- | --- | --- |
| `eval/OODeval/amazonOOD.py` | `MIRROR__eval__OODeval__amazonOOD.py` | 8.8 KB |

### `eval`

| GitHub 路径 | afs_laihuayi02 文件名 | 大小 |
| --- | --- | --- |
| `eval/eval_common.py` | `MIRROR__eval__eval_common.py` | 17.1 KB |
| `eval/eval_lamp_base.py` | `MIRROR__eval__eval_lamp_base.py` | 9.9 KB |
| `eval/eval_ood.py` | `MIRROR__eval__eval_ood.py` | 2.8 KB |
| `eval/eval_quality.py` | `MIRROR__eval__eval_quality.py` | 21.4 KB |

### `eval/quality`

| GitHub 路径 | afs_laihuayi02 文件名 | 大小 |
| --- | --- | --- |
| `eval/quality/eval_disaster.py` | `MIRROR__eval__quality__eval_disaster.py` | 41.3 KB |
| `eval/quality/eval_expert.py` | `MIRROR__eval__quality__eval_expert.py` | 34.9 KB |

### `eval/rag-methods`

| GitHub 路径 | afs_laihuayi02 文件名 | 大小 |
| --- | --- | --- |
| `eval/rag-methods/eval_RAG.py` | `MIRROR__eval__rag-methods__eval_RAG.py` | 12.1 KB |
| `eval/rag-methods/eval_TRSE.py` | `MIRROR__eval__rag-methods__eval_TRSE.py` | 16.0 KB |
| `eval/rag-methods/eval_lastK.py` | `MIRROR__eval__rag-methods__eval_lastK.py` | 12.3 KB |

### `paper`

| GitHub 路径 | afs_laihuayi02 文件名 | 大小 |
| --- | --- | --- |
| `paper/19994 我的 ICLR 提交论文.pdf` | `MIRROR__paper__19994 我的 ICLR 提交论文.pdf` | 1.2 MB |
| `paper/ICLR 审稿意见.md` | `MIRROR__paper__ICLR 审稿意见.md` | 10.1 KB |
| `paper/ICLRRebuttal.tex` | `MIRROR__paper__ICLRRebuttal.tex` | 127.9 KB |
| `paper/ICLRsubmission.tex` | `MIRROR__paper__ICLRsubmission.tex` | 127.9 KB |
| `paper/arxiv.tex` | `MIRROR__paper__arxiv.tex` | 127.9 KB |
| `paper/fancyhdr.sty` | `MIRROR__paper__fancyhdr.sty` | 20.0 KB |
| `paper/iclr.tex` | `MIRROR__paper__iclr.tex` | 127.9 KB |
| `paper/iclr2027_conference.bst` | `MIRROR__paper__iclr2027_conference.bst` | 26.3 KB |
| `paper/iclr2027_conference.sty` | `MIRROR__paper__iclr2027_conference.sty` | 8.9 KB |
| `paper/math_commands.tex` | `MIRROR__paper__math_commands.tex` | 12.0 KB |
| `paper/natbib.sty` | `MIRROR__paper__natbib.sty` | 44.1 KB |
| `paper/reference.bib` | `MIRROR__paper__reference.bib` | 20.0 KB |

### `paper/figs`

| GitHub 路径 | afs_laihuayi02 文件名 | 大小 |
| --- | --- | --- |
| `paper/figs/1motivation.pdf` | `MIRROR__paper__figs__1motivation.pdf` | 180.0 KB |
| `paper/figs/2methods.pdf` | `MIRROR__paper__figs__2methods.pdf` | 432.2 KB |
| `paper/figs/3ood.pdf` | `MIRROR__paper__figs__3ood.pdf` | 35.1 KB |
| `paper/figs/4Geval.pdf` | `MIRROR__paper__figs__4Geval.pdf` | 33.3 KB |
| `paper/figs/5ablationLong.pdf` | `MIRROR__paper__figs__5ablationLong.pdf` | 26.0 KB |
| `paper/figs/5ablationShort.pdf` | `MIRROR__paper__figs__5ablationShort.pdf` | 25.8 KB |

### `paper/绘图`

| GitHub 路径 | afs_laihuayi02 文件名 | 大小 |
| --- | --- | --- |
| `paper/绘图/【old】图2：MIRROR-F.pdf` | `MIRROR__paper__绘图__【old】图2：MIRROR-F.pdf` | 439.8 KB |
| `paper/绘图/【old】图2：MIRROR-F.pptx` | `MIRROR__paper__绘图__【old】图2：MIRROR-F.pptx` | 105.3 KB |
| `paper/绘图/【投稿】图1：方法对比.pdf` | `MIRROR__paper__绘图__【投稿】图1：方法对比.pdf` | 180.0 KB |
| `paper/绘图/【投稿】图1：方法对比.pptx` | `MIRROR__paper__绘图__【投稿】图1：方法对比.pptx` | 123.6 KB |
| `paper/绘图/【投稿】绘图.pptx` | `MIRROR__paper__绘图__【投稿】绘图.pptx` | 332.7 KB |
| `paper/绘图/图2methods.pdf` | `MIRROR__paper__绘图__图2methods.pdf` | 432.2 KB |
| `paper/绘图/图2methods.pptx` | `MIRROR__paper__绘图__图2methods.pptx` | 98.4 KB |

### `train`

| GitHub 路径 | afs_laihuayi02 文件名 | 大小 |
| --- | --- | --- |
| `train/train_NextQuill.py` | `MIRROR__train__train_NextQuill.py` | 16.0 KB |
| `train/train_PerCE.py` | `MIRROR__train__train_PerCE.py` | 13.5 KB |
| `train/train_contextsft.py` | `MIRROR__train__train_contextsft.py` | 1.3 KB |
| `train/train_mirror.py` | `MIRROR__train__train_mirror.py` | 25.6 KB |
| `train/train_mirrorf.py` | `MIRROR__train__train_mirrorf.py` | 34.8 KB |
| `train/train_sft.py` | `MIRROR__train__train_sft.py` | 11.3 KB |

