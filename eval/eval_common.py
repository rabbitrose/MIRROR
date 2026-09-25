"""Shared data, inference, metric, and result utilities for MIRROR evaluation."""
import json
import os
import time
import traceback
from datetime import datetime
from multiprocessing import Pool

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'models'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'longlamp-sft')
RESULTS_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'TestResults'))
TRAINED_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'TrainModels'))
TARGET_MODEL_NAMES = {'Llama-3.2-3B-SFT'}
                                                          
if os.environ.get('EVAL_MODEL_NAMES'):
    TARGET_MODEL_NAMES = {n.strip() for n in os.environ['EVAL_MODEL_NAMES'].split(',')
                          if n.strip()}

                                     
                                                         
                                                  
DATA_PATH = os.environ.get(
    'LONGLAMP_DATA_DIR', os.path.join(BASE_DIR, 'data', 'longlamp-sft'))
                                    
                                                        
                                                           
                                                
LONGLAMP_DATA_OVERRIDDEN = bool(os.environ.get('LONGLAMP_DATA_DIR'))
AMAZON_DATA_PATH = os.path.join(BASE_DIR, 'data', 'amazon-sft')
                                                  
                                                                    
LAMP_DATA_PATH = os.path.join(BASE_DIR, 'data', 'lamp-sft')

LONGLAMP_TASKS = {
    'abstract_generation': 'abstract_generation_user',
    'product_review': 'product_review_user',
    'topic_writing': 'topic_writing_user',
}
AMAZON_TASKS = {
    'book_review': 'book_review_user',
    'movie_review': 'movie_review_user',
    'cd_review': 'cd_review_user',
}
LAMP_TASKS = {
    'news_headline': 'news_headline_user',                
    'scholarly_title': 'scholarly_title_user',            
}
                                           
TASKS = dict(LONGLAMP_TASKS, **AMAZON_TASKS)
TASKS.update(LAMP_TASKS)
TASK_ROOT = {t: DATA_PATH for t in LONGLAMP_TASKS}
TASK_ROOT.update({t: AMAZON_DATA_PATH for t in AMAZON_TASKS})
TASK_ROOT.update({t: LAMP_DATA_PATH for t in LAMP_TASKS})
                                      
                             
DATASET_GROUPS = {
    'longlamp': list(LONGLAMP_TASKS),
    'amazon': list(AMAZON_TASKS),
    'lamp': list(LAMP_TASKS),
}
DATASET_SUFFIX = {'longlamp': '-long', 'amazon': '-amazon', 'lamp': '-lamp'}


def task_dir(task):
    """某任务 parquet 所在目录（自动区分 LongLaMP / Amazon 数据根）。"""
    return os.path.join(TASK_ROOT[task], TASKS[task])


def resolve_tasks(spec, default_group='longlamp'):
    """把任务选择字符串解析成任务名列表。

    spec 取值（逗号分隔可混合）：
      'all'       -> 全部 8 个任务
      'longlamp'  -> abstract_generation / product_review / topic_writing
      'amazon'    -> book_review / movie_review / cd_review
      'lamp'      -> news_headline / scholarly_title
      具体任务名  -> 只取该任务
    spec 为空时用 default_group。只返回数据目录存在的任务（缺数据的自动跳过）。
    """
    spec = (spec or default_group).strip()
    names = []
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if tok == 'all':
            names += list(TASKS)
        elif tok in DATASET_GROUPS:
            names += DATASET_GROUPS[tok]
        elif tok in TASKS:
            names.append(tok)
        else:
            raise RuntimeError('未知任务选择：%s' % tok)
    seen, out = set(), []
    for t in names:
        if t in seen:
            continue
        seen.add(t)
        if glob.glob(os.path.join(task_dir(t), '*.parquet')):
            out.append(t)
        else:
            print('跳过 %s：数据目录无 parquet（%s）' % (t, task_dir(t)), flush=True)
    return out


                                                
EVAL_TASKS = resolve_tasks(os.environ.get('EVAL_TASKS'), 'longlamp')
NUM_RETRIEVE = 2                                                        
PROFILE_CHAR_BUDGET = 1500                       
                                                            
                                                     
                                                                       
                                                            
USER_TOKEN_BUDGET = int(os.environ.get('EVAL_USER_TOKEN_BUDGET', '13500'))
MAX_MODEL_LEN = int(os.environ.get('EVAL_MAX_MODEL_LEN', '16384'))
MAX_NEW_TOKENS = int(os.environ.get('EVAL_MAX_NEW_TOKENS', '512'))
                                                         
                                                      
                                                
                                                        
BERTSCORE_MODEL = os.environ.get('EVAL_BERTSCORE_MODEL', 'roberta-large')
DECODE_TEMPERATURE = float(os.environ.get('EVAL_DECODE_TEMPERATURE', '0.4'))
                                                                            
                                                   
DECODE_SEED = 20260906
                                               
GPU_MEM_UTIL = float(os.environ.get('EVAL_GPU_MEM_UTIL', '0.85'))
                                             
GPU_FREE_THRESHOLD_MIB = int(os.environ.get('EVAL_GPU_FREE_MIB', '4096'))
BUILD_WORKERS = 48                                      

SYSTEM_PROMPT = ('You are a helpful assistant. Write only the requested text itself, '
                 'with no preamble, no explanation, and no surrounding quotation marks.')


def format_duration(seconds):
    seconds = int(round(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return '%02d小时%02d分钟%02d秒' % (hours, minutes, seconds)


def save_evaluation_result(model_name, algorithm, method, retrieval_seconds,
                           model_seconds, infer_seconds, metrics, created_date,
                           details, **metadata):
    """统一保存 summary 和逐样本 detail。"""
    elapsed = retrieval_seconds + model_seconds
    results = {
        'benchmark': 'LongLaMP (user-based test)',
        'method': method,
        'algorithm': algorithm,
        'model': model_name,
        'infer_seconds': round(infer_seconds, 1),
        'preprocessing_seconds': round(retrieval_seconds, 1),
        'preprocessing_elapsed': format_duration(retrieval_seconds),
        'generation_and_metrics_seconds': round(model_seconds, 1),
        'generation_and_metrics_elapsed': format_duration(model_seconds),
        'elapsed_seconds': round(elapsed, 1),
        'elapsed': format_duration(elapsed),
        'bertscore_model': BERTSCORE_MODEL,
        'tasks': metrics,
        'created_date': created_date,
    }
    if algorithm != 'base':
        results['retrieval_seconds'] = round(retrieval_seconds, 1)
        results['retrieval_elapsed'] = format_duration(retrieval_seconds)
    results.update(metadata)
    stem = '%s_%s_%s' % (model_name, algorithm, created_date)
    summary_path = os.path.join(RESULTS_DIR, stem + '_sum.json')
    detail_path = os.path.join(RESULTS_DIR, stem + '_detail.json')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as output:
        json.dump(results, output, ensure_ascii=False, indent=2)
    detail_result = {
        'benchmark': 'LongLaMP (user-based test)',
        'model': model_name,
        'algorithm': algorithm,
        'created_date': created_date,
        'num_samples': len(details),
        'details': details,
    }
    with open(detail_path, 'w', encoding='utf-8') as output:
        json.dump(detail_result, output, ensure_ascii=False, indent=2)
    return summary_path, detail_path, results


def _clip(text, budget=PROFILE_CHAR_BUDGET):
    return ' '.join(str(text or '').split())[:budget]

def dataset_group(spec=None, default_group='longlamp'):
    """把任务选择字符串归一成数据集组名（longlamp / amazon / lamp / mixed）。

    训练脚本用它给产出模型名加后缀（-long / -amazon / -lamp），这样"模型名"
    就自带"该用哪批测试集"的信息，评测时不会张冠李戴。
    """
    spec = (spec or default_group).strip()
    tasks = set()
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if tok == 'all':
            tasks |= set(TASKS)
        elif tok in DATASET_GROUPS:
            tasks |= set(DATASET_GROUPS[tok])
        elif tok in TASKS:
            tasks.add(tok)
    hits = [g for g, members in DATASET_GROUPS.items() if tasks & set(members)]
    return hits[0] if len(hits) == 1 else 'mixed'


def dataset_suffix(spec=None, default_group='longlamp'):
    """数据集组名对应的模型名后缀；混合多组时用 -mixed。"""
    group = dataset_group(spec, default_group)
    return DATASET_SUFFIX.get(group, '-mixed')


def with_dataset_suffix(name, spec=None, default_group='longlamp'):
    """给模型名补上数据集后缀；已带任一已知后缀时原样返回（避免重复追加）。"""
    known = tuple(DATASET_SUFFIX.values()) + ('-mixed',)
    if name.endswith(known):
        return name
    return name + dataset_suffix(spec, default_group)


def load_task(task):
    """读取某任务 test 划分的全部 parquet 分片"""
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(task_dir(task), 'test-*.parquet')))
    if not files:
        return None
    rows = []
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256):
            rows.extend(batch.to_pylist())
    return rows

def strip_thinking(text):
    """剥掉 Qwen3 的 <think>...</think> 段，只保留正式答案。

    不剥的话思考内容会被当成答案参与 ROUGE/BLEU 打分。三种情况都要覆盖：
      * 完整成对 —— 取最后一个 </think> 之后的内容；
      * 只有闭合标签（enable_thinking=True 时模板已经把 <think> 放进 prompt，
        模型只补内容和 </think>）—— 同上；
      * 只有开标签、生成被 max_tokens 截断 —— 整段都是思考，没有答案可取，
        返回空串，让它按空预测计分，而不是把思考混进答案里。
    """
    if '</think>' in text:
        return text.split('</think>')[-1]
    if '<think>' in text:
        return ''
    return text


def postprocess(text):
    """去掉思考段、模型爱加的前缀和整体包裹的引号；长文本保留换行，不截首行"""
    text = strip_thinking(text).strip()
    for prefix in ('Abstract:', 'Review:', 'Post:', 'Content:', 'Answer:', 'Here is the abstract:',
                   'Here is the review:', 'Here is the post:'):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    if len(text) > 1 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return text.strip()


def compute_metrics(preds, refs, bert_scorer=None):
    """计算 ROUGE、BLEU、METEOR 和 BERTScore。"""
    from rouge_score import rouge_scorer
    import sacrebleu
    from nltk.translate.meteor_score import meteor_score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
    r1 = rl = meteor = 0.0
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        r1 += s['rouge1'].fmeasure
        rl += s['rougeL'].fmeasure
        meteor += meteor_score([r.split()], p.split())
    n = max(len(preds), 1)
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    if bert_scorer is None:
        from bert_score import BERTScorer
        bert_scorer = BERTScorer(lang='en', model_type=BERTSCORE_MODEL, device='cuda:0',
                                 batch_size=128, rescale_with_baseline=False)
    precision, recall, f1 = bert_scorer.score(preds, refs)
    return {
        'rouge-1': round(r1 / n, 4),
        'rouge-L': round(rl / n, 4),
        'bleu': round(bleu, 4),
        'meteor': round(meteor / n, 4),
        'bertscore_precision': round(float(precision.mean()), 4),
        'bertscore_recall': round(float(recall.mean()), 4),
        'bertscore_f1': round(float(f1.mean()), 4),
    }


def _metric_worker(task, preds, refs, gpu_id, out_path):
    """单任务指标 worker：三个任务分别使用一张 GPU 并行计算 BERTScore。"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    metrics = compute_metrics(preds, refs)
    metrics['n'] = len(preds)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f)


def available_gpus(reserve_threshold_mib=None):
    """只用空闲的卡：显存占用超过阈值的卡留给别的任务（例如正在跑的训练）。

    recall.py 常驻占用约 708 MiB，阈值取 4 GiB 既能过滤真正在训练的卡，
    又不会把 recall.py 占着的空闲卡误判成忙。
    与训练共卡时用 EVAL_GPU_FREE_MIB 放宽阈值。
    """
    if reserve_threshold_mib is None:
        reserve_threshold_mib = GPU_FREE_THRESHOLD_MIB
    query = ('nvidia-smi --query-gpu=index,memory.used '
             '--format=csv,noheader,nounits')
    free = []
    for line in os.popen(query).read().strip().splitlines():
        index, used = [part.strip() for part in line.split(',')]
        if int(used) < reserve_threshold_mib:
            free.append(int(index))
    if not free:
        raise RuntimeError('没有空闲 GPU 可用于评测')
    return free


def run_metrics(per_task, gpu_ids=None):
    """并行计算各任务指标，返回 task -> metrics。"""
    import multiprocessing as mp
    import shutil
    import tempfile
    gpu_ids = gpu_ids or available_gpus()
    ctx = mp.get_context('spawn')
    tmp = tempfile.mkdtemp(prefix='longlamp_metrics_')
    try:
        procs = []
        for order, task in enumerate(t for t in TASKS if t in per_task):
            gpu_id = gpu_ids[order % len(gpu_ids)]
            path = os.path.join(tmp, '%s.json' % task)
            d = per_task[task]
            p = ctx.Process(target=_metric_worker,
                            args=(task, d['preds'], d['refs'], gpu_id, path))
            p.start()
            procs.append((task, p, path))
        results = {}
        for task, p, path in procs:
            p.join()
            if p.exitcode != 0 or not os.path.exists(path):
                raise RuntimeError('%s 指标进程失败，exitcode=%s' %
                                   (task, p.exitcode))
            with open(path, encoding='utf-8') as f:
                results[task] = json.load(f)
        return results
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _infer_shard(gpu_id, prompts, out_path, model_path):
    """单卡 worker：只可见一张卡，独立起一个 vLLM 实例跑自己那份 prompt，结果写临时文件。

    3B 模型单卡就装得下，手写数据并行（8 个单卡实例）比 tensor_parallel_size=8
    快得多——后者每层都要跨卡通信，而 vLLM 的 LLM 离线接口又不支持进程内数据并行。
    用 Process 而不是 Pool：Pool 的 worker 是 daemon 进程，vLLM 还要再 fork
    自己的 EngineCore 子进程，会直接报 daemonic processes are not allowed to have children。
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ.setdefault('VLLM_LOGGING_LEVEL', 'WARNING')
                                                 
    os.environ['VLLM_PORT'] = str(21000 + gpu_id * 64)
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_path, dtype='bfloat16', max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=GPU_MEM_UTIL, enable_prefix_caching=True,
              tensor_parallel_size=1)
                                                                 
                                         
    outs = llm.generate(prompts, SamplingParams(
        temperature=DECODE_TEMPERATURE, max_tokens=MAX_NEW_TOKENS, seed=DECODE_SEED))
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump([o.outputs[0].text for o in outs], f)


def run_inference(prompts, gpu_ids, model_path):
    """按空闲卡切片并行推理，再按原顺序拼回"""
    import multiprocessing as mp
    import shutil
    import tempfile
    if isinstance(gpu_ids, int):                          
        gpu_ids = list(range(gpu_ids))
    shards = len(gpu_ids)
    ctx = mp.get_context('spawn')
    tmp = tempfile.mkdtemp(prefix='longlamp_infer_')
    try:
        procs = []
        for order, gpu_id in enumerate(gpu_ids):
            path = os.path.join(tmp, '%d.json' % gpu_id)
            p = ctx.Process(target=_infer_shard,
                            args=(gpu_id, prompts[order::shards], path, model_path))
            p.start()
            procs.append((order, gpu_id, p, path))
        merged = [None] * len(prompts)
        for order, gpu_id, p, path in procs:
            p.join()
            if p.exitcode != 0 or not os.path.exists(path):
                raise RuntimeError('GPU %d 的推理进程失败，exitcode=%s' %
                                   (gpu_id, p.exitcode))
            with open(path, encoding='utf-8') as f:
                texts = json.load(f)
            for k, t in enumerate(texts):
                merged[order + k * shards] = t
        return merged
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



