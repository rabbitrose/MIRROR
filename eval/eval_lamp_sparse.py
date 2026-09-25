"""LongLaMP BM25 稀疏检索评测。

评测 LongLaMP 三个可用任务（user-based 划分）：
  abstract_generation  论文摘要生成
  product_review       商品评论生成（书 / 影视 / CD 等 Amazon 品类混合）
  topic_writing        Reddit 帖子正文生成
第四个任务 Personalized Email Completion 依赖 LDC 授权的 Avocado 邮件数据，公开仓库没有，无法评测。

个性化做法与 LongLaMP 一致：BM25 从用户历史 profile 里召回最相关的若干条拼进 prompt。

按 CLAUDE.md 约定：直接 `python eval_lamp_sparse.py` 运行，不接收命令行参数。
"""
import glob
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


def profile_corpus(task, p):
    """profile 条目参与 BM25 检索的文本"""
    if task == 'abstract_generation':
        return '%s %s' % (p.get('title', ''), p.get('abstract', ''))
    if task == 'product_review':
        return '%s %s' % (p.get('description', ''), p.get('reviewText', ''))
    if task in AMAZON_TASKS:
        return '%s %s' % (p.get('item_title', ''), p.get('review_text', ''))
    if task == 'news_headline':
        return '%s %s' % (p.get('title', ''), p.get('text', ''))
    if task == 'scholarly_title':
        return '%s %s' % (p.get('title', ''), p.get('abstract', ''))
    return '%s %s' % (p.get('summary', ''), p.get('content', ''))


def render_profiles(task, profs):
    """把召回的 profile 渲染成 few-shot 风格的 prompt 前缀"""
    if task == 'abstract_generation':
        parts = ['"%s" is a title for the abstract "%s"' % (_clip(p.get('title'), 300), _clip(p.get('abstract')))
                 for p in profs]
    elif task == 'product_review':
        parts = ['the review with summary "%s" and rating %s for the product "%s" is "%s"'
                 % (_clip(p.get('summary'), 200), p.get('overall', ''),
                    _clip(p.get('description'), 500), _clip(p.get('reviewText')))
                 for p in profs]
    elif task in AMAZON_TASKS:
                                                         
        parts = ['the review with rating %s and title "%s" for the item "%s" (%s) is "%s"'
                 % (p.get('rating', ''), _clip(p.get('review_title'), 200),
                    _clip(p.get('item_title'), 200), _clip(p.get('item_desc'), 400),
                    _clip(p.get('review_text')))
                 for p in profs]
    elif task == 'news_headline':
                               
        parts = ['"%s" is a headline for the article "%s"'
                 % (_clip(p.get('title'), 300), _clip(p.get('text')))
                 for p in profs]
    elif task == 'scholarly_title':
                               
        parts = ['"%s" is a title for the abstract "%s"'
                 % (_clip(p.get('title'), 300), _clip(p.get('abstract')))
                 for p in profs]
    else:
        parts = ['the reddit post with summary "%s" is "%s"' % (_clip(p.get('summary'), 300), _clip(p.get('content')))
                 for p in profs]
    return '%s. Following the given patterns, ' % ', and '.join(parts)


def build_one(args):
    """单条样本：BM25 召回 top-k profile，拼成用户内容"""
    from rank_bm25 import BM25Okapi
    task, inp, profile = args
    if not profile:
        return inp
    corpus = [profile_corpus(task, p) for p in profile]
    try:
        bm25 = BM25Okapi([c.split() for c in corpus])
        profs = bm25.get_top_n(inp.split(), list(profile), n=NUM_RETRIEVE)
    except Exception:
        profs = list(profile)[:NUM_RETRIEVE]
    return render_profiles(task, profs) + inp


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


def main():
    t_start = time.time()
                                                      
    import nltk
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        if not nltk.download('wordnet', quiet=True):
            raise RuntimeError('METEOR 依赖的 NLTK WordNet 下载失败')
    models = []
    for root in (MODELS_DIR, TRAINED_DIR):
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if name not in TARGET_MODEL_NAMES:
                continue
            if not os.path.isdir(path) or name.startswith('__'):
                continue
            if not os.path.isfile(os.path.join(path, 'config.json')):
                continue
            if not glob.glob(os.path.join(path, '*.safetensors')):
                continue
            if not (os.path.isfile(os.path.join(path, 'tokenizer.json')) or
                    os.path.isfile(os.path.join(path, 'tokenizer.model'))):
                continue
            models.append(path)
    if not models:
        print('没有找到完整模型：%s' % ', '.join(sorted(TARGET_MODEL_NAMES)))
        return
    print('发现 %d 个模型：%s' %
          (len(models), ', '.join(os.path.basename(p) for p in models)), flush=True)

    loaded = {}
    for task in EVAL_TASKS:
        rows = load_task(task)
        if not rows:
            print('跳过 %s：parquet 缺失' % task)
            continue
        loaded[task] = rows
        print('%s: %d 条测试样本' % (task, len(rows)), flush=True)
    if not loaded:
        print('没有可评测的数据')
        return

                                                         
    print('构造 prompt（BM25 召回 top-%d）...' % NUM_RETRIEVE, flush=True)
    retrieval_started = time.time()
    jobs, index, refs, raw_inputs, sample_ids = [], [], [], [], []
    for task, rows in loaded.items():
        for row_number, r in enumerate(rows):
            jobs.append((task, r['input'], r.get('profile') or []))
            index.append(task)
            refs.append(r['output'])
            raw_inputs.append(r['input'])
            sample_ids.append('%s-%06d' % (task, row_number))
    with Pool(BUILD_WORKERS) as pool:
        contents = pool.map(build_one, jobs, chunksize=4)
    retrieval_seconds = time.time() - retrieval_started
    print('prompt 构造完成，用时 %.1fs' % retrieval_seconds, flush=True)

    gpu_ids = available_gpus()
    if os.environ.get('EVAL_GPU_IDS'):
        gpu_ids = [int(x) for x in os.environ['EVAL_GPU_IDS'].split(',') if x.strip()]
    print('本次评测使用 GPU：%s' % gpu_ids, flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    date = datetime.now().strftime('%Y%m%d')
    failures = []
    for model_path in models:
        model_name = os.path.basename(model_path)
        model_started = time.time()
        print('\n===== %s =====' % model_name, flush=True)
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_path)
                                                                          
                                                                  
                                                         
            tok.truncation_side = 'left'
            enc = tok(contents, truncation=True, max_length=USER_TOKEN_BUDGET,
                      add_special_tokens=False)
            clipped = tok.batch_decode(enc['input_ids'], skip_special_tokens=True)
            prompts = [tok.apply_chat_template(
                [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': c}],
                tokenize=False, add_generation_prompt=True) for c in clipped]
            print('开始推理：%d 条，GPU %s 各起一个 vLLM 实例' %
                  (len(prompts), gpu_ids), flush=True)
            t0 = time.time()
            texts = run_inference(prompts, gpu_ids, model_path)
            infer_sec = time.time() - t0
            print('推理完成，用时 %.1fs；开始计算指标' % infer_sec, flush=True)
            per_task = {t: {'preds': [], 'refs': []} for t in loaded}
            for task, ref, txt in zip(index, refs, texts):
                per_task[task]['preds'].append(postprocess(txt))
                per_task[task]['refs'].append(ref)
            metrics = run_metrics(per_task, gpu_ids)
            for task in metrics:
                metrics[task]['n'] = len(per_task[task]['preds'])
            model_seconds = time.time() - model_started
            details = [{
                'sample_id': sample_id,
                'task': task,
                'input': raw_input,
                'model_input': model_input,
                'model_output': postprocess(text),
                'golden_truth': ref,
            } for sample_id, task, raw_input, model_input, ref, text in zip(
                sample_ids, index, raw_inputs, clipped, refs, texts)]
            summary_path, detail_path, results = save_evaluation_result(
                model_name=model_name,
                algorithm='sparse',
                method='BM25 sparse retrieval',
                retrieval_seconds=retrieval_seconds,
                model_seconds=model_seconds,
                infer_seconds=infer_sec,
                metrics=metrics,
                created_date=date,
                details=details,
                retriever='rank_bm25.BM25Okapi',
                retriever_pooling='bm25_okapi_lexical_score',
                num_retrieve=NUM_RETRIEVE,
            )
            print('结果已覆盖写入: %s 和 %s（耗时 %s）' %
                  (summary_path, detail_path, results['elapsed']), flush=True)
        except Exception as exc:
            print('模型 %s 评测失败：%s' % (model_name, exc), flush=True)
            traceback.print_exc()
            failures.append(model_name)
    if failures:
        raise RuntimeError('以下模型评测失败：%s' % ', '.join(failures))
    print('\n全部 %d 个模型评测完成，总用时 %.1fs' %
          (len(models), time.time() - t_start), flush=True)


if __name__ == '__main__':
    main()
