"""LongLaMP / Amazon / LaMP 的稀疏检索（BM25）评测：只喂 H 条最相关的用户历史。

与 eval_lamp_base.py 的**唯一**区别是历史怎么选：
    eval_lamp_base.py  按原始顺序纳入全部历史（只受字符/token 预算约束）
    eval_laskK.py      只取时间最近的 K 条
    本文件            用 BM25 对当前 question 检索最相关的 H 条（默认 H=5）

三者构成一组干净的消融：全量 / 近期 / 相关。它回答的是"个性化到底靠什么"——
是靠信息量（全量）、靠时效（近期）、还是靠与当前问题的相关性（检索）。

实现要点：
  * BM25 语料用 common.profile_corpus(task, p)，该函数已覆盖 8 个任务的字段口径
    （LongLaMP 三任务 / Amazon 三任务 / LaMP 两任务）；
  * 查询用样本自己的 input（question），取 top-H，按相关度降序渲染（标准 RAG 口径）；
  * BM25 是 CPU 瓶颈，用多进程并行构造（BUILD_WORKERS）；
  * thinking 显式传 enable_thinking=False。这一点必须和 eval_lamp_base.py 对齐：
    eval_lamp_sparse.py 因为漏传该参数，Qwen3 模板默认开思考，2026-09-17 实测 30%
    样本输出为空、BERTScore 被压到 0.55。本文件不复现那个坑。

支持的数据集（EVAL_TASKS 选择，与训练侧 TRAIN_TASKS 同一套注册表）：
    EVAL_TASKS=longlamp / amazon / lamp / all

正式运行方式：python eval_RAG.py
  EVAL_RAG_H          检索条数（默认 5）
  EVAL_MODEL_NAMES    逗号分隔模型名
  EVAL_TASKS          数据集/任务选择
  EVAL_THINKING       off（默认）/ on
  EVAL_MAX_NEW_TOKENS 生成上限（短文本任务建议 128）
  EVAL_GPU_IDS        限定可用卡

结果覆盖写入 TestResults/<模型名>_rag<H>_<日期>_{sum,detail}.json。
汇报指标只有三项：ROUGE-1 / METEOR / BERTScore。
"""
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))                                             
import eval_lamp_sparse as common
RAG_H = int(os.environ.get('EVAL_RAG_H', '5'))
if RAG_H <= 0:
    raise RuntimeError('EVAL_RAG_H 必须为正整数：%s' % RAG_H)
BUILD_WORKERS = int(os.environ.get('EVAL_BUILD_WORKERS', '48'))

MODEL_NAMES = ['Qwen3-1.7B']
if os.environ.get('EVAL_MODEL_NAMES'):
    MODEL_NAMES = [n.strip() for n in os.environ['EVAL_MODEL_NAMES'].split(',')
                   if n.strip()]
TRAINED_DIR = common.TRAINED_DIR
FULL_CONTEXT_CHAR_BUDGET = int(os.environ.get('EVAL_FULL_CONTEXT_CHARS', '60000'))
EVAL_THINKING = os.environ.get('EVAL_THINKING', 'off')
if EVAL_THINKING not in ('off', 'on'):
    raise RuntimeError('EVAL_THINKING 只能取 off/on：%s' % EVAL_THINKING)

                                     
_ALGO_BASE = 'rag%d_think' % RAG_H if EVAL_THINKING == 'on' else 'rag%d' % RAG_H
if abs(common.DECODE_TEMPERATURE - 0.4) < 1e-9:
    ALGORITHM = _ALGO_BASE
else:
    ALGORITHM = '%s_t%s' % (
        _ALGO_BASE, ('%g' % common.DECODE_TEMPERATURE).replace('.', 'p'))
_ALGO_SUFFIX = os.environ.get('EVAL_ALGO_SUFFIX', '').strip()
if _ALGO_SUFFIX:
    ALGORITHM = '%s_%s' % (ALGORITHM, _ALGO_SUFFIX)

                               
REPORTED = ('rouge-1', 'meteor', 'bertscore_f1')


def resolve_model(name):
    """基座模型在 models 下，训练后的模型在 TrainModels 下"""
    for root in (common.MODELS_DIR, TRAINED_DIR):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    raise RuntimeError('模型不存在：%s' % name)


def retrieve_one(args):
    """单条样本：BM25 对 question 检索 top-H 条 profile，返回 (prefix, 纳入条数)。

    放在模块顶层是为了能被 multiprocessing.Pool pickle。检索失败（空 profile、
    分词后为空导致 BM25 构造异常）时退回"取前 H 条"，与 eval_lamp_sparse 一致。
    """
    task, inp, profile = args
    if not profile:
        return '', 0
    corpus = [common.profile_corpus(task, p) for p in profile]
    try:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([c.split() for c in corpus])
        profs = bm25.get_top_n(inp.split(), list(profile), n=RAG_H)
    except Exception:                                                
        profs = list(profile)[:RAG_H]
                                           
    selected, used = [], 0
    for item in profs:
        rendered = common.render_profiles(task, [item])
        if selected and used + len(rendered) > FULL_CONTEXT_CHAR_BUDGET:
            break
        selected.append(item)
        used += len(rendered)
    prefix = common.render_profiles(task, selected) if selected else ''
    return prefix, len(selected)
def clip_content(tokenizer, prefix, question, budget):
    """question 必须完整保留，剩余额度才留给历史前缀。同 eval_lamp_base.clip_content。"""
    question_ids = tokenizer(question, add_special_tokens=False)['input_ids']
    if len(question_ids) >= budget:
        kept_text = tokenizer.decode(question_ids[:budget], skip_special_tokens=True)
        return kept_text, 0
    room = budget - len(question_ids)
    if not prefix:
        return question, 0
    prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
    if len(prefix_ids) > room:
        prefix_ids = prefix_ids[-room:]
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    return prefix_text + question, len(prefix_ids)


def render_prompt(tokenizer, content):
    """渲染对话模板；显式传 enable_thinking，不重复 eval_lamp_sparse 漏传的坑。"""
    messages = [{'role': 'system', 'content': common.SYSTEM_PROMPT},
                {'role': 'user', 'content': content}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=(EVAL_THINKING == 'on'))
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def prepare_inputs():
    """构造 BM25 top-H 输入，多个模型共用同一份（按 EVAL_TASKS 覆盖三个数据集）。"""
    from multiprocessing import Pool
    loaded = {}
    for task in common.EVAL_TASKS:
        rows = common.load_task(task)
        if rows:
            loaded[task] = rows
            print('%s: %d 条测试样本' % (task, len(rows)), flush=True)
    started = time.time()
    jobs, index, refs, sample_ids, questions, profile_counts = [], [], [], [], [], []
    for task, rows in loaded.items():
        for row_number, row in enumerate(rows):
            profile = row.get('profile') or []
            jobs.append((task, row['input'], profile))
            questions.append(row['input'])
            index.append(task)
            refs.append(row['output'])
            sample_ids.append('%s-%06d' % (task, row_number))
            profile_counts.append(len(profile))
    print('BM25 检索 top-%d（%d 条样本，%d 进程）...'
          % (RAG_H, len(jobs), BUILD_WORKERS), flush=True)
    with Pool(BUILD_WORKERS) as pool:
        pairs = pool.map(retrieve_one, jobs, chunksize=4)
    prefixes = [p for p, _ in pairs]
    included_counts = [n for _, n in pairs]
    seconds = time.time() - started
    print('检索完成，用时 %.1fs' % seconds, flush=True)
    return {
        'loaded': loaded, 'prefixes': prefixes, 'questions': questions,
        'index': index, 'refs': refs, 'raw_inputs': questions,
        'sample_ids': sample_ids, 'profile_counts': profile_counts,
        'included_counts': included_counts,
        'preprocessing_seconds': seconds,
    }


def report(metrics):
    """只打印 ROUGE-1 / METEOR / BERTScore，并给出宏平均。"""
    tasks = list(metrics)
    for task in tasks:
        row = metrics[task]
        print('  %-20s ROUGE-1=%.4f  METEOR=%.4f  BERTScore=%.4f  (n=%d)'
              % (task, row['rouge-1'], row['meteor'], row['bertscore_f1'],
                 row.get('n', 0)), flush=True)
    if tasks:
        macro = {k: sum(metrics[t][k] for t in tasks) / len(tasks) for k in REPORTED}
        print('  %-20s ROUGE-1=%.4f  METEOR=%.4f  BERTScore=%.4f'
              % ('宏平均', macro['rouge-1'], macro['meteor'],
                 macro['bertscore_f1']), flush=True)
        return macro
    return {}
def evaluate(model_name, shared, gpu_ids, date):
    model_path = resolve_model(model_name)
    print('\n===== %s (BM25 top-%d) =====' % (model_name, RAG_H), flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    clipped, prefix_tokens = [], []
    for prefix, question in zip(shared['prefixes'], shared['questions']):
        content, kept = clip_content(tokenizer, prefix, question,
                                     common.USER_TOKEN_BUDGET)
        clipped.append(content)
        prefix_tokens.append(kept)
    prompts = [render_prompt(tokenizer, content) for content in clipped]

    model_started = time.time()
    infer_started = time.time()
    texts = common.run_inference(prompts, gpu_ids, model_path)
    infer_seconds = time.time() - infer_started
    print('推理完成，用时 %.1fs；开始计算指标' % infer_seconds, flush=True)

    per_task = {task: {'preds': [], 'refs': []} for task in shared['loaded']}
    for task, ref, text in zip(shared['index'], shared['refs'], texts):
        per_task[task]['preds'].append(common.postprocess(text))
        per_task[task]['refs'].append(ref)
    metrics = common.run_metrics(per_task, gpu_ids)

    details = [{
        'sample_id': sample_id,
        'task': task,
        'input': raw_input,
        'model_input': model_input,
        'model_output': common.postprocess(text),
        'golden_truth': ref,
        'profile_entries_total': total,
        'profile_entries_retrieved': included,
        'prefix_tokens_kept': kept,
    } for sample_id, task, raw_input, model_input, text, ref, total, included, kept in zip(
        shared['sample_ids'], shared['index'], shared['raw_inputs'], clipped,
        texts, shared['refs'], shared['profile_counts'], shared['included_counts'],
        prefix_tokens)]

    model_seconds = time.time() - model_started
    summary_path, detail_path, results = common.save_evaluation_result(
        model_name=model_name,
        algorithm=ALGORITHM,
        method='Sparse retrieval (BM25) top-%d user history (thinking=%s)'
               % (RAG_H, EVAL_THINKING),
        retrieval_seconds=shared['preprocessing_seconds'],
        model_seconds=model_seconds,
        infer_seconds=infer_seconds,
        metrics=metrics,
        created_date=date,
        details=details,
        context_policy='bm25_top_H_profile_entries',
        rag_h=RAG_H,
        retriever='BM25Okapi (rank_bm25), query=question, order=relevance desc',
        max_model_len=common.MAX_MODEL_LEN,
        user_token_budget=common.USER_TOKEN_BUDGET,
        decode_temperature=common.DECODE_TEMPERATURE,
        decode_seed=common.DECODE_SEED,
        max_new_tokens=common.MAX_NEW_TOKENS,
        thinking=EVAL_THINKING,
    )
    print('结果已覆盖写入：%s 和 %s（耗时 %s）' %
          (summary_path, detail_path, results['elapsed']), flush=True)
    report(metrics)


def main():
    started = time.time()
    import nltk
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        if not nltk.download('wordnet', quiet=True):
            raise RuntimeError('METEOR 依赖的 NLTK WordNet 不可用')
    shared = prepare_inputs()
    if not shared['loaded']:
        raise RuntimeError('没有可评测的数据')
    avg = sum(shared['included_counts']) / max(len(shared['included_counts']), 1)
    print('BM25 top-%d，平均实际纳入 %.1f 条历史' % (RAG_H, avg), flush=True)
    gpu_ids = common.available_gpus()
    if os.environ.get('EVAL_GPU_IDS'):
        gpu_ids = [int(x) for x in os.environ['EVAL_GPU_IDS'].split(',') if x.strip()]
    print('本次评测使用 GPU：%s' % gpu_ids, flush=True)
    date = datetime.now().strftime('%Y%m%d')
    for model_name in MODEL_NAMES:
        evaluate(model_name, shared, gpu_ids, date)
    print('\n运行总耗时：%s' % common.format_duration(time.time() - started), flush=True)


if __name__ == '__main__':
    main()



