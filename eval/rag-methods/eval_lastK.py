"""LongLaMP / Amazon / LaMP 的 last-K 历史评测：只保留时间上最近的 K 条用户历史。

与 eval_lamp_base.py 的**唯一**区别是历史用多少条：
    eval_lamp_base.py  用全部历史（只受 60000 字符 + 13500 token 预算约束）
    本文件            先把 profile 砍到最近 K 条（默认 K=5），再走同一套拼接/截断

用途：做"历史长度 -> 个性化收益"的消融。全量历史里，模型到底用到了多少？把历史压到
5 条还能保住多少个性化效果？这条曲线是判断方法是"真的在建模用户偏好"还是"在吃长上下文
的信息量"的关键证据。

时间最近怎么判定（逐数据集实测的字段，见 RECENCY_FIELDS）：
  * abstract_generation 有 year，book/movie/cd_review 有 timestamp -> 按该字段升序排，取末尾 K 条；
  * product_review / topic_writing / news_headline / scholarly_title 没有日期字段
    -> 退回"原始顺序即时间正序"这一已验证事实（eval_lamp_base.full_context 的注释里记了
    实测依据：abstract_generation 上 index~year 的 Spearman 中位数 0.821，99.2% 的用户为
    正相关），直接取 profile 末尾 K 条。
  两条路径都保证"取到的是最近的 K 条"，并保持时间正序渲染。

支持的数据集（由 EVAL_TASKS 选择，与训练侧 TRAIN_TASKS 同一套注册表）：
    EVAL_TASKS=longlamp  abstract_generation / product_review / topic_writing
    EVAL_TASKS=amazon    book_review / movie_review / cd_review
    EVAL_TASKS=lamp      news_headline / scholarly_title
    EVAL_TASKS=all       全部 8 个任务

正式运行方式：python eval_laskK.py
  EVAL_LAST_K         保留的历史条数（默认 5）
  EVAL_MODEL_NAMES    逗号分隔模型名
  EVAL_TASKS          数据集/任务选择
  EVAL_THINKING       off（默认）/ on
  EVAL_MAX_NEW_TOKENS 生成上限（短文本任务建议 128）
  EVAL_GPU_IDS        限定可用卡

结果覆盖写入 TestResults/<模型名>_lastK<K>_<日期>_{sum,detail}.json。
汇报指标只有三项：ROUGE-1 / METEOR / BERTScore。
"""
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))                                             
import eval_common as common
LAST_K = int(os.environ.get('EVAL_LAST_K', '5'))
if LAST_K <= 0:
    raise RuntimeError('EVAL_LAST_K 必须为正整数：%s' % LAST_K)

MODEL_NAMES = ['Qwen3-1.7B']
if os.environ.get('EVAL_MODEL_NAMES'):
    MODEL_NAMES = [n.strip() for n in os.environ['EVAL_MODEL_NAMES'].split(',')
                   if n.strip()]
TRAINED_DIR = common.TRAINED_DIR
                                               
FULL_CONTEXT_CHAR_BUDGET = int(os.environ.get('EVAL_FULL_CONTEXT_CHARS', '60000'))
EVAL_THINKING = os.environ.get('EVAL_THINKING', 'off')
if EVAL_THINKING not in ('off', 'on'):
    raise RuntimeError('EVAL_THINKING 只能取 off/on：%s' % EVAL_THINKING)

                                       
_ALGO_BASE = 'lastK%d_think' % LAST_K if EVAL_THINKING == 'on' else 'lastK%d' % LAST_K
if abs(common.DECODE_TEMPERATURE - 0.4) < 1e-9:
    ALGORITHM = _ALGO_BASE
else:
    ALGORITHM = '%s_t%s' % (
        _ALGO_BASE, ('%g' % common.DECODE_TEMPERATURE).replace('.', 'p'))
_ALGO_SUFFIX = os.environ.get('EVAL_ALGO_SUFFIX', '').strip()
if _ALGO_SUFFIX:
    ALGORITHM = '%s_%s' % (ALGORITHM, _ALGO_SUFFIX)

                                        
RECENCY_FIELDS = {
    'abstract_generation': 'year',
    'book_review': 'timestamp',
    'movie_review': 'timestamp',
    'cd_review': 'timestamp',
}
                               
REPORTED = ('rouge-1', 'meteor', 'bertscore_f1')
def resolve_model(name):
    """基座模型在 models 下，训练后的模型在 TrainModels 下"""
    for root in (common.MODELS_DIR, TRAINED_DIR):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    raise RuntimeError('模型不存在：%s' % name)


def keep_last_k(task, profile, k):
    """只保留时间上最近的 K 条历史，保持时间正序渲染。

    有日期字段（year / timestamp）的任务按该字段升序排、取末尾 K 条；没有日期字段的
    直接取 profile 的末尾 K 条（数据集原始顺序已验证与时间正相关）。
    """
    if not profile or k <= 0:
        return []
    field = RECENCY_FIELDS.get(task)
    if field and all(field in p for p in profile):
                          
        sorted_prof = sorted(profile, key=lambda p: p[field])
        selected = sorted_prof[-k:]
    else:
                              
        selected = profile[-k:]
    return selected


def full_context_lastk(task, inp, profile):
    """与 eval_lamp_base.full_context 同逻辑，但 profile 先砍到 K 条。

    返回 (prefix, question, 纳入条数)。
    """
    kept = keep_last_k(task, profile, LAST_K)
                                            
    selected = []
    used = 0
    for item in reversed(kept):
        rendered = common.render_profiles(task, [item])
        if selected and used + len(rendered) > FULL_CONTEXT_CHAR_BUDGET:
            break
        selected.append(item)
        used += len(rendered)
    selected.reverse()
    prefix = common.render_profiles(task, selected) if selected else ''
    return prefix, inp, len(selected)


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
    """渲染对话模板；不认 enable_thinking 的模板自动退回无参渲染。"""
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
    """构造 last-K 历史输入，多个模型共用同一份（按 EVAL_TASKS 覆盖三个数据集）。"""
    loaded = {}
    for task in common.EVAL_TASKS:
        rows = common.load_task(task)
        if rows:
            loaded[task] = rows
            print('%s: %d 条测试样本' % (task, len(rows)), flush=True)
    started = time.time()
    prefixes, questions, index, refs, sample_ids = [], [], [], [], []
    profile_counts, included_counts = [], []
    for task, rows in loaded.items():
        for row_number, row in enumerate(rows):
            profile = row.get('profile') or []
            prefix, question, included = full_context_lastk(
                task, row['input'], profile)
            prefixes.append(prefix)
            questions.append(question)
            index.append(task)
            refs.append(row['output'])
            sample_ids.append('%s-%06d' % (task, row_number))
            profile_counts.append(len(profile))
            included_counts.append(included)
    return {
        'loaded': loaded, 'prefixes': prefixes, 'questions': questions,
        'index': index, 'refs': refs, 'raw_inputs': questions,
        'sample_ids': sample_ids, 'profile_counts': profile_counts,
        'included_counts': included_counts,
        'preprocessing_seconds': time.time() - started,
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
        macro = {k: sum(metrics[t][k] for t in tasks) / len(tasks)
                 for k in REPORTED}
        print('  %-20s ROUGE-1=%.4f  METEOR=%.4f  BERTScore=%.4f'
              % ('宏平均', macro['rouge-1'], macro['meteor'],
                 macro['bertscore_f1']), flush=True)
        return macro
    return {}
def evaluate(model_name, shared, gpu_ids, date):
    model_path = resolve_model(model_name)
    print('\n===== %s (last-K=%d) =====' % (model_name, LAST_K), flush=True)
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
        'profile_entries_included': included,
        'prefix_tokens_kept': kept,
    } for sample_id, task, raw_input, model_input, text, ref, total, included, kept in zip(
        shared['sample_ids'], shared['index'], shared['raw_inputs'], clipped,
        texts, shared['refs'], shared['profile_counts'], shared['included_counts'],
        prefix_tokens)]

    model_seconds = time.time() - model_started
    summary_path, detail_path, results = common.save_evaluation_result(
        model_name=model_name,
        algorithm=ALGORITHM,
        method='Last-K user context (K=%d, thinking=%s)' % (LAST_K, EVAL_THINKING),
        retrieval_seconds=shared['preprocessing_seconds'],
        model_seconds=model_seconds,
        infer_seconds=infer_seconds,
        metrics=metrics,
        created_date=date,
        details=details,
        context_policy='last_K_profile_entries',
        last_k=LAST_K,
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
    avg_included = (sum(shared['included_counts']) / max(len(shared['included_counts']), 1))
    print('last-K=%d, 平均实际纳入 %.1f 条历史' % (LAST_K, avg_included), flush=True)
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




