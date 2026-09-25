"""LongLaMP / Amazon / LaMP 的 TRSE 评测：user summary + 时间最近 H 条历史 + question。

TRSE = Temporal-Recency + Summary-Enhanced 的上下文组织方式。与 eval_lamp_base.py 的
区别是「喂给模型的用户信息」由三段组成，而不是一整串原始历史：

    eval_lamp_base.py  全部原始历史 + question
    eval_laskK.py      时间最近 K 条 + question
    eval_RAG.py        BM25 top-H 条 + question
    本文件            user summary（对全部历史的画像总结）+ 时间最近 H 条 + question

动机：前三档是"给多少历史"的消融，本档是"历史怎么组织"的改进——summary 提供跨全部
历史的**全局画像**（语气、词汇、长度习惯、话题偏好），最近 H 条提供**近期写作实况**。
全局 + 近期互补：summary 不会因为截断丢掉早期信息，近期条目反映当前的风格状态。

时间最近怎么判定（逐数据集实测的字段，见 RECENCY_FIELDS）：
  * abstract_generation 有 year，book/movie/cd_review 有 timestamp -> 按该字段升序排，取末尾 H 条；
  * product_review / topic_writing / news_headline / scholarly_title 没有日期字段
    -> 退回"原始顺序即时间正序"这一已验证事实（eval_lamp_base.full_context 注释里记了
    实测依据：abstract_generation 上 index~year 的 Spearman 中位数 0.821，99.2% 的用户
    为正相关），直接取 profile 末尾 H 条。
  两条路径都保证取到最近 H 条，并保持时间正序渲染。

user summary 从哪来（不在本文件里现算）：
    由 logs/prepare_summary.py 离线生成，按 key = sha1(task \\0 input \\0 output)[:16]
    查表，产物在 eval/data/summaries/<SUMMARY_MODEL>/summaries.json。
    summary **只由用户历史生成、不含 gold**（gold 只参与 key 的计算，不进 prompt）。
    表里查不到的样本会退化成「只有最近 H 条 + question」，并在日志里报缺失比例——
    缺失比例高时说明该数据集的 summary 还没生成，应先跑 prepare_summary.py。

支持的数据集（EVAL_TASKS 选择，与训练侧 TRAIN_TASKS 同一套注册表）：
    EVAL_TASKS=longlamp / amazon / lamp / all

正式运行方式：python eval_TRSE.py
  EVAL_TRSE_H         保留的最近历史条数（默认 5）
  SUMMARY_MODEL       用哪个模型生成的 summary 表（默认 Qwen3-1.7B）
  EVAL_MODEL_NAMES    逗号分隔模型名
  EVAL_TASKS          数据集/任务选择
  EVAL_THINKING       off（默认）/ on
  EVAL_MAX_NEW_TOKENS 生成上限（短文本任务建议 128）
  EVAL_GPU_IDS        限定可用卡

结果覆盖写入 TestResults/<模型名>_trse<H>_<日期>_{sum,detail}.json。
汇报指标只有三项：ROUGE-1 / METEOR / BERTScore。
"""
import hashlib
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))                                             
import eval_common as common
TRSE_H = int(os.environ.get('EVAL_TRSE_H', '5'))
if TRSE_H <= 0:
    raise RuntimeError('EVAL_TRSE_H 必须为正整数：%s' % TRSE_H)
SUMMARY_MODEL = os.environ.get('SUMMARY_MODEL', 'Qwen3-1.7B')
SUMMARY_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'data', 'summaries', SUMMARY_MODEL))
SUMMARY_PATH = os.path.join(SUMMARY_DIR, 'summaries.json')
                                                
SUMMARY_CHAR_CAP = int(os.environ.get('EVAL_TRSE_SUMMARY_CHARS', '3000'))

MODEL_NAMES = ['Qwen3-1.7B']
if os.environ.get('EVAL_MODEL_NAMES'):
    MODEL_NAMES = [n.strip() for n in os.environ['EVAL_MODEL_NAMES'].split(',')
                   if n.strip()]
TRAINED_DIR = common.TRAINED_DIR
FULL_CONTEXT_CHAR_BUDGET = int(os.environ.get('EVAL_FULL_CONTEXT_CHARS', '60000'))
EVAL_THINKING = os.environ.get('EVAL_THINKING', 'off')
if EVAL_THINKING not in ('off', 'on'):
    raise RuntimeError('EVAL_THINKING 只能取 off/on：%s' % EVAL_THINKING)

                                      
_ALGO_BASE = 'trse%d_think' % TRSE_H if EVAL_THINKING == 'on' else 'trse%d' % TRSE_H
if abs(common.DECODE_TEMPERATURE - 0.4) < 1e-9:
    ALGORITHM = _ALGO_BASE
else:
    ALGORITHM = '%s_t%s' % (
        _ALGO_BASE, ('%g' % common.DECODE_TEMPERATURE).replace('.', 'p'))
_ALGO_SUFFIX = os.environ.get('EVAL_ALGO_SUFFIX', '').strip()
if _ALGO_SUFFIX:
    ALGORITHM = '%s_%s' % (ALGORITHM, _ALGO_SUFFIX)

REPORTED = ('rouge-1', 'meteor', 'bertscore_f1')
SUMMARY_HEAD = 'Here is a profile of the user based on all of their past writing:\n'
RETRIEVED_HEAD = ('\n\nHere are the most recent items from the '
                  "user's history:\n")
                                        
RECENCY_FIELDS = {
    'abstract_generation': 'year',
    'book_review': 'timestamp',
    'movie_review': 'timestamp',
    'cd_review': 'timestamp',
}


def resolve_model(name):
    """基座模型在 models 下，训练后的模型在 TrainModels 下"""
    for root in (common.MODELS_DIR, TRAINED_DIR):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    raise RuntimeError('模型不存在：%s' % name)


def summary_key(task, inp, out):
    """必须与 logs/prepare_summary.py 的实现逐字节一致，否则查不到表。"""
    h = hashlib.sha1()
    h.update(task.encode('utf-8'))
    h.update(b'\x00')
    h.update((inp or '').encode('utf-8'))
    h.update(b'\x00')
    h.update((out or '').encode('utf-8'))
    return h.hexdigest()[:16]


def load_summaries():
    """读 summary 查表；表不存在时返回空 dict（本文件会退化成纯 last-H 档并告警）。"""
    if not os.path.exists(SUMMARY_PATH):
        print('警告：summary 表不存在 %s —— 本次将退化成「仅最近 %d 条历史」，'
              '请先跑 logs/prepare_summary.py 生成对应数据集的 summary'
              % (SUMMARY_PATH, TRSE_H), flush=True)
        return {}
    with open(SUMMARY_PATH, encoding='utf-8') as fh:
        table = json.load(fh)
    print('summary 表已载入：%s（%d 条，模型=%s）'
          % (SUMMARY_PATH, len(table), SUMMARY_MODEL), flush=True)
    return table
def recent_one(args):
    """单条样本：取时间最近的 H 条 profile，返回 (prefix, 纳入条数)。

    有日期字段（year / timestamp）的任务按该字段升序排、取末尾 H 条；没有日期字段的
    直接取 profile 末尾 H 条（数据集原始顺序已验证与时间正相关）。两条路径都保持
    时间正序渲染。
    """
    task, inp, profile = args
    if not profile:
        return '', 0
    field = RECENCY_FIELDS.get(task)
    if field and all(field in p for p in profile):
        kept = sorted(profile, key=lambda p: p[field])[-TRSE_H:]
    else:
        kept = list(profile)[-TRSE_H:]
    selected, used = [], 0
    for item in reversed(kept):                               
        rendered = common.render_profiles(task, [item])
        if selected and used + len(rendered) > FULL_CONTEXT_CHAR_BUDGET:
            break
        selected.append(item)
        used += len(rendered)
    selected.reverse()                           
    prefix = common.render_profiles(task, selected) if selected else ''
    return prefix, len(selected)


def compose_prefix(summary, recent):
    """把 user summary 与最近 H 条历史拼成一个前缀：summary 在前，近期条目在后。

    summary 放前面是刻意的：它是全局画像，先给模型"这个人是谁"的框架，再给
    "最近几条写作实况"。两段都带显式标题，避免模型把画像当成待续写的正文。
    """
    parts = []
    if summary:
        parts.append(SUMMARY_HEAD + summary.strip()[:SUMMARY_CHAR_CAP])
    if recent:
        parts.append(RETRIEVED_HEAD + recent)
    if not parts:
        return ''
    return ''.join(parts) + '\n\n'


def clip_content(tokenizer, prefix, question, budget):
    """question 必须完整保留，剩余额度才留给前缀。同 eval_lamp_base.clip_content。"""
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
    """渲染对话模板；显式传 enable_thinking，与 eval_lamp_base 对齐。"""
    messages = [{'role': 'system', 'content': common.SYSTEM_PROMPT},
                {'role': 'user', 'content': content}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=(EVAL_THINKING == 'on'))
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


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
def prepare_inputs():
    """构造 summary + top-H 输入，多个模型共用同一份（按 EVAL_TASKS 覆盖三个数据集）。"""
    table = load_summaries()
    loaded = {}
    for task in common.EVAL_TASKS:
        rows = common.load_task(task)
        if rows:
            loaded[task] = rows
            print('%s: %d 条测试样本' % (task, len(rows)), flush=True)
    started = time.time()
    jobs, index, refs, sample_ids, questions = [], [], [], [], []
    profile_counts, summaries = [], []
    missing = 0
    for task, rows in loaded.items():
        for row_number, row in enumerate(rows):
            profile = row.get('profile') or []
            jobs.append((task, row['input'], profile))
            questions.append(row['input'])
            index.append(task)
            refs.append(row['output'])
            sample_ids.append('%s-%06d' % (task, row_number))
            profile_counts.append(len(profile))
            summary = table.get(summary_key(task, row['input'], row['output']), '')
            if not summary:
                missing += 1
            summaries.append(summary)
    print('取时间最近 %d 条历史（%d 条样本）...' % (TRSE_H, len(jobs)), flush=True)
    pairs = [recent_one(j) for j in jobs]
    retrieved = [p for p, _ in pairs]
    included_counts = [n for _, n in pairs]
    prefixes = [compose_prefix(s, r) for s, r in zip(summaries, retrieved)]
    seconds = time.time() - started
    total = max(len(jobs), 1)
    print('历史选取完成，用时 %.1fs；summary 缺失 %d/%d (%.1f%%)'
          % (seconds, missing, total, 100.0 * missing / total), flush=True)
    if missing == total:
        print('警告：全部样本都没有 summary，本次等价于纯 last-%d 档' % TRSE_H,
              flush=True)
    return {
        'loaded': loaded, 'prefixes': prefixes, 'questions': questions,
        'index': index, 'refs': refs, 'raw_inputs': questions,
        'sample_ids': sample_ids, 'profile_counts': profile_counts,
        'included_counts': included_counts, 'summaries': summaries,
        'summary_missing': missing,
        'preprocessing_seconds': seconds,
    }
def evaluate(model_name, shared, gpu_ids, date):
    model_path = resolve_model(model_name)
    print('\n===== %s (summary + last-%d) =====' % (model_name, TRSE_H),
          flush=True)
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
        'has_summary': bool(summary),
        'prefix_tokens_kept': kept,
    } for sample_id, task, raw_input, model_input, text, ref, total, included,
        summary, kept in zip(
        shared['sample_ids'], shared['index'], shared['raw_inputs'], clipped,
        texts, shared['refs'], shared['profile_counts'],
        shared['included_counts'], shared['summaries'], prefix_tokens)]

    model_seconds = time.time() - model_started
    summary_path, detail_path, results = common.save_evaluation_result(
        model_name=model_name,
        algorithm=ALGORITHM,
        method='TRSE: user summary + most recent %d history items + question '
               '(thinking=%s)' % (TRSE_H, EVAL_THINKING),
        retrieval_seconds=shared['preprocessing_seconds'],
        model_seconds=model_seconds,
        infer_seconds=infer_seconds,
        metrics=metrics,
        created_date=date,
        details=details,
        context_policy='user_summary_plus_most_recent_H',
        trse_h=TRSE_H,
        summary_model=SUMMARY_MODEL,
        summary_missing=shared['summary_missing'],
        summary_char_cap=SUMMARY_CHAR_CAP,
        history_selector='most recent H entries (date field when available, '
                         'else profile tail), rendered in chronological order',
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
    print('TRSE：summary + 最近 %d 条，平均实际纳入 %.1f 条历史'
          % (TRSE_H, avg), flush=True)
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




