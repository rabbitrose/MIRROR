"""为 PSOPD 预制 user profile report：teacher 读用户完整历史，写成一份完整用户画像报告。

PSOPD 的 teacher 把 privileged context 里的 user history 换成这份画像报告
（teacher 看到的是 report + gold answer），所以要在训练前离线把每个 (task, QA pair)
的报告生成好，训练时按 key 直接查表。

报告是分节的（voice/tone、vocabulary、length & structure、topics、sentiment、
imitation checklist），不是 3-6 句的简 summary——20260917 的实测表明简 summary 在
Amazon 短评论任务上把具体措辞和评分倾向压掉了，teacher 的 privileged 优势因此消失。

约定：
  - summary 只由**用户历史**生成，不含 gold（gold 在训练时才拼到 teacher 输入里）。
  - 逐 QA pair 生成，key = sha1(task \\0 input \\0 output)[:16]，train_psopd.py 用
    完全相同的实现复算 key 查表。
  - teacher 用哪个模型由 SUMMARY_MODEL 指定（自蒸馏时 = student 基座）。产物按模型隔离：
      eval/data/summaries/<model>/summaries.json      {key: summary}
      eval/data/summaries/<model>/meta.json           生成配置与计数
  - 断点续跑：已在 summaries.json 里的 key 跳过，只补新的。

环境变量：
  SUMMARY_MODEL        teacher/基座模型名（默认 Qwen3-1.7B）
  SUMMARY_TASKS        任务选择，同 resolve_tasks（默认 all = 6 个任务）
  SUMMARY_SPLITS       train,test（默认两者都做）
  SUMMARY_TRAIN_CAP    train split 每任务最多生成多少条（默认 0 = 全部；只想覆盖
                       训练实际采样的子集时设成 >= 训练的 SAMPLES_PER_TASK）
  SUMMARY_MAX_NEW      画像报告最大 token 数（默认 1024）
  SUMMARY_HIST_BUDGET  喂给 teacher 的历史 token 上限（默认 12000）
  EVAL_GPU_IDS / EVAL_GPU_MEM_UTIL  与 recall.py 共卡时限定卡与显存份额

按 CLAUDE.md 约定放在 logs/（辅助脚本）。直接 `python logs/prepare_summary.py` 运行。
"""
import glob
import hashlib
import json
import os
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))                       
HUAYI_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(HUAYI_DIR, 'eval'))
import eval_lamp_sparse as common                      
import eval_lamp_base as evalfull                       
SUMMARY_MODEL = os.environ.get('SUMMARY_MODEL', 'Qwen3-1.7B')
SUMMARY_TASKS = os.environ.get('SUMMARY_TASKS', 'all')
SUMMARY_SPLITS = [s.strip() for s in
                  os.environ.get('SUMMARY_SPLITS', 'train,test').split(',') if s.strip()]
SUMMARY_TRAIN_CAP = int(os.environ.get('SUMMARY_TRAIN_CAP', '0'))
SUMMARY_TEST_CAP = int(os.environ.get('SUMMARY_TEST_CAP', '0'))
SUMMARY_MAX_NEW = int(os.environ.get('SUMMARY_MAX_NEW', '1024'))
SUMMARY_HIST_BUDGET = int(os.environ.get('SUMMARY_HIST_BUDGET', '12000'))
THINKING = os.environ.get('SUMMARY_THINKING', 'off')
OUT_ROOT = os.path.join(HUAYI_DIR, 'eval', 'data', 'summaries')

TASK_DESC = {
    'abstract_generation': 'writing a paper abstract from a title',
    'product_review': 'writing a product review',
    'topic_writing': 'writing a Reddit post body from a title',
    'book_review': 'writing an Amazon book review',
    'movie_review': 'writing an Amazon movie/TV review',
    'cd_review': 'writing an Amazon music (CD/Vinyl) review',
}

SUMMARY_SYSTEM = (
    'You are an expert writing analyst. You read a single user\'s past writing and '
    'produce a complete, faithful profile report of that specific user, detailed '
    'enough that another writer could reproduce their next piece in the same voice. '
    'Ground every claim in the evidence you were given and never invent facts.')
                                                                  
                                                          
                                           
                          
SUMMARY_INSTRUCTION = (
    '\n\nBased only on the writing history above, write a complete user profile report '
    'for the task of %s. Use exactly these sections, each as a short paragraph headed '
    'by its name:\n'
    '1. Voice and tone — how this user sounds: formality, warmth, humour, directness, '
    'whether they address the reader.\n'
    '2. Vocabulary and phrasing — the words, jargon, intensifiers and sentence shapes '
    'they actually reuse; quote a few short characteristic phrases.\n'
    '3. Length and structure — their typical piece length (give an approximate word '
    'count), how they open, how they organise the middle, and how they end; note '
    'whether they stop abruptly or wrap up.\n'
    '4. Recurring topics and preferences — what subjects, genres, authors, artists or '
    'product categories they keep returning to, and what they value or dislike.\n'
    '5. Sentiment and judgement tendencies — whether they skew positive or critical, '
    'how they hedge or qualify praise, and how they voice complaints.\n'
    '6. Imitation checklist — 3 to 5 concrete, imperative rules another writer should '
    'follow to pass as this user.\n'
    'Be specific and evidence-based. Do not quote whole sentences from the history and '
    'do not mention that you were given a history.')


def summary_key(task, inp, out):
    """train_psopd.py 必须用完全相同的实现来查表。"""
    h = hashlib.sha1()
    h.update(task.encode('utf-8')); h.update(b'\x00')
    h.update((inp or '').encode('utf-8')); h.update(b'\x00')
    h.update((out or '').encode('utf-8'))
    return h.hexdigest()[:16]


def render_summary_prompt(tokenizer, task, history):
    user = ('Here is the writing history of a single user (their past examples):\n\n%s%s'
            % (history, SUMMARY_INSTRUCTION % TASK_DESC.get(task, task)))
    messages = [{'role': 'system', 'content': SUMMARY_SYSTEM},
                {'role': 'user', 'content': user}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=(THINKING == 'on'))
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
                


def build_history(tokenizer, task, row):
    """用户完整历史（不含 gold），按 token 预算右侧保留最近条目。"""
    prefix, _question, _ = evalfull.full_context(
        task, row['input'], row.get('profile') or [])
    if not prefix:
        return ''
    ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
    if len(ids) > SUMMARY_HIST_BUDGET:
        ids = ids[-SUMMARY_HIST_BUDGET:]
    return tokenizer.decode(ids, skip_special_tokens=True)


def load_rows(task, split):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(
        common.task_dir(task), '%s-*.parquet' % split)))
    rows = []
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256):
            rows.extend(batch.to_pylist())
    return rows


                                                                     
                                                                
                                                       
                                           
_SFT_SEED = 20260904


def sample_rows(task, rows, limit):
    import random
    if len(rows) <= limit:
        return rows
    return random.Random('%s-%d' % (task, _SFT_SEED)).sample(rows, limit)
             


def main():
    from transformers import AutoTokenizer
    started = time.time()
    model_path = os.path.join(HUAYI_DIR, 'models', SUMMARY_MODEL)
    if not os.path.isdir(model_path):
        model_path = os.path.join(common.TRAINED_DIR, SUMMARY_MODEL)
    if not os.path.isdir(model_path):
        raise RuntimeError('模型不存在：%s' % SUMMARY_MODEL)

    tasks = common.resolve_tasks(SUMMARY_TASKS, 'all')
    if not tasks:
        raise RuntimeError('没有可用任务')
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    gpu_ids = common.available_gpus()
    if os.environ.get('EVAL_GPU_IDS'):
        gpu_ids = [int(x) for x in os.environ['EVAL_GPU_IDS'].split(',') if x.strip()]
                                                                 
                                                 
    os.environ['EVAL_MAX_NEW_TOKENS'] = str(SUMMARY_MAX_NEW)
    print('summary teacher=%s  tasks=%s  splits=%s  GPU=%s'
          % (SUMMARY_MODEL, tasks, SUMMARY_SPLITS, gpu_ids), flush=True)

    out_dir = os.path.join(OUT_ROOT, SUMMARY_MODEL)
    os.makedirs(out_dir, exist_ok=True)
    store_path = os.path.join(out_dir, 'summaries.json')
    summaries = {}
    if os.path.isfile(store_path):
        with open(store_path, encoding='utf-8') as fh:
            summaries = json.load(fh)
        print('已有 %d 条 summary，增量补齐' % len(summaries), flush=True)

    counts = {}
    for task in tasks:
        for split in SUMMARY_SPLITS:
            rows = load_rows(task, split)
            if not rows:
                print('跳过 %s/%s：无数据' % (task, split), flush=True)
                continue
            if split == 'train' and SUMMARY_TRAIN_CAP > 0:
                rows = sample_rows(task, rows, SUMMARY_TRAIN_CAP)
            elif split == 'test' and SUMMARY_TEST_CAP > 0:
                rows = sample_rows(task, rows, SUMMARY_TEST_CAP)
            todo, keys = [], []
            for row in rows:
                key = summary_key(task, row['input'], row['output'])
                if key in summaries:
                    continue
                todo.append(build_history(tokenizer, task, row))
                keys.append(key)
            print('%s/%s：共 %d 条，需生成 %d 条'
                  % (task, split, len(rows), len(todo)), flush=True)
            if not todo:
                continue
            prompts = [render_summary_prompt(tokenizer, task, h) for h in todo]
            texts = common.run_inference(prompts, gpu_ids, model_path)
            for key, text in zip(keys, texts):
                summaries[key] = common.postprocess(text)
            counts['%s/%s' % (task, split)] = len(todo)
            with open(store_path, 'w', encoding='utf-8') as fh:
                json.dump(summaries, fh, ensure_ascii=False)
            print('  已写入 %s（累计 %d 条）' % (store_path, len(summaries)), flush=True)

    meta = {
        'model': SUMMARY_MODEL, 'tasks': tasks, 'splits': SUMMARY_SPLITS,
        'train_cap': SUMMARY_TRAIN_CAP, 'test_cap': SUMMARY_TEST_CAP,
        'max_new_tokens': SUMMARY_MAX_NEW,
        'prompt_version': 'profile_report_v2',
        'hist_budget': SUMMARY_HIST_BUDGET, 'thinking': THINKING,
        'key': 'sha1(task \\0 input \\0 output)[:16]',
        'total_summaries': len(summaries), 'generated_this_run': counts,
        'created_date': datetime.now().strftime('%Y%m%d'),
        'elapsed': common.format_duration(time.time() - started),
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print('完成：%s（%d 条，耗时 %s）'
          % (store_path, len(summaries), meta['elapsed']), flush=True)


if __name__ == '__main__':
    main()
