"""LongLaMP 训练集过滤（PerCE / NextQuill 式的数据清洗）。

这两个工作都只公开了"对 LaMP/LongLaMP 做 context cropping 与 dataset filtering"，
没有发布具体脚本，因此这里把规则写成可直接复现的形式，并输出每条规则丢弃的数量。
**只过滤训练集**：val / test 保持官方原样（评测口径不能因为过滤而改变）。

规则（按顺序）：
  1. 历史条数 >= MIN_HISTORY：用户没有任何写作历史时学不到个人风格；
  2. gold 输出 token 数在 [MIN_OUT, MAX_OUT]：过短的输出没有风格信息，过长的超出训练预算；
  3. gold 输出语言为英文（langdetect，仅对 >= 20 字符的输出判定）；
  4. 同一任务内按归一化输出文本去重。
上下文侧不做裁剪：训练时按统一额度（历史+问题 13500 token）逐样本截断。

用法：python data/filter_longlamp.py            # 产出到 /data/datasets/longlamp-sft-filtered
"""
import glob
import hashlib
import json
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HUAYI_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(HUAYI_DIR, 'eval'))
import eval_common as common

SRC = os.environ.get('FILTER_SRC', '/data/datasets/longlamp-sft')
DST = os.environ.get('FILTER_DST', '/data/datasets/longlamp-sft-filtered')
# 默认过滤 LongLaMP 三任务；FILTER_TASKS 可换成 LaMP 的 news_headline,scholarly_title
TASKS = tuple(t.strip() for t in os.environ.get(
    'FILTER_TASKS', 'abstract_generation,product_review,topic_writing').split(',') if t.strip())
MIN_HISTORY = int(os.environ.get("FILTER_MIN_HISTORY", "3"))
# 输出长度下限按任务体裁定：长文本（abstract/review/post）取 16 token；
# 标题类任务（LaMP-4 headline / LaMP-5 title）gold 本身就只有十几个 token，取 3。
MIN_OUT = int(os.environ.get("FILTER_MIN_OUT", "16"))
MAX_OUT = int(os.environ.get("FILTER_MAX_OUT", "512"))
TOKENIZER = os.environ.get('FILTER_TOKENIZER', os.path.join(HUAYI_DIR, 'models', 'Qwen3-1.7B'))


def normalize(text):
    return re.sub(r'\s+', ' ', text or '').strip().lower()


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from langdetect import detect, LangDetectException

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    os.makedirs(DST, exist_ok=True)
    for task in TASKS:
        src_dir = os.path.join(SRC, task + '_user')
        dst_dir = os.path.join(DST, task + '_user')
        os.makedirs(dst_dir, exist_ok=True)
        rows = []
        for path in sorted(glob.glob(os.path.join(src_dir, 'train-*.parquet'))):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
                rows.extend(batch.to_pylist())
        stat = {'raw': len(rows), 'no_history': 0, 'short_out': 0, 'long_out': 0,
                'non_english': 0, 'duplicate': 0}
        seen, kept = set(), []
        for row in rows:
            profile = row.get('profile') or []
            output = (row.get('output') or '').strip()
            if len(profile) < MIN_HISTORY:
                stat['no_history'] += 1
                continue
            n_tok = len(tokenizer(output, add_special_tokens=False)['input_ids'])
            if n_tok < MIN_OUT:
                stat['short_out'] += 1
                continue
            if n_tok > MAX_OUT:
                stat['long_out'] += 1
                continue
            if len(output) >= 20:
                try:
                    if detect(output) != 'en':
                        stat['non_english'] += 1
                        continue
                except LangDetectException:
                    pass
            key = hashlib.sha1(normalize(output).encode('utf-8')).hexdigest()
            if key in seen:
                stat['duplicate'] += 1
                continue
            seen.add(key)
            kept.append(row)
        out_path = os.path.join(dst_dir, 'train-00000-of-00001.parquet')
        pq.write_table(pa.Table.from_pylist(kept), out_path)
        stat['kept'] = len(kept)
        print('%s: %s' % (task, json.dumps(stat, ensure_ascii=False)), flush=True)
        # val / test 原样链接过去，保证评测口径不变
        for split in ('val', 'test'):
            for path in sorted(glob.glob(os.path.join(src_dir, '%s-*.parquet' % split))):
                link = os.path.join(dst_dir, os.path.basename(path))
                if os.path.islink(link) or os.path.exists(link):
                    os.remove(link)
                os.symlink(os.path.abspath(path), link)
    print('过滤完成，训练集写入 %s' % DST, flush=True)


if __name__ == '__main__':
    main()
