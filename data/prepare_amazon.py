"""准备 Amazon Review（Books / Movies_and_TV / CDs_and_Vinyl）的训练/测试数据。

参考 juntaoyou/NextQuill 的 create-dataset.py / personal_dataset.py：
  - 数据源 McAuley-Lab/Amazon-Reviews-2023 的原始 jsonl（raw/review_categories、
    raw/meta_categories）。datasets>=3 不再支持其加载脚本，这里直接流式读 jsonl。
  - 每个用户的评论在原始文件里是**连续**的（已实测），所以可以边流边按 user 聚合，
    user_id 一变就 flush，内存 O(1)，攒够目标用户数即early-stop，不下载整份文件。
  - 每个用户按时间排序：最后一条为 target（要生成的评论），其余为 profile（历史）。
  - meta 提供 asin -> (title, description)，第二遍流式扫 meta 只保留需要的 asin。

产物与 LongLaMP 同构（{reviewerId, input, output, profile}），落到
  eval/data/amazon-sft/<task>_user/{train,test}-00000-of-00001.parquet
task 名：book_review / movie_review / cd_review。这样现有 parquet 加载器可直接复用，
只需在任务注册表里给 Amazon 任务挂上各自的 render/full_context。

按 CLAUDE.md 约定放在 logs/（辅助脚本）。直接 `python logs/prepare_amazon.py` 运行。
可用环境变量覆盖规模：AMZ_TRAIN_USERS / AMZ_TEST_USERS / AMZ_MIN_HIS / AMZ_MAX_HIS /
AMZ_SCAN_CAP / AMZ_CATS（逗号分隔，取 Books,Movies_and_TV,CDs_and_Vinyl 子集）。
"""
import json
import os
import sys
import time

import requests
from huggingface_hub import hf_hub_url

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HUAYI_DIR = os.path.dirname(BASE_DIR)
OUT_ROOT = os.path.join(HUAYI_DIR, 'eval', 'data', 'amazon-sft')
REPO = 'McAuley-Lab/Amazon-Reviews-2023'

CAT_TO_TASK = {
    'Books': 'book_review',
    'Movies_and_TV': 'movie_review',
    'CDs_and_Vinyl': 'cd_review',
}
CATS = [c.strip() for c in os.environ.get(
    'AMZ_CATS', 'CDs_and_Vinyl,Movies_and_TV,Books').split(',') if c.strip()]

TRAIN_USERS = int(os.environ.get('AMZ_TRAIN_USERS', '3000'))
TEST_USERS = int(os.environ.get('AMZ_TEST_USERS', '800'))
MIN_HISTORY = int(os.environ.get('AMZ_MIN_HIS', '5'))                     
MAX_HISTORY = int(os.environ.get('AMZ_MAX_HIS', '20'))                       
SCAN_CAP = int(os.environ.get('AMZ_SCAN_CAP', '4000000'))             
DESC_CHARS = int(os.environ.get('AMZ_DESC_CHARS', '600'))                         
TEXT_CHARS = int(os.environ.get('AMZ_TEXT_CHARS', '4000'))          
SEED = int(os.environ.get('AMZ_SEED', '20260910'))
                 


def stream_lines(rel_path):
    """流式逐行读 HF 上的 jsonl（不落地整份文件）。"""
    url = hf_hub_url(REPO, rel_path, repo_type='dataset')
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                yield line


def clip(text, n):
    return ' '.join(str(text or '').split())[:n]


def desc_to_text(desc):
    """meta 的 description 可能是 list[str] 或 str。"""
    if isinstance(desc, list):
        return ' '.join(str(x) for x in desc)
    return str(desc or '')


def select_users(category):
    """第一遍：流式按 user 聚合，攒够 TRAIN_USERS+TEST_USERS 个合格用户即停。

    合格 = 评论数 >= MIN_HISTORY + 1（留一条做 target）。返回 {user_id: [review,...]}
    以及需要的 parent_asin 集合。评论在文件内按 user 连续，故只需维护"当前用户"缓冲。
    """
    need = TRAIN_USERS + TEST_USERS
    selected, asins = {}, set()
    cur_uid, cur_buf = None, []
    scanned = 0
    started = time.time()

    def flush(uid, buf):
        if uid is None or len(buf) < MIN_HISTORY + 1:
            return
        if len(selected) >= need:
            return
        buf.sort(key=lambda r: r.get('timestamp') or 0)
        buf = buf[-(MAX_HISTORY + 1):]                                           
        selected[uid] = buf
        for r in buf:
            asins.add(r['parent_asin'])

    from huggingface_hub import hf_hub_download
    dl0 = time.time()
    review_path = hf_hub_download(REPO, 'raw/review_categories/%s.jsonl' % category,
                                  repo_type='dataset')
    print('  [%s] review 文件就绪 %.0fs -> 本地扫描' % (category, time.time() - dl0),
          flush=True)
    review_fh = open(review_path, 'r', encoding='utf-8')
    for line in review_fh:
        scanned += 1
        try:
            ex = json.loads(line)
        except Exception:
            continue
        uid = ex.get('user_id')
        pa = ex.get('parent_asin') or ex.get('asin')
        if not uid or not pa:
            continue
        rec = {'parent_asin': pa, 'rating': ex.get('rating'),
               'title': clip(ex.get('title'), 200),
               'text': clip(ex.get('text'), TEXT_CHARS),
               'timestamp': ex.get('timestamp')}
        if uid != cur_uid:
            flush(cur_uid, cur_buf)
            if len(selected) >= need or scanned > SCAN_CAP:
                break
            cur_uid, cur_buf = uid, [rec]
        else:
            cur_buf.append(rec)
        if scanned % 500000 == 0:
            print('  [%s] 已扫 %d 条评论，合格用户 %d/%d（%.0fs）'
                  % (category, scanned, len(selected), need, time.time() - started),
                  flush=True)
    review_fh.close()
    flush(cur_uid, cur_buf)
    print('  [%s] 选出用户 %d，需要 asin %d，扫描 %d 条'
          % (category, len(selected), len(asins), scanned), flush=True)
    return selected, asins
              


def fetch_meta(category, needed):
    """先把 meta jsonl 下载到本地（hf_hub_download，实测 ~55MB/s，Books 14.7GB 约 5 分钟），
    再本地逐行扫描，只保留 needed 里 asin 的 (title, description)，找齐即 early-stop。

    改用"下载到本地再扫"而不是 HTTP 逐行流式：后者 iter_lines 吞吐很低（0.95GB 扫几分钟
    都没扫完），下载走的是 hub 的块传输，快一个量级，且本地扫描不受网络抖动影响。
    """
    from huggingface_hub import hf_hub_download
    started = time.time()
    path = hf_hub_download(REPO, 'raw/meta_categories/meta_%s.jsonl' % category,
                           repo_type='dataset')
    print('  [%s meta] 下载完成 %.0fs -> 本地扫描' % (category, time.time() - started),
          flush=True)
    meta = {}
    scanned = 0
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            scanned += 1
            try:
                ex = json.loads(line)
            except Exception:
                continue
            pa = ex.get('parent_asin')
            if pa in needed and pa not in meta:
                meta[pa] = (clip(ex.get('title'), 300),
                            clip(desc_to_text(ex.get('description')), DESC_CHARS))
                if len(meta) >= len(needed):
                    break
            if scanned % 1000000 == 0:
                print('  [%s meta] 已扫 %d，命中 %d/%d（%.0fs）'
                      % (category, scanned, len(meta), len(needed),
                         time.time() - started), flush=True)
    print('  [%s meta] 命中 %d/%d，扫描 %d 条（%.0fs）'
          % (category, len(meta), len(needed), scanned, time.time() - started),
          flush=True)
    return meta


def build_records(selected, meta):
    """把每个用户拼成 {reviewerId, input, output, profile}，与 LongLaMP 同构。

    input  = 目标 item 的标题/描述 + 目标评分/标题 的自包含指令（要生成的就是正文）。
    output = 目标评论正文。
    profile= 历史评论，每条含 item 标题/描述、评分、评论标题、评论正文。
    """
    records = []
    for uid, buf in selected.items():
        target = buf[-1]
        hist = buf[:-1]
        t_title, t_desc = meta.get(target['parent_asin'], ('', ''))
        if not t_title and not t_desc:
            continue                                                   
        if not (target.get('text') or '').strip():
            continue                                                
        profile = []
        for r in hist:
            i_title, i_desc = meta.get(r['parent_asin'], ('', ''))
            profile.append({
                'item_title': i_title, 'item_desc': i_desc,
                'rating': r.get('rating'), 'review_title': r.get('title'),
                'review_text': r.get('text'), 'timestamp': r.get('timestamp'),
            })
        if len(profile) < MIN_HISTORY:
            continue
        inp = ('[Item Title]: %s\n[Item Description]: %s\n'
               '[Review Rating]: %s\n[Review Title]: %s\n'
               'Write the personalized review text for this item.'
               % (t_title, t_desc, target.get('rating'), target.get('review_title')
                  if target.get('review_title') is not None else target.get('title')))
        records.append({'reviewerId': uid, 'input': inp,
                        'output': target['text'], 'profile': profile})
    return records
              


def write_parquet(rows, path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def main():
    import random
    os.makedirs(OUT_ROOT, exist_ok=True)
    summary = {}
    for cat in CATS:
        task = CAT_TO_TASK[cat]
        print('==== %s -> %s ====' % (cat, task), flush=True)
        selected, asins = select_users(cat)
        if not selected:
            print('  [%s] 无合格用户，跳过' % cat, flush=True)
            continue
        meta = fetch_meta(cat, asins)
        records = build_records(selected, meta)
        random.Random(SEED).shuffle(records)
        n_test = min(TEST_USERS, len(records) // 5)
        test_rows, train_rows = records[:n_test], records[n_test:]
        out_dir = os.path.join(OUT_ROOT, '%s_user' % task)
        write_parquet(train_rows, os.path.join(out_dir, 'train-00000-of-00001.parquet'))
        write_parquet(test_rows, os.path.join(out_dir, 'test-00000-of-00001.parquet'))
        summary[task] = {'train': len(train_rows), 'test': len(test_rows),
                         'total': len(records)}
        print('  [%s] 写入 train=%d test=%d -> %s'
              % (task, len(train_rows), len(test_rows), out_dir), flush=True)
    print('==== 完成 ====', flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()



