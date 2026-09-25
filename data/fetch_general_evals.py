"""下载 9 个新的通用能力评测集到 eval/data/disaster/<name>/，统一存成 parquet。
走 HF 镜像（HF_ENDPOINT 已指向 hf-mirror.com）。可重复执行，已存在的跳过。
用法：python fetch_general_evals.py
"""
import os
import sys

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'eval', 'data', 'disaster'))

                                                  
JOBS = [
    ('piqa', 'ybisk/piqa', None, {'validation': 'validation'}),
    ('openbookqa', 'allenai/openbookqa', 'main', {'test': 'test'}),
    ('commonsenseqa', 'tau/commonsense_qa', None, {'validation': 'validation'}),
    ('arc_easy', 'allenai/ai2_arc', 'ARC-Easy', {'test': 'test'}),
    ('arc_challenge', 'allenai/ai2_arc', 'ARC-Challenge', {'test': 'test'}),
    ('sciq', 'allenai/sciq', None, {'test': 'test'}),
    ('mmlu_pro', 'TIGER-Lab/MMLU-Pro', None,
     {'test': 'test', 'validation': 'validation'}),
    ('gsm8k', 'openai/gsm8k', 'main', {'test': 'test'}),
    ('logiqa', 'lucasmccabe/logiqa', None, {'test': 'test'}),
]


def main():
    from datasets import load_dataset
    for name, repo, config, splits in JOBS:
        out_dir = os.path.join(BASE, name)
        done = all(os.path.exists(os.path.join(out_dir, '%s.parquet' % local))
                   for local in splits)
        if done:
            print('跳过 %s（已存在）' % name, flush=True)
            continue
        os.makedirs(out_dir, exist_ok=True)
        for local, hf_split in splits.items():
            try:
                ds = (load_dataset(repo, config, split=hf_split)
                      if config else load_dataset(repo, split=hf_split))
                path = os.path.join(out_dir, '%s.parquet' % local)
                ds.to_parquet(path)
                print('OK %s/%s -> %d 条' % (name, local, len(ds)), flush=True)
            except Exception as exc:                                     
                print('FAIL %s/%s: %s' % (name, local, exc), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
