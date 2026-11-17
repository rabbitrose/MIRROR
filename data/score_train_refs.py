"""Reviewer 1 W2 基线：给训练用的用户参考答案打质量分，筛出高质量子集。

与所有方法共用同一个 800 条训练池（train_sft.sample_rows，同任务同种子，与骨干无关），
用主实验的 G-Eval 协议（Qwen3-30B-A3B，reference-free，SOURCE INPUT = 任务输入）
给每条参考答案打 1-5 分，保留得分最高的一半（并列按固定种子打散）。
输出 data/ref_quality/<task>.json：{"keep": [row_key...], "scores": {row_key: overall}}
row_key = md5(input + "\x00" + output)，训练端据此过滤。
用法：TRAIN_TASKS=<task> python data/score_train_refs.py
"""
import hashlib, json, os, random, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'eval')); sys.path.insert(0, os.path.join(ROOT, 'train'))
os.environ.setdefault('LONGLAMP_DATA_DIR', '/data/datasets/longlamp-sft-filtered')
os.environ.setdefault('LAMP_DATA_DIR', '/data/datasets/lamp-sft-filtered')
import train_sft as common
import eval_quality as Q

KEEP_FRAC = float(os.environ.get('RQF_KEEP_FRAC', '0.5'))
OUT_DIR = os.path.join(ROOT, 'data', 'ref_quality')


def row_key(row):
    return hashlib.md5((row['input'] + '\x00' + row['output']).encode('utf-8')).hexdigest()


def main():
    task = os.environ['TRAIN_TASKS']
    rows = common.sample_rows(task, common.load_train_rows()[task])
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    judge = Q.resolve_judge_path()
    tok = AutoTokenizer.from_pretrained(judge)
    prompts = [Q.build_prompt(tok, task, r['input'], r['output']) for r in rows]
    llm = LLM(model=judge, dtype='bfloat16', max_model_len=Q.JUDGE_MAX_MODEL_LEN,
              gpu_memory_utilization=Q.JUDGE_GPU_MEM_UTIL, tensor_parallel_size=1,
              enforce_eager=Q.JUDGE_ENFORCE_EAGER, trust_remote_code=True)
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=Q.JUDGE_MAX_NEW))
    scores = {}
    for r, o in zip(rows, outs):
        s = Q.parse_score(o.outputs[0].text)
        scores[row_key(r)] = s['overall'] if s else 0.0
    keys = list(scores)
    random.Random('rqf-%s' % task).shuffle(keys)
    keys.sort(key=lambda k: -scores[k])
    keep = keys[:int(round(len(keys) * KEEP_FRAC))]
    os.makedirs(OUT_DIR, exist_ok=True)
    allv = list(scores.values()); kv = [scores[k] for k in keep]
    rec = {'task': task, 'judge': Q.JUDGE_MODEL, 'pool': len(rows), 'keep_frac': KEEP_FRAC,
           'mean_all': sum(allv) / len(allv), 'mean_keep': sum(kv) / len(kv),
           'min_keep': min(kv), 'keep': keep, 'scores': scores}
    json.dump(rec, open(os.path.join(OUT_DIR, task + '.json'), 'w'), indent=1)
    print('%s: pool=%d keep=%d mean_all=%.3f mean_keep=%.3f min_keep=%.2f'
          % (task, len(rows), len(keep), rec['mean_all'], rec['mean_keep'], rec['min_keep']), flush=True)


if __name__ == '__main__':
    main()
    import subprocess, signal
    def _kids(pid):
        res = []
        for c in subprocess.run(['pgrep', '-P', str(pid)], capture_output=True, text=True).stdout.split():
            res += _kids(int(c)) + [int(c)]
        return res
    for c in _kids(os.getpid()):
        try:
            os.kill(c, signal.SIGKILL)
        except OSError:
            pass
    os._exit(0)
