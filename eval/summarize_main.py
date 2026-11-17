"""汇总个性化主表：按基座分块，每个 ID 任务列出所有方法的 ROUGE-1 / METEOR / BERTScore-F1。

命名约定（和 summarize_ood.py 一致的两个坑）：
  基座的三种无训练基线结果文件名不含任务，算法名带任务标识（base_abs / rag2_abs / lastK5_abs）；
  训练模型的算法名只有 base，任务信息在模型名里（-long / -lamp-news / -lamp-title）。
  Qwen2.5-3B 的结果文件用全名 Qwen2.5-3B-Instruct，检查点短名是 Qwen2.5-3B。

用法：python eval/summarize_main.py  [> TestResults/main_table.md]
"""
import glob
import json
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import eval_common as common

METRICS = ("rouge-1", "meteor", "bertscore_f1")
TASKS = {
    "abstract_generation": ("-long", "abs"),
    "news_headline": ("-lamp-news", "news"),
    "scholarly_title": ("-lamp-title", "title"),
}
# 无训练基线：显示名 -> 基座上的算法名前缀
BASELINES = (("Base (zero-shot)", "base"), ("RAG top-2", "rag2"), ("Last-K (K=5)", "lastK5"))
# 训练方法：显示名 -> 检查点中缀
TRAINED = (("ContextSFT", "contextSFT"), ("PerCE", "PerCE"), ("NextQuill", "NextQuill"),
           ("MIRROR", "MIRROR"), ("MIRROR-F", "MIRROR-F"))
BACKBONES = ("Qwen3-1.7B", "Qwen3-4B", "Qwen2.5-3B")
BASE_RESULT_NAME = {"Qwen2.5-3B": "Qwen2.5-3B-Instruct"}


def latest(model, algo):
    pat = re.compile(r"^%s_%s_\d{8}_sum\.json$" % (re.escape(model), re.escape(algo)))
    files = sorted(f for f in glob.glob(os.path.join(common.RESULTS_DIR,
                                                     "%s_%s_*_sum.json" % (model, algo)))
                   if pat.match(os.path.basename(f)))
    if not files:
        return None
    return json.load(open(files[-1], encoding="utf-8")).get("tasks", {})


def score(model, algo, task):
    tasks = latest(model, algo)
    if not tasks or task not in tasks:
        return None
    return {m: tasks[task][m] for m in METRICS}


def main():
    for b in BACKBONES:
        base_res = BASE_RESULT_NAME.get(b, b)
        print("\n## %s" % b)
        for task, (suf, tag) in TASKS.items():
            rows = []
            for name, algo in BASELINES:
                rows.append((name, score(base_res, "%s_%s" % (algo, tag), task)))
            for name, mid in TRAINED:
                rows.append((name, score("%s-%s%s" % (b, mid, suf), "base", task)))
            if not any(s for _, s in rows):
                continue
            print("\n### %s" % task)
            print("  %-18s %8s %8s %8s" % ("method", "R-1", "METEOR", "BS-F1"))
            best = {m: max(s[m] for _, s in rows if s) for m in METRICS}
            for name, s in rows:
                if s is None:
                    print("  %-18s %8s %8s %8s" % (name, "-", "-", "-"))
                    continue
                cells = ["%8.4f%s" % (s[m], "*" if s[m] == best[m] else " ") for m in METRICS]
                print("  %-18s %s" % (name, "".join(c.rjust(9) for c in cells)))


if __name__ == "__main__":
    main()
