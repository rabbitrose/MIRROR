"""Amazon 三任务 OOD 评测：book_review / movie_review / cd_review。

模型在这三个 Amazon 品类评论生成任务上属于分布外（OOD）——训练只在 LongLaMP 摘要
或 LaMP 上进行，从未见过 Amazon 数据。本脚本复用 eval_lamp_base 的完整评测流程
（full user context、无检索、按 token 预算截断），仅把任务锁定为 amazon 三任务，
并在最后汇报 ROUGE-1 / METEOR / BERTScore（按 CLAUDE.md 文本质量口径）。

正式运行方式：python eval_ood.py
  EVAL_MODEL_NAMES   逗号分隔模型名（默认 Qwen3-1.7B）
  EVAL_GPU_IDS       限定可用卡（多进程并行时各锁一半）
结果覆盖写入 TestResults/<模型>_base_ood_<日期>_{sum,detail}.json。
"""
import glob
import json
import os

                                                              
                                                    
os.environ.setdefault("EVAL_TASKS", "amazon")
os.environ.setdefault("EVAL_ALGO_SUFFIX", "ood")
os.environ.setdefault("EVAL_MODEL_NAMES", "Qwen3-1.7B")

import eval_common as common                      
import eval_lamp_base as base                          

AMAZON_TASKS = common.DATASET_GROUPS["amazon"]                                           


def report(model_name, algorithm):
    """读取刚落盘的 sum.json，汇报三任务的 ROUGE-1 / METEOR / BERTScore-F1 + 宏平均。"""
    pats = sorted(glob.glob(os.path.join(
        common.RESULTS_DIR, "%s_%s_*_sum.json" % (model_name, algorithm))))
    if not pats:
        print("  [report] 找不到 %s 的 %s 结果" % (model_name, algorithm), flush=True)
        return
    tasks = json.load(open(pats[-1], encoding="utf-8")).get("tasks", {})
    print("\n===== %s | Amazon OOD 文本质量 =====" % model_name, flush=True)
    print("%-14s %8s %8s %10s" % ("task", "ROUGE-1", "METEOR", "BERTScore"), flush=True)
    acc = {"rouge-1": [], "meteor": [], "bertscore_f1": []}
    for t in AMAZON_TASKS:
        m = tasks.get(t)
        if not m:
            print("%-14s %8s" % (t, "N/A"), flush=True)
            continue
        for k in acc:
            acc[k].append(m[k])
        print("%-14s %8.4f %8.4f %10.4f"
              % (t, m["rouge-1"], m["meteor"], m["bertscore_f1"]), flush=True)
    if acc["rouge-1"]:
        macro = {k: sum(v) / len(v) for k, v in acc.items()}
        print("%-14s %8.4f %8.4f %10.4f"
              % ("macro-avg", macro["rouge-1"], macro["meteor"],
                 macro["bertscore_f1"]), flush=True)


def main():
    if common.EVAL_TASKS != AMAZON_TASKS:
        print("警告：EVAL_TASKS 解析为 %s（预期 amazon 三任务）" % common.EVAL_TASKS,
              flush=True)
    base.main()                                                   
    for model_name in base.MODEL_NAMES:
        report(model_name, base.ALGORITHM)


if __name__ == "__main__":
    main()
