"""汇总 OOD 结果：同时报 OOD 绝对分、Base 的 OOD 分、相对 Base 的变化量，以及 ID 增益。

为什么不只报一个 degradation：论文图注写"相对 Base"、正文写"相对 in-distribution 表现"，
两种定义互相矛盾；而且单看"相对 Base 的偏差"会奖励"根本没学到东西"的模型（Reviewer 3 的 W3）。
所以这里把四件事并列报出来：
  OOD        各方法在 Amazon 三任务上的真实表现（宏平均）
  Base_OOD   同规模基座的 OOD 表现，作为参照线
  ΔOOD       OOD - Base_OOD，正数表示 OOD 上比基座更好，负数才是真正的退化
  ΔID        in-distribution 任务上相对基座的增益
这样"学到了多少"（ΔID）与"忘掉了多少"（ΔOOD）能同时看见。

命名上的两个坑：基座的 ID 评测算法名带任务标识（base_abs / base_news / base_title，因为
基座结果文件名不含任务），训练模型的算法名只有 base（任务信息在模型名里）；Qwen2.5-3B 的
结果文件用的是全名 Qwen2.5-3B-Instruct，而检查点短名是 Qwen2.5-3B。

用法：python eval/summarize_ood.py  [> TestResults/ood_summary.md]
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
AMAZON_TASKS = common.DATASET_GROUPS["amazon"]
# 任务 -> (检查点后缀, 基座算法名里的任务标识)
TASKS = {
    "abstract_generation": ("-long", "abs"),
    "news_headline": ("-lamp-news", "news"),
    "scholarly_title": ("-lamp-title", "title"),
}
METHODS = ("contextSFT", "PerCE", "NextQuill", "MIRROR", "MIRROR-F")
BACKBONES = ("Qwen3-1.7B", "Qwen3-4B", "Qwen2.5-3B")
# 结果文件里基座用的名字（检查点短名去掉了 -Instruct）
BASE_RESULT_NAME = {"Qwen2.5-3B": "Qwen2.5-3B-Instruct"}


def latest(model, algo):
    """取某模型某算法最新一份 sum.json 的 tasks 字段。"""
    pat = re.compile(r"^%s_%s_\d{8}_sum\.json$" % (re.escape(model), re.escape(algo)))
    files = sorted(f for f in glob.glob(os.path.join(common.RESULTS_DIR,
                                                     "%s_%s_*_sum.json" % (model, algo)))
                   if pat.match(os.path.basename(f)))
    if not files:
        return None
    return json.load(open(files[-1], encoding="utf-8")).get("tasks", {})


def macro(tasks, keys):
    """在给定任务上做宏平均；任务缺失则返回 None。"""
    if not tasks:
        return None
    vals = {m: [] for m in METRICS}
    for t in keys:
        if t not in tasks:
            return None
        for m in METRICS:
            vals[m].append(tasks[t][m])
    return {m: sum(v) / len(v) for m, v in vals.items()}


def cell(d):
    return "%.4f/%.4f/%.4f" % tuple(d[m] for m in METRICS)


def main():
    for b in BACKBONES:
        base_res = BASE_RESULT_NAME.get(b, b)
        base_ood = macro(latest(base_res, "base_ood"), AMAZON_TASKS)
        print("\n## %s" % b)
        if base_ood is None:
            print("  基座 OOD 结果缺失，跳过")
            continue
        print("  Base_OOD: R1=%.4f MET=%.4f BS=%.4f" % tuple(base_ood[m] for m in METRICS))
        for task, (suf, tag) in TASKS.items():
            base_id = macro(latest(base_res, "base_" + tag), [task])
            models = ["%s-%s%s" % (b, m, suf) for m in METHODS]
            if task == "abstract_generation":
                models.append("%s-MIRROR-F-uniform%s" % (b, suf))
            rows = []
            for model in models:
                ood = macro(latest(model, "base_ood"), AMAZON_TASKS)
                if ood is None:
                    continue
                idm = macro(latest(model, "base"), [task])
                dood = {m: ood[m] - base_ood[m] for m in METRICS}
                did = {m: idm[m] - base_id[m] for m in METRICS} if (idm and base_id) else None
                rows.append((model.replace(b, "").replace(suf, "").strip("-"),
                             cell(ood), cell(dood), cell(did) if did else "ID 未跑"))
            if not rows:
                continue
            print("\n  ### %s（ID 任务）" % task)
            if base_id is None:
                print("  注意：基座在该任务上的 ID 结果缺失，ΔID 无法计算")
            print("  %-20s %-24s %-24s %-24s" % ("method", "OOD (R1/MET/BS)",
                                                 "dOOD vs Base", "dID vs Base"))
            for r in rows:
                print("  %-20s %-24s %-24s %-24s" % r)


if __name__ == "__main__":
    main()
