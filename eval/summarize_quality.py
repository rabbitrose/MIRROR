"""汇总内容质量评测：G-Eval 四维 + ExPerT（content P/R/F1 + style + C&S），按基座分块。

读的是 run_judge.sh 的 maingold 产物（GOLD + Base + 5 个训练方法，同一裁判同一批输出），
每个方法再给出相对同规模 Base zero-shot 的差值，这样"训练到底让质量掉了多少"一眼可见。

用法：python eval/summarize_quality.py  [> TestResults/quality_table.md]
"""
import glob
import json
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import eval_common as common

TASKS = {"abstract_generation": ("-long", "abs"),
         "news_headline": ("-lamp-news", "news"),
         "scholarly_title": ("-lamp-title", "title")}
METHODS = ("contextSFT", "PerCE", "NextQuill", "MIRROR", "MIRROR-F")
BACKBONES = ("Qwen3-1.7B", "Qwen3-4B", "Qwen2.5-3B")
# 裁判结果文件名用的是 run_judge.sh 的"规模"字段，不是检查点短名
JUDGE_SIZE = {"Qwen3-1.7B": "1.7B", "Qwen3-4B": "4B"}
GEVAL = ("coherence", "consistency", "fluency", "relevance", "mean_score")
EXP = ("content_precision", "content_recall", "content_f1", "style", "expert_overall")


def latest(tag):
    fs = sorted(glob.glob(os.path.join(common.RESULTS_DIR,
                                       "quality_score_%s_*_sum.json" % tag)))
    return json.load(open(fs[-1], encoding="utf-8")) if fs else None


def main():
    for b in BACKBONES:
        print("\n## %s" % b)
        for task, (suf, tag) in TASKS.items():
            d = latest("%s_base_%s_maingold" % (task, JUDGE_SIZE.get(b, b)))
            if d is None:
                print("\n### %s  —— 缺 maingold 结果" % task)
                continue
            q = d["per_task_mean_score"][task]
            e = d["expert"]["per_task"][task]
            names = [("GOLD", "GOLD-%s" % tag), ("Base", "%s-BASE-%s" % (b, tag))]
            names += [(m, "%s-%s%s" % (b, m, suf)) for m in METHODS]
            base_key = "%s-BASE-%s" % (b, tag)
            print("\n### %s（裁判 %s，每模型 n=%d，解析 %d/%d）"
                  % (task, d["judge_model"], d["samples_per_task"], d["parsed"], d["unparsed"]))
            print("  %-12s %5s %5s %5s %5s %7s   %5s %5s %5s %5s %7s   %7s"
                  % ("model", "Coh", "Con", "Flu", "Rel", "G-Eval",
                     "P", "R", "F1", "Style", "C&S", "dG-Eval"))
            for disp, key in names:
                if key not in q:
                    print("  %-12s 缺" % disp)
                    continue
                g = [q[key][k] for k in GEVAL]
                x = [e[key][k] for k in EXP]
                dg = ("" if key == base_key or base_key not in q
                      else "%+.3f" % (q[key]["mean_score"] - q[base_key]["mean_score"]))
                print("  %-12s %5.2f %5.2f %5.2f %5.2f %7.3f   %5.2f %5.2f %5.2f %5.2f %7.3f   %7s"
                      % (disp, g[0], g[1], g[2], g[3], g[4], x[0], x[1], x[2], x[3], x[4], dg))


if __name__ == "__main__":
    main()
