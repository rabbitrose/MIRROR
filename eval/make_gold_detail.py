"""把 gold 参考答案本身包装成一份 detail.json，让裁判能给 gold 打分。

用途：按参考答案质量分层，验证"模仿低质量参考会更伤"这个机制。
产出 TestResults/GOLD_base_<任务标识>_<日期>_detail.json，模型名 GOLD。
"""
import glob
import json
import os
import re
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import eval_common as common

TAGS = {"abstract_generation": "abs", "news_headline": "news", "scholarly_title": "title"}


def latest(model, algo):
    pat = re.compile(r"^%s_%s_\d{8}_detail\.json$" % (re.escape(model), re.escape(algo)))
    fs = sorted(f for f in glob.glob("%s/%s_%s_*_detail.json" % (common.RESULTS_DIR, model, algo))
                if pat.match(os.path.basename(f)))
    return fs[-1] if fs else None


def main():
    date = datetime.now().strftime("%Y%m%d")
    for task, tag in TAGS.items():
        src = latest("Qwen3-1.7B", "base_" + tag)
        if not src:
            print("跳过 %s：找不到基座 detail" % task); continue
        det = json.load(open(src))["details"]
        rows = []
        for r in det:
            g = (r.get("golden_truth") or "").strip()
            if not g:
                continue
            rows.append({"sample_id": r["sample_id"], "task": r["task"],
                         "input": r["input"], "model_output": g, "golden_truth": g})
        out = os.path.join(common.RESULTS_DIR, "GOLD_base_%s_%s_detail.json" % (tag, date))
        json.dump({"benchmark": "gold reference as candidate", "model": "GOLD",
                   "algorithm": "base_" + tag, "created_date": date,
                   "num_samples": len(rows), "details": rows},
                  open(out, "w", encoding="utf-8"), ensure_ascii=False)
        print("%s -> %s（%d 条）" % (task, os.path.basename(out), len(rows)))


if __name__ == "__main__":
    main()
