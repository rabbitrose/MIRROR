"""给基座与 GOLD 的 detail 文件建任务标识别名，让它们能和训练模型在同一次裁判评测里比较。

背景：eval_quality.py 对一次运行内所有模型使用同一个 ALGO。训练模型的 detail 算法名是
base（任务信息在模型名里），而基座和 GOLD 的算法名带任务标识（base_abs / base_news /
base_title），两者无法共存于一次运行。这里用软链接造出"模型名带任务标识、算法名为 base"
的别名：
    Qwen3-1.7B_base_abs_<date>_detail.json -> Qwen3-1.7B-BASE-abs_base_<date>_detail.json
    GOLD_base_abs_<date>_detail.json       -> GOLD-abs_base_<date>_detail.json
"""
import glob
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import eval_common as common

TAGS = ("abs", "news", "title")
BASES = ("Qwen3-1.7B", "Qwen3-4B", "Qwen2.5-3B-Instruct", "Qwen2.5-3B")
ALIAS_PREFIX = {"Qwen2.5-3B-Instruct": "Qwen2.5-3B"}


def latest(model, algo, kind):
    pat = re.compile(r"^%s_%s_\d{8}_%s\.json$" % (re.escape(model), re.escape(algo), kind))
    fs = sorted(f for f in glob.glob("%s/%s_%s_*_%s.json" % (common.RESULTS_DIR, model, algo, kind))
                if pat.match(os.path.basename(f)))
    return fs[-1] if fs else None


def link(src, alias_model, kind):
    date = re.search(r"_(\d{8})_", os.path.basename(src)).group(1)
    dst = os.path.join(common.RESULTS_DIR, "%s_base_%s_%s.json" % (alias_model, date, kind))
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)
    return os.path.basename(dst)


def main():
    n = 0
    for tag in TAGS:
        for kind in ("detail", "sum"):
            src = latest("GOLD", "base_" + tag, kind)
            if src:
                print("GOLD-%s <- %s" % (tag, os.path.basename(src)),
                      "->", link(src, "GOLD-" + tag, kind)); n += 1
            for b in BASES:
                src = latest(b, "base_" + tag, kind)
                if src:
                    ab = ALIAS_PREFIX.get(b, b)
                    print("%s-BASE-%s <- %s" % (ab, tag, os.path.basename(src)),
                          "->", link(src, "%s-BASE-%s" % (ab, tag), kind)); n += 1
    print("共建立 %d 个别名" % n)


if __name__ == "__main__":
    main()
