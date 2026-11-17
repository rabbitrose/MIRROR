"""不依赖 LLM 裁判的客观退化统计。

动机：现在"SFT 损害文本质量"这一结论完全依赖同族裁判（Qwen3-30B 评 Qwen3）。
本脚本只用已落盘的 detail.json 算客观量，给该结论一条独立证据链：
  len_tok      输出 token 数（用 Qwen3 tokenizer）
  len_ratio    输出长度 / gold 长度，接近 1 说明在机械对齐参考长度
  rep4         4-gram 重复率 = 1 - 去重4gram数/总4gram数，退化的标准信号
  distinct2    2-gram 多样性 = 去重2gram数/总2gram数
  ttr          type-token ratio
  empty        空输出占比
用法：python eval/analyze_degeneration.py [> TestResults/degeneration.txt]
"""
import glob
import json
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import eval_common as common

TASKS = {"abstract_generation": ("abs", "-long"),
         "news_headline": ("news", "-lamp-news"),
         "scholarly_title": ("title", "-lamp-title")}
WORD = re.compile(r"[A-Za-z0-9']+")
# (显示名, 检查点前缀, 结果文件里基座的名字)；Qwen2.5-3B 的结果文件用全名
BACKBONES = (("1.7B", "Qwen3-1.7B", "Qwen3-1.7B"),
             ("4B", "Qwen3-4B", "Qwen3-4B"),
             ("Qwen2.5-3B", "Qwen2.5-3B", "Qwen2.5-3B-Instruct"))


def ngrams(toks, n):
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def stats(rows, tokenizer):
    out = {k: 0.0 for k in ("len_tok", "len_ratio", "rep4", "distinct2", "ttr")}
    n = empty = 0
    for r in rows:
        text = (r.get("model_output") or "").strip()
        gold = (r.get("golden_truth") or "").strip()
        if not text:
            empty += 1
            continue
        toks = WORD.findall(text.lower())
        if len(toks) < 2:
            empty += 1
            continue
        n += 1
        out["len_tok"] += len(tokenizer(text, add_special_tokens=False)["input_ids"])
        g = len(tokenizer(gold, add_special_tokens=False)["input_ids"]) if gold else 0
        out["len_ratio"] += (len(toks) / max(1, len(WORD.findall(gold.lower())))) if gold else 0.0
        q4 = ngrams(toks, 4)
        out["rep4"] += (1 - len(set(q4)) / len(q4)) if q4 else 0.0
        b2 = ngrams(toks, 2)
        out["distinct2"] += (len(set(b2)) / len(b2)) if b2 else 0.0
        out["ttr"] += len(set(toks)) / len(toks)
    if not n:
        return None
    res = {k: v / n for k, v in out.items()}
    res["empty"] = empty / max(1, len(rows))
    res["n"] = n
    return res


def latest_detail(model, algo):
    pat = re.compile(r"^%s_%s_\d{8}_detail\.json$" % (re.escape(model), re.escape(algo)))
    fs = sorted(f for f in glob.glob("%s/%s_%s_*_detail.json"
                                     % (common.RESULTS_DIR, model, algo))
                if pat.match(os.path.basename(f)))
    return fs[-1] if fs else None


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(common.MODELS_DIR, "Qwen3-1.7B"))
    for task, (atag, suf) in TASKS.items():
        print("\n### %s" % task)
        for size, P, BR in BACKBONES:
            rows = [("Base", BR, "base_" + atag), ("GOLD", None, None)]
            for m in ("contextSFT", "PerCE", "NextQuill", "MIRROR", "MIRROR-F"):
                rows.append((m, "%s-%s%s" % (P, m, suf), "base"))
            print("  -- %s" % size)
            print("     %-12s %8s %9s %7s %9s %6s %6s"
                  % ("method", "len_tok", "len_ratio", "rep4", "distinct2", "ttr", "n"))
            gold_done = False
            for name, model, algo in rows:
                if name == "GOLD":
                    f = latest_detail(BR, "base_" + atag)
                    if not f or gold_done:
                        continue
                    det = json.load(open(f))["details"]
                    fake = [{"model_output": r.get("golden_truth"),
                             "golden_truth": r.get("golden_truth")} for r in det]
                    st = stats(fake, tok)
                    gold_done = True
                else:
                    f = latest_detail(model, algo)
                    st = stats(json.load(open(f))["details"], tok) if f else None
                if not st:
                    print("     %-12s 缺" % name)
                    continue
                print("     %-12s %8.1f %9.3f %7.4f %9.4f %6.3f %6d"
                      % (name, st["len_tok"], st["len_ratio"], st["rep4"],
                         st["distinct2"], st["ttr"], st["n"]))


if __name__ == "__main__":
    main()
