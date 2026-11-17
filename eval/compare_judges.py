"""对比 Qwen3-30B-A3B 与 gemma-2-9b-it 两个裁判：逐个评测集合给出各模型 G-Eval / ExPerT 均值、
排序一致性（Spearman），以及主表关键结论在 Gemma 下是否成立（MIRROR / MIRROR-F vs 最强模仿基线，配对 bootstrap）。
输出 TestResults/judge_compare_gemma9b_<日期>.md"""
import glob, json, os, random, re
from datetime import datetime
R = "/root/MIRROR/TestResults/"
def latest(tag):
    fs = [f for f in glob.glob(R + "quality_score_%s_2*_sum.json" % tag) if re.search(r"_\d{8}_sum\.json$", f)]
    return max(fs) if fs else None
def spearman(a, b):
    ks = list(a); n = len(ks)
    if n < 3: return None
    ra = {k: i for i, k in enumerate(sorted(ks, key=lambda k: a[k]))}
    rb = {k: i for i, k in enumerate(sorted(ks, key=lambda k: b[k]))}
    d = sum((ra[k] - rb[k]) ** 2 for k in ks)
    return 1 - 6 * d / (n * (n * n - 1))
def boot(d, B=2000):
    random.seed(0); n = len(d)
    bs = sorted(sum(d[random.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(d) / n, bs[int(.025 * B)], bs[int(.975 * B) - 1]
lines = ["# Qwen3-30B-A3B vs gemma-2-9b-it 裁判对比（%s）" % datetime.now().strftime("%F %T"), ""]
qtags = sorted(set(re.sub(r"^quality_score_|_\d{8}_sum\.json$", "", os.path.basename(f))
                   for f in glob.glob(R + "quality_score_*_sum.json")))
qtags = [t for t in qtags if not t.startswith("gemma")]
rhos = []
for t in qtags:
    q, g = latest(t), latest("gemma9b_" + t)
    if not q or not g: continue
    Q, G = json.load(open(q)), json.load(open(g))
    task = list(Q["per_task_mean_score"])[0]
    qm, gm = Q["per_task_mean_score"][task], G["per_task_mean_score"].get(task, {})
    common = [m for m in qm if m in gm]
    lines.append("## %s" % t)
    lines.append("| model | G-Eval Qwen | G-Eval Gemma | ExPerT style Qwen | ExPerT style Gemma |")
    lines.append("|---|---|---|---|---|")
    qe = Q.get("expert", {}).get("per_task", {}).get(task, {}); ge = G.get("expert", {}).get("per_task", {}).get(task, {})
    for m in common:
        lines.append("| %s | %.3f | %.3f | %s | %s |" % (m, qm[m]["mean_score"], gm[m]["mean_score"],
                     qe.get(m, {}).get("style", "-"), ge.get(m, {}).get("style", "-")))
    rho = spearman({m: qm[m]["mean_score"] for m in common}, {m: gm[m]["mean_score"] for m in common})
    if rho is not None: rhos.append(rho)
    lines.append("Spearman(G-Eval 排序) = %s" % ("%.3f" % rho if rho is not None else "n/a（模型数<3）"))
    if t.endswith("maingold"):
        ps = sorted(glob.glob(R + "quality_score_gemma9b_%s_2*_persample.json" % t))
        if ps:
            S = {}
            for r in json.load(open(ps[-1]))["geval"]:
                S.setdefault(r["model"], {})[r["sample_id"]] = r["overall"]
            pick = lambda key: [m for m in S if re.search(r"-%s(-long|-lamp-news|-lamp-title)$" % re.escape(key), m)]
            imit = [m for k in ["contextSFT", "PerCE", "NextQuill"] for m in pick(k)]
            if imit:
                best = max(imit, key=lambda m: sum(S[m].values()) / len(S[m]))
                for k in ["MIRROR", "MIRROR-F"]:
                    for m in pick(k):
                        ids = [i for i in S[m] if i in S[best]]
                        lines.append("- Gemma: %s − %s = %+.3f [%+.3f, %+.3f]" % ((m, best) + boot([S[m][i] - S[best][i] for i in ids])))
    lines.append("")
if rhos:
    lines.insert(2, "共 %d 个可比评测集合，G-Eval 排序 Spearman 均值 %.3f（最小 %.3f）" % (len(rhos), sum(rhos) / len(rhos), min(rhos)))
    lines.insert(3, "")
out = R + "judge_compare_gemma9b_%s.md" % datetime.now().strftime("%Y%m%d")
open(out, "w").write("\n".join(lines) + "\n")
print(out)
