"""基于规则的信息完整性（论文附录 Rule-Based Information Completeness Analysis）。

纯 CPU，直接读 TestResults 里已有的 *_detail.json，不重新生成。定义逐条照附录：
  W(x)   正则 [A-Za-z0-9][A-Za-z0-9'\\-]* 抽词
  core   只留字母数字；Informative = 含数字 / 第 2 位起有大写 / 长度≥3 且首字母大写且不在停用词
  S_i    gold 中 Informative 且 lower(core) 作为完整词出现在学生输入 c_i 里的词（集合）
  O_im   模型输出的 lower(w) 集合
  R_im   |S_i ∩ O_im| / |S_i|，|S_i|=0 的样本剔除；IC = 逐样本 recall 的算术平均
c_i 统一取同任务 Base（full user context）评测时的 model_input，保证所有方法分母相同；
检索类方法的实际输入不同，但目标集合仍按同一份 c_i 定义。

样本集合与 Base 不同的方法（如只在用户子集上评测的 OPPU），额外给出 Base 在同一子集上
的分数，便于同口径比较。

用法：python eval/eval_completeness.py
  输出 TestResults/completeness_<日期>.json 与 completeness_<日期>.txt
"""
import glob
import json
import os
import re
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(BASE_DIR, '..', 'TestResults'))

# 与 train/train_mirrorf.py 的 SFT_STOP 逐字一致（附录：训练期 focal selector 用的停用词表）
STOP = frozenset(
    'the a an this that these those it its he she they we you i in on at of to for '
    'and or but if then so as is are was were be been being have has had do does did '
    'will would can could may might must should i\'m it\'s there here what when where '
    'who why how not no yes with without from by about into over under after before '
    'my your his her our their'.split())
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")

FAMILIES = [('Qwen3-1.7B', 'Qwen3-1.7B'), ('Qwen3-4B', 'Qwen3-4B'),
            ('Qwen2.5-3B', 'Qwen2.5-3B-Instruct')]
TASKS = [('abstract_generation', 'abs', '-long'),
         ('news_headline', 'news', '-lamp-news'),
         ('scholarly_title', 'title', '-lamp-title')]
TRAINED = ['contextSFT', 'contextSFT-RQF', 'MIRROR-noref', 'MIRROR-shufref', 'PerCE', 'NextQuill', 'OPPU', 'MIRROR', 'MIRROR-F',
           'MIRROR-F-uniform', 'MIRROR-F-a0', 'MIRROR-F-a0.1', 'MIRROR-F-a0.3',
           'MIRROR-F-a0.7', 'MIRROR-F-a0.9', 'MIRROR-F-a1']
RETRIEVAL = ['rag2', 'rag5', 'lastK5', 'trse5']


def core(w):
    return ''.join(ch for ch in w if ch.isalnum())


def informative(w):
    c = core(w)
    if len(c) < 2:
        return False
    return (any(ch.isdigit() for ch in c) or any(ch.isupper() for ch in c[1:])
            or (len(c) >= 3 and c[0].isupper() and c.lower() not in STOP))


def target_set(gold, context):
    ctx = set(w.lower() for w in WORD.findall(context or ''))
    return set(core(w).lower() for w in WORD.findall(gold or '')
               if informative(w) and core(w).lower() in ctx)


def latest(stem):
    """stem = <模型名>_<算法>；取日期最新的 detail 文件。"""
    hits = [f for f in glob.glob(os.path.join(RESULTS, stem + '_2*_detail.json'))
            if re.fullmatch(re.escape(stem) + r'_\d{8}_detail\.json', os.path.basename(f))]
    return max(hits, key=lambda f: os.path.basename(f)[-20:-12]) if hits else None


def load(path):
    with open(path, encoding='utf-8') as f:
        return {d['sample_id']: d for d in json.load(f)['details']}


def ic(details, targets, ids=None):
    vals = []
    for sid, s in targets.items():
        if ids is not None and sid not in ids:
            continue
        d = details.get(sid)
        if d is None:
            continue
        out = set(w.lower() for w in WORD.findall(d.get('model_output') or ''))
        vals.append(len(s & out) / len(s))
    return (round(sum(vals) / len(vals), 4) if vals else None), len(vals)
def main():
    report, lines = {}, []
    for fam, base_name in FAMILIES:
        for task, atag, suf in TASKS:
            base_path = latest('%s_base_%s' % (base_name, atag))
            if not base_path:
                continue
            base = load(base_path)
            targets = {}
            for sid, d in base.items():
                s = target_set(d.get('golden_truth'), d.get('model_input'))
                if s:
                    targets[sid] = s
            entries = {}
            base_ic, n = ic(base, targets)
            entries['Base'] = {'ic': base_ic, 'n': n, 'file': os.path.basename(base_path)}
            cands = [('%s%s' % (m, ''), '%s-%s%s_base' % (fam, m, suf)) for m in TRAINED]
            cands += [(r, '%s_%s_%s' % (base_name, r, atag)) for r in RETRIEVAL]
            for label, stem in cands:
                path = latest(stem)
                if not path:
                    continue
                det = load(path)
                val, n = ic(det, targets)
                e = {'ic': val, 'n': n, 'file': os.path.basename(path)}
                if set(det) != set(base):
                    e['base_on_same_subset'] = ic(base, targets, set(det))[0]
                entries[label] = e
            report['%s/%s' % (fam, task)] = {
                'valid_examples': len(targets), 'total_examples': len(base),
                'mean_target_size': round(sum(map(len, targets.values()))
                                          / max(len(targets), 1), 2),
                'methods': entries}
            lines.append('## %s  %s  (有效样本 %d/%d)' % (fam, task, len(targets), len(base)))
            for label, e in entries.items():
                extra = ('   [子集上 Base=%.4f]' % e['base_on_same_subset']
                         if e.get('base_on_same_subset') is not None else '')
                lines.append('   %-18s IC=%s  n=%d%s' % (label, e['ic'], e['n'], extra))
    date = datetime.now().strftime('%Y%m%d')
    meta = {'metric': 'rule-based information completeness (appendix), per-example recall '
                      'of informative & grounded gold words, macro over examples',
            'context_source': 'Base full-context model_input of the same task',
            'created_date': date, 'results': report}
    with open(os.path.join(RESULTS, 'completeness_%s.json' % date), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RESULTS, 'completeness_%s.txt' % date), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
