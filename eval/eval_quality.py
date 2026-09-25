"""内容质量 LLM-as-judge 评测（reference-free，G-Eval 协议）。

抛开与 gold 的匹配，用强裁判（默认 Qwen3-30B-A3B，vLLM）对**单份输出**按 G-Eval
（Liu et al., EMNLP 2023）+ SummEval 四维（Fabbri et al., TACL 2021）打分：
Coherence / Consistency / Fluency / Relevance 各 1-5，overall = 四维均值，再按
方法 × 任务 求平均。裁判只看题目 + 一份候选输出，不看 gold，不做成对比较。

各任务只给一句"候选是什么体裁"的中性说明（TASK_FORMS），四个评分维度全部共用，
维度定义直接沿用上述两篇论文，不对任何方法做加权。

正式运行方式：python eval_quality.py
  QUALITY_MODELS     逗号分隔模型名
  QUALITY_ALGO       读哪档 detail（默认 base）
  QUALITY_TASKS      评测任务，同 resolve_tasks（默认 longlamp）
  QUALITY_SAMPLES    每任务抽多少条（默认 100，取各模型 sample_id 交集保证同题可比）
  QUALITY_TAG        结果文件名标签，避免同日多档互相覆盖
  JUDGE_MODEL/JUDGE_TP/JUDGE_GPU_MEM_UTIL/JUDGE_MAX_MODEL_LEN/EVAL_GPU_IDS
结果写入 TestResults/quality_score_<tag>_<日期>_sum.json。
"""
import glob
import json
import os
import random
import re
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import eval_lamp_sparse as common                      

MODELS = [n.strip() for n in os.environ.get(
    'QUALITY_MODELS',
    'Qwen3-1.7B,Qwen3-1.7B-contextSFT,Qwen3-1.7B-NextQuill,Qwen3-1.7B-psopd'
).split(',') if n.strip()]
ALGO = os.environ.get('QUALITY_ALGO', 'base')
TASKS = common.resolve_tasks(os.environ.get('QUALITY_TASKS'), 'longlamp')
SAMPLES_PER_TASK = int(os.environ.get('QUALITY_SAMPLES', '100'))
SAMPLE_SEED = int(os.environ.get('QUALITY_SEED', '20260917'))

JUDGE_MODEL = os.environ.get('JUDGE_MODEL', 'Qwen3-30B-A3B')
JUDGE_TP = int(os.environ.get('JUDGE_TP', '8'))
JUDGE_GPU_MEM_UTIL = float(os.environ.get('JUDGE_GPU_MEM_UTIL', '0.85'))
JUDGE_MAX_MODEL_LEN = int(os.environ.get('JUDGE_MAX_MODEL_LEN', '8192'))
JUDGE_MAX_NEW = int(os.environ.get('JUDGE_MAX_NEW', '512'))
JUDGE_ENFORCE_EAGER = os.environ.get('JUDGE_ENFORCE_EAGER', '1') == '1'
CANDIDATE_CHAR_CAP = int(os.environ.get('QUALITY_CAND_CHARS', '4000'))

                                                              
                                                    
RUN_EXPERT = os.environ.get('QUALITY_EXPERT', '1') == '1'
                                       
EXPERT_TEXT_CAP = int(os.environ.get('EXPERT_TEXT_CHARS', '3500'))
EXPERT_MAX_ASPECTS = int(os.environ.get('EXPERT_MAX_ASPECTS', '40'))
               
                                                                             
                                                                    
                                                                    
                                
GEVAL_DIMENSIONS = (
    '- Coherence (1-5): the collective quality of the response. It should be '
    'well-structured and well-organized, not just a heap of related sentences.\n'
    '- Consistency (1-5): factual alignment between the response and the given source '
    'input. A consistent response contains only statements that are entailed by or '
    'faithful to the source; penalize hallucinated or unsupported facts.\n'
    '- Fluency (1-5): the quality of the response in terms of grammar, spelling, word '
    'choice, punctuation, and overall readability.\n'
    '- Relevance (1-5): selection of the important, on-topic content from the source. '
    'The response should include the salient, pertinent information and exclude '
    'redundant or off-topic content.')

DIM_KEYS = ['coherence', 'consistency', 'fluency', 'relevance']

                                              
TASK_FORMS = {
    'abstract_generation':
        'The CANDIDATE is a research-paper ABSTRACT written for the given paper TITLE.',
    'product_review':
        'The CANDIDATE is a PRODUCT REVIEW written in response to the given request.',
    'topic_writing':
        'The CANDIDATE is a REDDIT-STYLE POST BODY written for the given title/prompt.',
    'news_headline':
        'The CANDIDATE is a NEWS HEADLINE written for the given article; it is expected '
        'to be a short headline rather than a full text.',
    'scholarly_title':
        'The CANDIDATE is the TITLE of an academic paper written for the given abstract; '
        'it is expected to be a short title rather than a full text.',
}
RUBRICS = TASK_FORMS                                          


def find_detail(model):
    """只匹配 <model>_<algo>_<8位日期>_detail.json，避开 base_think / base_gold 等变体。"""
    pat = re.compile(r'^%s_%s_\d{8}_detail\.json$'
                     % (re.escape(model), re.escape(ALGO)))
    cands = sorted(f for f in glob.glob(os.path.join(
        common.RESULTS_DIR, '%s_%s_*_detail.json' % (model, ALGO)))
        if pat.match(os.path.basename(f)))
    return cands[-1] if cands else None


def load_outputs(model):
    path = find_detail(model)
    if not path:
        raise RuntimeError('找不到 %s 的 %s 档 detail 文件' % (model, ALGO))
    out = {}
    for r in json.load(open(path, encoding='utf-8'))['details']:
        out.setdefault(r['task'], {})[r['sample_id']] = {
            'input': r.get('input', ''),
            'output': common.postprocess(r.get('model_output') or ''),
            'gold': (r.get('golden_truth') or '').strip(),
        }
    print('  %s <- %s' % (model, os.path.basename(path)), flush=True)
    return out


def resolve_judge_path():
    for root in (common.MODELS_DIR, common.TRAINED_DIR):
        p = os.path.join(root, JUDGE_MODEL)
        if os.path.isdir(p):
            return p
    raise RuntimeError('裁判模型不存在：%s' % JUDGE_MODEL)


def build_prompt(tokenizer, task, question, candidate):
    """G-Eval 打分 prompt：给体裁说明 + 四维定义 + 评分步骤，要求四行 1-5 分。"""
    user = ('You will be given one CANDIDATE text written for a task. Your job is to rate '
            'the candidate on four metrics. Read these instructions carefully and keep them '
            'open while reviewing.\n\n'
            '%s\n\n'
            'Evaluation Criteria:\n%s\n\n'
            'Evaluation Steps:\n'
            '1. Read the SOURCE INPUT (the task prompt / article / abstract) carefully.\n'
            '2. Read the CANDIDATE and compare it against the source input.\n'
            '3. Assign an integer score from 1 (worst) to 5 (best) for each of the four '
            'metrics, strictly following the criteria above.\n\n'
            'SOURCE INPUT:\n"""\n%s\n"""\n\n'
            'CANDIDATE:\n"""\n%s\n"""\n\n'
            'Do not use any external reference answer. Give a one or two sentence '
            'justification, then end with exactly four lines:\n'
            'COHERENCE: <1-5>\nCONSISTENCY: <1-5>\nFLUENCY: <1-5>\nRELEVANCE: <1-5>'
            % (TASK_FORMS[task], GEVAL_DIMENSIONS,
               (question or '').strip()[:CANDIDATE_CHAR_CAP],
               (candidate or '').strip()[:CANDIDATE_CHAR_CAP]))
    messages = [{'role': 'system',
                 'content': 'You are a meticulous, impartial NLG evaluation judge.'},
                {'role': 'user', 'content': user}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def parse_score(text):
    """解析 G-Eval 四维 1-5 分，返回 {dim: score, 'overall': mean}；缺任一维则 None。"""
    t = common.strip_thinking(text) if hasattr(common, 'strip_thinking') else text
    out = {}
    for dim in DIM_KEYS:
        m = re.findall(r'%s\s*[:：]\s*(\d+(?:\.\d+)?)' % dim, t, flags=re.IGNORECASE)
        if not m:
            return None
        try:
            out[dim] = max(1.0, min(5.0, float(m[-1])))
        except ValueError:
            return None
    out['overall'] = round(sum(out[d] for d in DIM_KEYS) / len(DIM_KEYS), 3)
    return out
            


def build_extract_prompt(tokenizer, text):
    """ExPerT 第一步：把一段文本拆成原子 aspect（content + style 两类）。"""
    user = ('Extract the aspects of the following TEXT.\n'
            'Return STRICT JSON with exactly two keys:\n'
            '  "content": list of short strings, each an atomic factual/content point '
            'the text actually makes;\n'
            '  "style": list of short strings, each a writing-style attribute '
            '(tone, structure, formality, voice, length habit).\n'
            'Return ONLY the JSON object, no prose.\n\n'
            'TEXT:\n"""\n%s\n"""' % (text or '').strip()[:EXPERT_TEXT_CAP])
    messages = [{'role': 'system',
                 'content': 'You extract atomic aspects from text as strict JSON.'},
                {'role': 'user', 'content': user}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def parse_aspects(text):
    """从裁判输出里抠出 {"content":[...],"style":[...]}，失败则返回空列表。"""
    t = common.strip_thinking(text) if hasattr(common, 'strip_thinking') else text
    start = t.find('{')
    end = t.rfind('}')
    if start < 0 or end <= start:
        return [], []
    try:
        obj = json.loads(t[start:end + 1])
    except (ValueError, TypeError):
        return [], []
    def norm(x):
        return ([str(a).strip() for a in x if str(a).strip()][:EXPERT_MAX_ASPECTS]
                if isinstance(x, list) else [])
    return norm(obj.get('content')), norm(obj.get('style'))


def build_match_prompt(tokenizer, ref_aspects, cand_aspects):
    """ExPerT 第二步：对齐 reference 与 candidate 的 aspect，给出 content 召回/精确 + style。"""
    rc, rs = ref_aspects
    cc, cs = cand_aspects
    user = ('Compare a CANDIDATE against a REFERENCE using their pre-extracted aspects.\n\n'
            'REFERENCE content aspects: %s\nREFERENCE style aspects: %s\n\n'
            'CANDIDATE content aspects: %s\nCANDIDATE style aspects: %s\n\n'
            'Assess alignment:\n'
            '- CONTENT_RECALL: fraction 0-1 of REFERENCE content aspects expressed in CANDIDATE;\n'
            '- CONTENT_PRECISION: fraction 0-1 of CANDIDATE content aspects supported by REFERENCE;\n'
            '- STYLE: integer 1-10 for how well CANDIDATE style matches REFERENCE style.\n'
            'End with exactly three lines:\n'
            'CONTENT_RECALL: <0-1>\nCONTENT_PRECISION: <0-1>\nSTYLE: <1-10>'
            % (json.dumps(rc, ensure_ascii=False), json.dumps(rs, ensure_ascii=False),
               json.dumps(cc, ensure_ascii=False), json.dumps(cs, ensure_ascii=False)))
    messages = [{'role': 'system',
                 'content': 'You are a meticulous aspect-alignment judge for personalized text.'},
                {'role': 'user', 'content': user}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def parse_match(text):
    """解析 CONTENT_RECALL / CONTENT_PRECISION / STYLE 三行，算 content F1 与综合分。"""
    t = common.strip_thinking(text) if hasattr(common, 'strip_thinking') else text
    def grab(key, lo, hi):
        m = re.findall(r'%s\s*[:：]\s*(\d+(?:\.\d+)?)' % key, t, flags=re.IGNORECASE)
        if not m:
            return None
        try:
            return max(lo, min(hi, float(m[-1])))
        except ValueError:
            return None
    r = grab('content_recall', 0.0, 1.0)
    p = grab('content_precision', 0.0, 1.0)
    s = grab('style', 1.0, 10.0)
    if r is None or p is None or s is None:
        return None
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    overall = round((f1 * 10.0 + s) / 2.0, 3)                                    
    return {'content_recall': round(r, 3), 'content_precision': round(p, 3),
            'content_f1': round(f1, 3), 'style': round(s, 3), 'expert_overall': overall}


def run_expert(llm, tokenizer, jobs, all_outputs, SamplingParams):
    """在同一个裁判 llm 上跑 ExPerT：抽取 ref/cand aspect -> 匹配打分 -> 按方法×任务聚合。"""
    params = SamplingParams(temperature=0.0, max_tokens=JUDGE_MAX_NEW)
                                      
    ref_keys = sorted({(j['task'], j['sid']) for j in jobs})
    ref_gold = {}
    for task, sid in ref_keys:
        gold = ''
        for m in MODELS:
            rec = all_outputs.get(m, {}).get(task, {}).get(sid)
            if rec and rec.get('gold'):
                gold = rec['gold']; break
        ref_gold[(task, sid)] = gold
    ref_keys = [k for k in ref_keys if ref_gold[k]]                     
    if not ref_keys:
        print('ExPerT：没有可用的 gold 参考，跳过', flush=True)
        return None, 0, 0
                                      
    ref_prompts = [build_extract_prompt(tokenizer, ref_gold[k]) for k in ref_keys]
    valid_jobs = [j for j in jobs if ref_gold.get((j['task'], j['sid']))]
    cand_prompts = [build_extract_prompt(tokenizer, j['cand']) for j in valid_jobs]
    print('ExPerT 抽取 aspect：%d 参考 + %d 候选' % (len(ref_prompts), len(cand_prompts)),
          flush=True)
    ex_out = llm.generate(ref_prompts + cand_prompts, params)
    n_ref = len(ref_prompts)
    ref_asp = {ref_keys[i]: parse_aspects(ex_out[i].outputs[0].text)
               for i in range(n_ref)}
    cand_asp = [parse_aspects(ex_out[n_ref + i].outputs[0].text)
                for i in range(len(valid_jobs))]
                    
    match_prompts = [build_match_prompt(tokenizer, ref_asp[(j['task'], j['sid'])],
                                        cand_asp[i]) for i, j in enumerate(valid_jobs)]
    print('ExPerT 匹配打分：%d 条' % len(match_prompts), flush=True)
    mt_out = llm.generate(match_prompts, params)
    agg, ok, bad = {}, 0, 0
    for j, o in zip(valid_jobs, mt_out):
        parsed = parse_match(o.outputs[0].text)
        if parsed is None:
            bad += 1; continue
        ok += 1
        agg.setdefault((j['model'], j['task']), []).append(parsed)
    return agg, ok, bad


def summarize_expert(agg):
    """把逐样本 ExPerT 明细聚合成 方法×任务 与 宏平均。"""
    keys = ['content_recall', 'content_precision', 'content_f1', 'style', 'expert_overall']
    per_task = {}
    for (m, t), rows in agg.items():
        d = {k: round(sum(r[k] for r in rows) / len(rows), 3) for k in keys}
        d['n'] = len(rows)
        per_task.setdefault(t, {})[m] = d
    overall = {}
    for m in MODELS:
        ms = [per_task[t][m] for t in per_task if m in per_task.get(t, {})]
        if ms:
            overall[m] = {k: round(sum(x[k] for x in ms) / len(ms), 3) for k in keys}
    return per_task, overall
            


def build_jobs(all_outputs):
    """每个 (task, model, sample) 一条打分任务；用各模型 sample_id 交集保证同题可比。"""
    rng = random.Random(SAMPLE_SEED)
    jobs = []
    for task in TASKS:
        if task not in RUBRICS:
            print('跳过 %s：无 rubric' % task, flush=True)
            continue
        present = [m for m in MODELS if task in all_outputs[m]]
        if not present:
            continue
        common_ids = set(all_outputs[present[0]][task])
        for m in present[1:]:
            common_ids &= set(all_outputs[m][task])
        ids = sorted(common_ids)
        if len(ids) > SAMPLES_PER_TASK:
            ids = rng.sample(ids, SAMPLES_PER_TASK)
        for m in present:
            for sid in ids:
                rec = all_outputs[m][task][sid]
                jobs.append({'task': task, 'model': m, 'sid': sid,
                             'q': rec['input'], 'cand': rec['output']})
    return jobs


def main():
    import time
    started = time.time()
    if os.environ.get('EVAL_GPU_IDS'):
        os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['EVAL_GPU_IDS']
    os.environ.setdefault('VLLM_LOGGING_LEVEL', 'WARNING')
    judge_path = resolve_judge_path()

    print('裁判=%s (TP=%d, eager=%s)  模型=%s  任务=%s  每任务样本=%d'
          % (JUDGE_MODEL, JUDGE_TP, JUDGE_ENFORCE_EAGER, MODELS, TASKS,
             SAMPLES_PER_TASK), flush=True)
    all_outputs = {m: load_outputs(m) for m in MODELS}
    jobs = build_jobs(all_outputs)
    if not jobs:
        raise RuntimeError('没有可打分的样本')
    print('共 %d 条打分（%d 模型 × %d 任务 × 样本）'
          % (len(jobs), len(MODELS), len(TASKS)), flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(judge_path)
    prompts = [build_prompt(tokenizer, j['task'], j['q'], j['cand']) for j in jobs]
    llm = LLM(model=judge_path, dtype='bfloat16', max_model_len=JUDGE_MAX_MODEL_LEN,
              gpu_memory_utilization=JUDGE_GPU_MEM_UTIL, tensor_parallel_size=JUDGE_TP,
              enforce_eager=JUDGE_ENFORCE_EAGER, trust_remote_code=True)
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=JUDGE_MAX_NEW))
    scores = [parse_score(o.outputs[0].text) for o in outs]

                                                         
    expert_per_task = expert_overall = None
    expert_ok = expert_bad = 0
    if RUN_EXPERT:
        e_agg, expert_ok, expert_bad = run_expert(llm, tokenizer, jobs, all_outputs,
                                                  SamplingParams)
        if e_agg:
            expert_per_task, expert_overall = summarize_expert(e_agg)

                                        
    agg = {}                                   
    parsed = failed = 0
    for job, s in zip(jobs, scores):
        if s is None:
            failed += 1
            continue
        parsed += 1
        agg.setdefault((job['model'], job['task']), []).append(s)

    def dim_means(rows):
        d = {k: round(sum(r[k] for r in rows) / len(rows), 3) for k in DIM_KEYS}
        d['mean_score'] = round(sum(r['overall'] for r in rows) / len(rows), 3)
        d['n'] = len(rows)
        return d
    per_task = {}
    for (m, t), rows in agg.items():
        per_task.setdefault(t, {})[m] = dim_means(rows)
    overall = {}
    for m in MODELS:
        task_means = [per_task[t][m]['mean_score'] for t in per_task
                      if m in per_task.get(t, {})]
        overall[m] = round(sum(task_means) / len(task_means), 3) if task_means else None

    result = {
        'benchmark': 'content-quality LLM-as-judge, G-Eval protocol (SummEval 4 dims, 1-5)',
        'protocol': ('G-Eval (Liu et al., EMNLP 2023) with SummEval dimensions '
                     '(Fabbri et al., TACL 2021): Coherence / Consistency / Fluency / '
                     'Relevance, each 1-5; overall = mean of the four; reference-free.'),
        'judge_model': JUDGE_MODEL,
        'compared_models': MODELS,
        'source_algorithm': ALGO,
        'samples_per_task': SAMPLES_PER_TASK,
        'dimensions': GEVAL_DIMENSIONS,
        'task_forms': {t: TASK_FORMS[t] for t in per_task},
        'per_task_mean_score': per_task,
        'overall_mean_score_macro': overall,
        'parsed': parsed, 'unparsed': failed,
        'expert': {
            'method': 'ExPerT-style reference-based aspect alignment (content P/R/F1 + '
                      'style 1-10), same judge; content_f1 scaled to 10 and averaged '
                      'with style -> expert_overall (1-10)',
            'judge_model': JUDGE_MODEL,
            'per_task': expert_per_task,
            'overall_macro': expert_overall,
            'parsed': expert_ok, 'unparsed': expert_bad,
        } if RUN_EXPERT else None,
        'created_date': datetime.now().strftime('%Y%m%d'),
        'elapsed': common.format_duration(time.time() - started),
    }
    tag = os.environ.get('QUALITY_TAG', '').strip()
    stem = 'quality_score_%s%s_sum.json' % (
        (tag + '_') if tag else '', datetime.now().strftime('%Y%m%d'))
    path = os.path.join(common.RESULTS_DIR, stem)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print('结果已写入：%s（quality 解析 %d/%d，ExPerT 解析 %d/%d）'
          % (path, parsed, failed, expert_ok, expert_bad), flush=True)
    for t in per_task:
        row = '  '.join('%s=%.2f' % (m, per_task[t][m]['mean_score'])
                        for m in MODELS if m in per_task[t])
        print('  [quality %s] %s' % (t, row), flush=True)
    print('  quality 总体(宏平均): ' + '  '.join(
        '%s=%s' % (m, overall[m]) for m in MODELS), flush=True)
    if expert_per_task:
        for t in expert_per_task:
            row = '  '.join('%s(F1=%.2f,style=%.1f,all=%.2f)'
                            % (m, expert_per_task[t][m]['content_f1'],
                               expert_per_task[t][m]['style'],
                               expert_per_task[t][m]['expert_overall'])
                            for m in MODELS if m in expert_per_task[t])
            print('  [ExPerT %s] %s' % (t, row), flush=True)
        print('  ExPerT 总体(宏平均): ' + '  '.join(
            '%s(F1=%.2f,style=%.1f,all=%.2f)'
            % (m, expert_overall[m]['content_f1'], expert_overall[m]['style'],
               expert_overall[m]['expert_overall'])
            for m in MODELS if m in expert_overall), flush=True)


if __name__ == '__main__':
    main()
