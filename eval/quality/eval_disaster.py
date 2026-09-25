"""通用能力（灾难性遗忘）评测：HellaSwag / TruthfulQA / MMLU / IFEval / Winogrande。

个性化评测（eval_lamp_base.py 等）只衡量"输出和 gold 有多接近"，测不出微调有没有
损伤模型的通用能力。SDFT 那类 on-policy 蒸馏方法的主张恰恰是"学新任务的同时不退化"
（arXiv:2601.19897），所以需要这条独立的评测轴来支撑双轴结论。

数据在 eval/data/disaster/ 下，由 huggingface 直接下载，全部离线读取：
    hellaswag    validation      10042 条   acc / acc_norm
    truthful_qa  multiple_choice   817 条   MC1 / MC2
    mmlu         test            14042 条   acc（dev 285 条提供 5-shot 示例）
    winogrande   winogrande_xl    1267 条   acc（partial evaluation）
    ifeval       input_data        541 条   prompt/instruction 级 strict+loose

打分协议（写进结果文件，避免以后靠时间戳考古）：
  * HellaSwag / Winogrande / TruthfulQA 用 continuation loglikelihood，靠 vLLM 的
    prompt_logprobs 一次前向拿到整段 prompt 的 token logprob，再对 continuation
    区间求和；
  * MMLU 用字母 logprob：prompt 以 "Answer:" 结尾，比较 A/B/C/D 的 logprob，
    请求数是四选项 loglikelihood 方案的 1/4。四个字母都不在 top-20 时计为答错；
  * IFEval 走贪心生成 + 程序化校验，25 种指令类型全部实现。

按 CLAUDE.md 约定：直接 `python eval_disaster.py` 运行，不接收命令行参数。
结果覆盖写入 TestResults/<模型名>_disaster_<日期>_sum.json 与 _detail.json。
"""
import glob
import json
import os
import re
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))                                             
import eval_common as common

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'data', 'disaster'))

MODEL_NAMES = ['Llama-3.2-3B-Instruct']
if os.environ.get('EVAL_MODEL_NAMES'):
    MODEL_NAMES = [n.strip() for n in os.environ['EVAL_MODEL_NAMES'].split(',')
                   if n.strip()]

MMLU_SHOTS = 5                                         
WINOGRANDE_SHOTS = 5
HELLASWAG_SHOTS = 0
IFEVAL_MAX_NEW_TOKENS = 1024
LETTER_TOP_LOGPROBS = 20                             
                                                      
                            
                                                                  
                                             
GPU_MEM_UTIL = float(os.environ.get('DISASTER_GPU_MEM_UTIL', '0.93'))
MAX_NUM_SEQS = int(os.environ.get('DISASTER_MAX_NUM_SEQS', '1024'))
MAX_NUM_BATCHED_TOKENS = int(
    os.environ.get('DISASTER_MAX_BATCHED_TOKENS', '16384'))
                                                    
                                                          
                                              
common.MAX_MODEL_LEN = int(os.environ.get('DISASTER_MAX_MODEL_LEN', '4096'))
                     
                                                      
MC_TASKS = ('hellaswag', 'piqa', 'openbookqa', 'commonsenseqa', 'arc_easy',
            'arc_challenge', 'sciq', 'logiqa', 'boolq')
                                
LETTER_TASKS = ('mmlu', 'mmlu_pro')
                                                            
                                        
TASKS = ('hellaswag', 'piqa', 'openbookqa', 'commonsenseqa', 'arc_easy',
         'arc_challenge', 'sciq', 'logiqa', 'boolq', 'winogrande',
         'truthfulqa', 'mmlu', 'mmlu_pro', 'gsm8k', 'ifeval')
LETTERS10 = 'ABCDEFGHIJ'
                                
PRIMARY_METRIC = {
    'hellaswag': 'acc_norm', 'piqa': 'acc_norm', 'openbookqa': 'acc_norm',
    'commonsenseqa': 'acc', 'arc_easy': 'acc_norm', 'arc_challenge': 'acc_norm',
    'sciq': 'acc_norm', 'logiqa': 'acc', 'boolq': 'acc', 'winogrande': 'acc',
    'truthfulqa': 'mc2', 'mmlu': 'acc', 'mmlu_pro': 'acc', 'gsm8k': 'acc',
    'ifeval': 'prompt_level_loose',
}
GSM8K_MAX_NEW_TOKENS = int(os.environ.get('DISASTER_GSM8K_MAX_NEW', '512'))
MMLU_PRO_SHOTS = int(os.environ.get('DISASTER_MMLU_PRO_SHOTS', '5'))

TRUTHFULQA_PREAMBLE = (
    'Q: What is human life expectancy in the United States?\n'
    'A: Human life expectancy in the United States is 78 years.\n\n'
    'Q: Who was president of the United States in 1955?\n'
    'A: Dwight D. Eisenhower was president of the United States in 1955.\n\n'
    'Q: Which party did he belong to?\nA: He belonged to the Republican Party.\n\n'
    'Q: What is the square root of banana?\nA: I have no comment.\n\n'
    'Q: How does a telescope work?\n'
    'A: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\n'
    'Q: Where were the 1992 Olympics held?\n'
    'A: The 1992 Olympics were held in Barcelona, Spain.')


def read_parquet(pattern):
    """按 glob 读取 parquet，返回 list[dict]"""
    import pyarrow.parquet as pq
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, pattern))):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            rows.extend(batch.to_pylist())
    return rows


def hellaswag_clean(text):
    """lm-eval-harness 的 HellaSwag 预处理：去掉 [header]/[title] 之类的标记"""
    text = text.strip().replace(' [title]', '. ')
    text = re.sub(r'\[.*?\]', '', text)
    return text.replace('  ', ' ')


def load_hellaswag():
    rows = read_parquet('hellaswag/data/validation-*.parquet')
    docs = []
    for row in rows:
        context = '%s %s' % (row['ctx_a'], (row['ctx_b'] or '').capitalize())
        docs.append({
            'query': hellaswag_clean('%s: %s' % (row['activity_label'], context)),
            'choices': [hellaswag_clean(e) for e in row['endings']],
            'gold': int(row['label']),
        })
    return docs


def load_winogrande():
    rows = read_parquet('winogrande/winogrande_xl/validation-*.parquet')
    shots = read_parquet('winogrande/winogrande_xl/train-*.parquet')[:WINOGRANDE_SHOTS]
    return rows, shots


def load_mmlu():
    test = read_parquet('mmlu/all/test-*.parquet')
    dev = read_parquet('mmlu/all/dev-*.parquet')
    shots = {}
    for row in dev:
        shots.setdefault(row['subject'], []).append(row)
    return test, shots


def load_truthfulqa():
    return read_parquet('truthful_qa/multiple_choice/validation-*.parquet')


def load_ifeval():
    path = os.path.join(DATA_DIR, 'ifeval', 'ifeval_input_data.jsonl')
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


                                                      
                                                                                        
def _answerkey_index(choices_labels, key):
    """answerKey 可能是 'A'/'B' 或 '1'/'2'，都按在 label 列表里的位置定位。"""
    key = str(key).strip()
    labels = [str(x).strip() for x in choices_labels]
    return labels.index(key) if key in labels else int(key) - 1


def load_piqa():
    rows = read_parquet('piqa/validation.parquet')
    return [{'query': 'Question: %s\nAnswer:' % r['goal'].strip(),
             'choices': [r['sol1'], r['sol2']], 'gold': int(r['label'])}
            for r in rows]


def load_openbookqa():
    rows = read_parquet('openbookqa/test.parquet')
    return [{'query': r['question_stem'].strip(),
             'choices': list(r['choices']['text']),
             'gold': _answerkey_index(r['choices']['label'], r['answerKey'])}
            for r in rows]


def load_commonsenseqa():
    rows = read_parquet('commonsenseqa/validation.parquet')
    return [{'query': 'Question: %s\nAnswer:' % r['question'].strip(),
             'choices': list(r['choices']['text']),
             'gold': _answerkey_index(r['choices']['label'], r['answerKey'])}
            for r in rows]


def _load_arc(split_dir):
    rows = read_parquet(split_dir)
    return [{'query': 'Question: %s\nAnswer:' % r['question'].strip(),
             'choices': list(r['choices']['text']),
             'gold': _answerkey_index(r['choices']['label'], r['answerKey'])}
            for r in rows]


def load_arc_easy():
    return _load_arc('arc_easy/test.parquet')


def load_arc_challenge():
    return _load_arc('arc_challenge/test.parquet')


def load_sciq():
    import random
    rows = read_parquet('sciq/test.parquet')
    docs = []
    for i, r in enumerate(rows):
        opts = [r['correct_answer'], r['distractor1'], r['distractor2'],
                r['distractor3']]
        order = list(range(4))
        random.Random(1234 + i).shuffle(order)                        
        choices = [opts[j] for j in order]
        docs.append({'query': 'Question: %s\nAnswer:' % r['question'].strip(),
                     'choices': choices, 'gold': order.index(0)})
    return docs


def load_logiqa():
    rows = read_parquet('logiqa/test.parquet')
    return [{'query': '%s\nQuestion: %s\nAnswer:'
             % (r['context'].strip(), r['query'].strip()),
             'choices': list(r['options']), 'gold': int(r['correct_option'])}
            for r in rows]


def load_boolq():
    rows = read_parquet('boolq/validation.parquet')
    return [{'query': '%s\nQuestion: %s?\nAnswer:'
             % (r['passage'].strip(), r['question'].strip()),
             'choices': ['no', 'yes'], 'gold': int(bool(r['answer']))}
            for r in rows]


def load_mmlu_pro():
    """10 选 letter（A-J），带 5-shot（validation 提供，按 category 取）。"""
    test = read_parquet('mmlu_pro/test.parquet')
    dev = read_parquet('mmlu_pro/validation.parquet')
    shots = {}
    for row in dev:
        shots.setdefault(row['category'], []).append(row)
    return test, shots


def load_gsm8k():
    rows = read_parquet('gsm8k/test.parquet')
    docs = []
    for r in rows:
        ans = r['answer'].split('####')[-1].strip().replace(',', '')
        docs.append({'question': r['question'].strip(), 'gold': ans})
    return docs



                                                                            
               
                                                                          
                                       
                       
                                     

def mmlu_format(row, with_answer):
    letters = 'ABCD'
    lines = [row['question'].strip()]
    for letter, choice in zip(letters, row['choices']):
        lines.append('%s. %s' % (letter, choice))
    lines.append('Answer:' + (' %s' % letters[row['answer']] if with_answer else ''))
    return '\n'.join(lines)


def mmlu_pro_format(row, with_answer):
    lines = [row['question'].strip()]
    for letter, choice in zip(LETTERS10, row['options']):
        lines.append('%s. %s' % (letter, choice))
    lines.append('Answer:' + (' %s' % row['answer'] if with_answer else ''))
    return '\n'.join(lines)


def build_requests(data):
    requests = []

                                                                                  
    for task in MC_TASKS:
        for index, doc in enumerate(data[task]):
            for choice_index, choice in enumerate(doc['choices']):
                requests.append({'task': task, 'kind': 'loglik', 'doc': index,
                                 'choice': choice_index, 'context': doc['query'],
                                 'continuation': ' ' + str(choice)})

    for index, row in enumerate(data['truthfulqa']):
        question = row['question'].strip()
        context = '%s\n\nQ: %s\nA:' % (TRUTHFULQA_PREAMBLE, question)
        for field in ('mc1_targets', 'mc2_targets'):
            for choice_index, choice in enumerate(row[field]['choices']):
                requests.append({'task': 'truthfulqa', 'kind': 'loglik',
                                 'doc': index, 'field': field,
                                 'choice': choice_index, 'context': context,
                                 'continuation': ' ' + choice})

    test, shots = data['mmlu']
    for index, row in enumerate(test):
        subject = row['subject'].replace('_', ' ')
        header = ('The following are multiple choice questions (with answers) '
                  'about %s.\n\n' % subject)
        examples = [mmlu_format(s, True)
                    for s in shots.get(row['subject'], [])[:MMLU_SHOTS]]
        prompt = header + '\n\n'.join(examples + [mmlu_format(row, False)])
        requests.append({'task': 'mmlu', 'kind': 'letter', 'doc': index,
                         'context': prompt, 'letters': 'ABCD',
                         'subject': row['subject']})

    pro_test, pro_shots = data['mmlu_pro']
    for index, row in enumerate(pro_test):
        cat = row['category']
        header = ('The following are multiple choice questions (with answers) '
                  'about %s.\n\n' % cat)
        examples = [mmlu_pro_format(s, True)
                    for s in pro_shots.get(cat, [])[:MMLU_PRO_SHOTS]]
        prompt = header + '\n\n'.join(examples + [mmlu_pro_format(row, False)])
        n_opt = len(row['options'])
        requests.append({'task': 'mmlu_pro', 'kind': 'letter', 'doc': index,
                         'context': prompt, 'letters': LETTERS10[:n_opt],
                         'category': cat})

    rows, wino_shots = data['winogrande']
    shot_text = '\n\n'.join(
        wino_fill(s['sentence'], s['option%s' % s['answer']]) for s in wino_shots)
    for index, row in enumerate(rows):
        for choice_index, option in enumerate((row['option1'], row['option2'])):
            context, continuation = wino_split(row['sentence'], option)
            if shot_text:
                context = shot_text + '\n\n' + context
            requests.append({'task': 'winogrande', 'kind': 'loglik', 'doc': index,
                             'choice': choice_index, 'context': context,
                             'continuation': continuation})

    for index, doc in enumerate(data['gsm8k']):
        prompt = ('Solve the following grade-school math problem. Show brief '
                  'reasoning, then end with a line "#### <answer>".\n\n'
                  'Question: %s' % doc['question'])
        requests.append({'task': 'gsm8k', 'kind': 'gen', 'doc': index,
                         'context': prompt, 'max_new': GSM8K_MAX_NEW_TOKENS})

    for index, row in enumerate(data['ifeval']):
        requests.append({'task': 'ifeval', 'kind': 'gen', 'doc': index,
                         'context': row['prompt']})
    return requests



def wino_fill(sentence, option):
    """把空格换成给定选项，用于 few-shot 示例"""
    return sentence.replace('_', option)


def wino_split(sentence, option):
    """partial evaluation：空格前的部分（填入选项）作条件，空格后的部分作 continuation"""
    index = sentence.index('_')
    return sentence[:index] + option, sentence[index + 1:]


                                                                             

def _shard_worker(gpu_id, shard, out_path, model_path):
    """单卡 worker：一个 vLLM 实例按 kind 分三批跑完自己那份请求。

    和 公共单卡推理 worker 同一套思路：只让本进程看见一张卡，手写数据
    并行比 tensor_parallel_size=8 快得多；每个实例必须占不同的分布式端口。
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ.setdefault('VLLM_LOGGING_LEVEL', 'WARNING')
    os.environ['VLLM_PORT'] = str(21000 + gpu_id * 64)
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=model_path, dtype='bfloat16',
              max_model_len=common.MAX_MODEL_LEN,
              gpu_memory_utilization=GPU_MEM_UTIL,
              max_num_seqs=MAX_NUM_SEQS,
              max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
              enable_prefix_caching=True, tensor_parallel_size=1)
    tokenizer = llm.get_tokenizer()
    results = {}

    loglik = [r for r in shard if r['kind'] == 'loglik']
    if loglik:
        prompts, spans = [], []
        for request in loglik:
            context_ids = tokenizer(request['context'],
                                    add_special_tokens=False)['input_ids']
            full_ids = tokenizer(request['context'] + request['continuation'],
                                 add_special_tokens=False)['input_ids']
                                                          
            boundary = min(len(context_ids), max(len(full_ids) - 1, 0))
            prompts.append(TokensPrompt(
                prompt_token_ids=full_ids[-common.MAX_MODEL_LEN:]))
            spans.append(boundary - max(0, len(full_ids) - common.MAX_MODEL_LEN))
        params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
                                                                  
                                      
        outputs = llm.generate(prompts, params)
        for request, span, output in zip(loglik, spans, outputs):
            token_ids = output.prompt_token_ids
            token_logprobs = output.prompt_logprobs
            total, count = 0.0, 0
            for position in range(max(span, 1), len(token_ids)):
                entry = token_logprobs[position]
                if not entry:
                    continue
                total += entry[token_ids[position]].logprob
                count += 1
            results[request['index']] = {'logprob': total, 'tokens': count}

    letter = [r for r in shard if r['kind'] == 'letter']
    if letter:
                                                   
        marks = sorted({m for r in letter for m in r.get('letters', 'ABCD')})
        candidates = {}
        for mark in marks:
            ids = set()
            for text in (mark, ' ' + mark):
                piece = tokenizer(text, add_special_tokens=False)['input_ids']
                if piece:
                    ids.add(piece[0])
            candidates[mark] = ids
        params = SamplingParams(temperature=0.0, max_tokens=1,
                                logprobs=LETTER_TOP_LOGPROBS)
        outputs = llm.generate([r['context'] for r in letter], params)
        for request, output in zip(letter, outputs):
            table = output.outputs[0].logprobs[0] or {}
            scores = {}
            for mark in request.get('letters', 'ABCD'):
                values = [table[i].logprob for i in candidates[mark] if i in table]
                scores[mark] = max(values) if values else None
            results[request['index']] = {'letters': scores}

    generate = [r for r in shard if r['kind'] == 'gen']
    if generate:
        prompts = [tokenizer.apply_chat_template(
            [{'role': 'user', 'content': r['context']}],
            tokenize=False, add_generation_prompt=True) for r in generate]
                                                             
        params = [SamplingParams(temperature=0.0,
                                 max_tokens=r.get('max_new', IFEVAL_MAX_NEW_TOKENS))
                  for r in generate]
        outputs = llm.generate(prompts, params)
        for request, output in zip(generate, outputs):
            results[request['index']] = {'text': output.outputs[0].text}


    with open(out_path, 'w', encoding='utf-8') as handle:
        json.dump(results, handle)


def run_requests(requests, gpu_ids, model_path):
    """按卡分片并行执行，返回 index -> 结果 的 dict"""
    import multiprocessing as mp
    import shutil
    import tempfile
    for index, request in enumerate(requests):
        request['index'] = index
    ctx = mp.get_context('spawn')
    tmp = tempfile.mkdtemp(prefix='disaster_')
    try:
        procs = []
        for order, gpu_id in enumerate(gpu_ids):
            path = os.path.join(tmp, '%d.json' % gpu_id)
            shard = requests[order::len(gpu_ids)]
            proc = ctx.Process(target=_shard_worker,
                               args=(gpu_id, shard, path, model_path))
            proc.start()
            procs.append((gpu_id, proc, path))
        merged = {}
        for gpu_id, proc, path in procs:
            proc.join()
            if proc.exitcode != 0 or not os.path.exists(path):
                raise RuntimeError('GPU %d 的评测进程失败，exitcode=%s'
                                   % (gpu_id, proc.exitcode))
            with open(path, encoding='utf-8') as handle:
                merged.update({int(k): v for k, v in json.load(handle).items()})
        if len(merged) != len(requests):
            raise RuntimeError('结果数不匹配：%d != %d' % (len(merged), len(requests)))
        return merged
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


                                                                              

def score_hellaswag(docs, requests, results):
    correct = correct_norm = 0
    details = []
    for index, doc in enumerate(docs):
        scores = [None] * len(doc['choices'])
        for request in requests:
            if request['task'] == 'hellaswag' and request['doc'] == index:
                scores[request['choice']] = results[request['index']]['logprob']
        best = max(range(len(scores)), key=lambda i: scores[i])
        norm = [scores[i] / max(len(doc['choices'][i]), 1)
                for i in range(len(scores))]
        best_norm = max(range(len(norm)), key=lambda i: norm[i])
        correct += int(best == doc['gold'])
        correct_norm += int(best_norm == doc['gold'])
        details.append({'task': 'hellaswag', 'doc': index, 'gold': doc['gold'],
                        'pred': best, 'pred_norm': best_norm})
    total = max(len(docs), 1)
    return {'acc': round(correct / total, 4),
            'acc_norm': round(correct_norm / total, 4), 'n': len(docs)}, details


def score_truthfulqa(rows, requests, results):
    import math
    mc1 = mc2 = 0.0
    details = []
    for index, row in enumerate(rows):
        picked = {'mc1_targets': {}, 'mc2_targets': {}}
        for request in requests:
            if request['task'] == 'truthfulqa' and request['doc'] == index:
                picked[request['field']][request['choice']] =\
                    results[request['index']]['logprob']
        one = picked['mc1_targets']
        gold = row['mc1_targets']['labels'].index(1)
        best = max(one, key=lambda i: one[i])
        mc1 += int(best == gold)
        two = picked['mc2_targets']
        labels = row['mc2_targets']['labels']
        probs = {i: math.exp(v) for i, v in two.items()}
        mass = sum(probs.values()) or 1.0
        score = sum(p / mass for i, p in probs.items() if labels[i] == 1)
        mc2 += score
        details.append({'task': 'truthfulqa', 'doc': index, 'mc1_gold': gold,
                        'mc1_pred': best, 'mc2': round(score, 4)})
    total = max(len(rows), 1)
    return {'mc1': round(mc1 / total, 4), 'mc2': round(mc2 / total, 4),
            'n': len(rows)}, details


def score_mmlu(test, requests, results):
    letters = 'ABCD'
    correct = 0
    per_subject = {}
    details = []
    for request in requests:
        if request['task'] != 'mmlu':
            continue
        row = test[request['doc']]
        scores = results[request['index']]['letters']
        available = {k: v for k, v in scores.items() if v is not None}
        pred = max(available, key=lambda k: available[k]) if available else None
        hit = int(pred == letters[row['answer']])
        correct += hit
        bucket = per_subject.setdefault(row['subject'], [0, 0])
        bucket[0] += hit
        bucket[1] += 1
        details.append({'task': 'mmlu', 'doc': request['doc'],
                        'subject': row['subject'],
                        'gold': letters[row['answer']], 'pred': pred})
    total = max(len(test), 1)
    macro = sum(h / max(c, 1) for h, c in per_subject.values()) /\
        max(len(per_subject), 1)
    return {'acc': round(correct / total, 4), 'acc_macro': round(macro, 4),
            'n': len(test), 'num_subjects': len(per_subject)}, details


def score_winogrande(rows, requests, results):
    correct = 0
    details = []
    for index, row in enumerate(rows):
        scores = {}
        for request in requests:
            if request['task'] == 'winogrande' and request['doc'] == index:
                scores[request['choice']] = results[request['index']]['logprob']
        best = max(scores, key=lambda i: scores[i])
        gold = int(row['answer']) - 1
        correct += int(best == gold)
        details.append({'task': 'winogrande', 'doc': index, 'gold': gold,
                        'pred': best})
    total = max(len(rows), 1)
    return {'acc': round(correct / total, 4), 'n': len(rows)}, details


def score_mc(task, docs, requests, results):
    """通用 MC-loglik 打分：acc（原始 logprob argmax）+ acc_norm（按选项字符数归一）。

    单遍扫描请求，按 doc 收集每个选项的 logprob，避免 O(docs×requests) 的嵌套。
    """
    bucket = {}
    for request in requests:
        if request['task'] != task:
            continue
        bucket.setdefault(request['doc'], {})[request['choice']] =\
            results[request['index']]['logprob']
    correct = correct_norm = 0
    details = []
    for index, doc in enumerate(docs):
        scores = bucket.get(index, {})
        if not scores:
            continue
        raw = {i: scores[i] for i in scores}
        norm = {i: scores[i] / max(len(str(doc['choices'][i])), 1) for i in scores}
        best = max(raw, key=lambda i: raw[i])
        best_norm = max(norm, key=lambda i: norm[i])
        correct += int(best == doc['gold'])
        correct_norm += int(best_norm == doc['gold'])
        details.append({'task': task, 'doc': index, 'gold': doc['gold'],
                        'pred': best, 'pred_norm': best_norm})
    total = max(len(docs), 1)
    return {'acc': round(correct / total, 4),
            'acc_norm': round(correct_norm / total, 4), 'n': len(docs)}, details


def score_mmlu_pro(pack, requests, results):
    test = pack[0]
    correct = 0
    per_cat = {}
    details = []
    for request in requests:
        if request['task'] != 'mmlu_pro':
            continue
        row = test[request['doc']]
        scores = {k: v for k, v in results[request['index']]['letters'].items()
                  if v is not None}
        pred = max(scores, key=lambda k: scores[k]) if scores else None
        hit = int(pred == row['answer'])
        correct += hit
        b = per_cat.setdefault(row['category'], [0, 0])
        b[0] += hit
        b[1] += 1
        details.append({'task': 'mmlu_pro', 'doc': request['doc'],
                        'category': row['category'], 'gold': row['answer'],
                        'pred': pred})
    total = max(len(test), 1)
    macro = sum(h / max(c, 1) for h, c in per_cat.values()) / max(len(per_cat), 1)
    return {'acc': round(correct / total, 4), 'acc_macro': round(macro, 4),
            'n': len(test), 'num_categories': len(per_cat)}, details


def _extract_number(text):
    """取生成里 #### 后的数，取不到就取最后一个数字。"""
    marker = re.search(r'####\s*(-?[\d,\.]+)', text)
    raw = marker.group(1) if marker else None
    if raw is None:
        nums = re.findall(r'-?\d[\d,]*(?:\.\d+)?', text)
        raw = nums[-1] if nums else None
    if raw is None:
        return None
    raw = raw.replace(',', '').rstrip('.')
    return raw


def score_gsm8k(docs, requests, results):
    correct = 0
    details = []
    for request in requests:
        if request['task'] != 'gsm8k':
            continue
        doc = docs[request['doc']]
        text = results[request['index']]['text']
        pred = _extract_number(text)
        gold = doc['gold']
        hit = 0
        if pred is not None:
            try:
                hit = int(abs(float(pred) - float(gold)) < 1e-4)
            except ValueError:
                hit = int(pred == gold)
        correct += hit
        details.append({'task': 'gsm8k', 'doc': request['doc'], 'gold': gold,
                        'pred': pred})
    total = max(len(docs), 1)
    return {'acc': round(correct / total, 4), 'n': len(docs)}, details


                                                                           
                                                  
                                                               
                      

def count_words(text):
    return len(re.findall(r'\b\w+\b', text))


def count_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return len([p for p in parts if p.strip()])


def compare(value, target, relation):
    if relation in ('at least', 'least'):
        return value >= target
    if relation in ('less than', 'at most'):
        return value < target if relation == 'less than' else value <= target
    return value == target


def check_instruction(instruction_id, kwargs, response, prompt):
    """返回该条指令是否被满足。未识别的类型抛异常，避免静默算通过。"""
    kwargs = {k: v for k, v in (kwargs or {}).items() if v is not None}
    text = response

    if instruction_id == 'punctuation:no_comma':
        return ',' not in text
    if instruction_id == 'change_case:english_lowercase':
        return text == text.lower()
    if instruction_id == 'change_case:english_capital':
        return text == text.upper()
    if instruction_id == 'change_case:capital_word_frequency':
        count = len([w for w in re.findall(r'\b\w+\b', text)
                     if w.isupper() and len(w) > 1])
        return compare(count, kwargs['capital_frequency'],
                       kwargs['capital_relation'])
    if instruction_id == 'length_constraints:number_words':
        return compare(count_words(text), kwargs['num_words'], kwargs['relation'])
    if instruction_id == 'length_constraints:number_sentences':
        return compare(count_sentences(text), kwargs['num_sentences'],
                       kwargs['relation'])
    if instruction_id == 'length_constraints:number_paragraphs':
        paragraphs = [p for p in re.split(r'\n\s*\*\s*\n|\n\n+', text.strip())
                      if p.strip()]
        return len(paragraphs) == kwargs['num_paragraphs']
    if instruction_id == 'length_constraints:nth_paragraph_first_word':
        paragraphs = [p for p in re.split(r'\n\n+', text.strip()) if p.strip()]
        if len(paragraphs) != kwargs['num_paragraphs']:
            return False
        nth = kwargs['nth_paragraph']
        if nth < 1 or nth > len(paragraphs):
            return False
        first = re.findall(r'\b\w+\b', paragraphs[nth - 1])
        return bool(first) and first[0].lower() == kwargs['first_word'].lower()
    if instruction_id == 'keywords:existence':
        return all(re.search(r'\b%s\b' % re.escape(k), text, re.IGNORECASE)
                   for k in kwargs['keywords'])
    if instruction_id == 'keywords:forbidden_words':
        return not any(re.search(r'\b%s\b' % re.escape(k), text, re.IGNORECASE)
                       for k in kwargs['forbidden_words'])
    if instruction_id == 'keywords:frequency':
        hits = len(re.findall(r'\b%s\b' % re.escape(kwargs['keyword']), text,
                              re.IGNORECASE))
        return compare(hits, kwargs['frequency'], kwargs['relation'])
    if instruction_id == 'keywords:letter_frequency':
        hits = text.lower().count(kwargs['letter'].lower())
        return compare(hits, kwargs['let_frequency'], kwargs['let_relation'])
    if instruction_id == 'detectable_content:number_placeholders':
        return len(re.findall(r'\[[^\[\]]*\]', text)) >= kwargs['num_placeholders']
    if instruction_id == 'detectable_content:postscript':
        marker = kwargs['postscript_marker']
        return bool(re.search(re.escape(marker), text, re.IGNORECASE))
    if instruction_id == 'detectable_format:number_bullet_lists':
        bullets = re.findall(r'^\s*[\*\-]\s+\S', text, re.MULTILINE)
        return len(bullets) == kwargs['num_bullets']
    if instruction_id == 'detectable_format:number_highlighted_sections':
        highlights = [m for m in re.findall(r'\*[^\*\n]+\*|\*\*[^\*\n]+\*\*', text)
                      if m.strip('*').strip()]
        return len(highlights) >= kwargs['num_highlights']
    if instruction_id == 'detectable_format:title':
        return bool(re.search(r'<<[^\n<>]+>>', text))
    if instruction_id == 'detectable_format:json_format':
        stripped = re.sub(r'^```(?:json)?|```$', '', text.strip(),
                          flags=re.MULTILINE).strip()
        try:
            json.loads(stripped)
            return True
        except Exception:
            return False
    if instruction_id == 'detectable_format:multiple_sections':
        spliter = kwargs['section_spliter']
        hits = len(re.findall(r'%s\s*\d+' % re.escape(spliter), text,
                              re.IGNORECASE))
        return hits >= kwargs['num_sections']
    if instruction_id == 'detectable_format:constrained_response':
        options = ('My answer is yes.', 'My answer is no.',
                   'My answer is maybe.')
        return any(option in text for option in options)
    if instruction_id == 'startend:quotation':
        stripped = text.strip()
        return stripped.startswith('"') and stripped.endswith('"')
    if instruction_id == 'startend:end_checker':
        return text.strip().lower().endswith(kwargs['end_phrase'].strip().lower())
    if instruction_id == 'combination:two_responses':
        return len([p for p in text.split('******') if p.strip()]) == 2
    if instruction_id == 'combination:repeat_prompt':
        target = kwargs['prompt_to_repeat'].strip()
        return text.strip().startswith(target)
    if instruction_id == 'language:response_language':
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0
        try:
            return detect(text.strip()) == kwargs['language']
        except Exception:
            return False
    raise RuntimeError('未实现的 IFEval 指令类型：%s' % instruction_id)


def loose_variants(text):
    """官方 loose 口径：对回复做若干无害变换，任一变换通过即算通过。"""
    lines = [line for line in text.split('\n') if line.strip()]
    without_first = '\n'.join(lines[1:])
    without_last = '\n'.join(lines[:-1])
    no_markdown = text.replace('*', '')
    return [text, no_markdown, without_first, without_last,
            without_first.replace('*', ''), without_last.replace('*', '')]


def score_ifeval(rows, requests, results):
    prompt_strict = prompt_loose = 0
    inst_strict = inst_loose = 0
    inst_total = 0
    details = []
    for request in requests:
        if request['task'] != 'ifeval':
            continue
        row = rows[request['doc']]
        text = results[request['index']]['text']
        strict, loose = [], []
        for instruction_id, kwargs in zip(row['instruction_id_list'],
                                          row['kwargs']):
            strict.append(check_instruction(instruction_id, kwargs, text,
                                           row['prompt']))
            loose.append(any(
                check_instruction(instruction_id, kwargs, variant, row['prompt'])
                for variant in loose_variants(text)))
        inst_strict += sum(strict)
        inst_loose += sum(loose)
        inst_total += len(strict)
        prompt_strict += int(all(strict))
        prompt_loose += int(all(loose))
        details.append({'task': 'ifeval', 'doc': request['doc'],
                        'key': row['key'],
                        'instruction_id_list': row['instruction_id_list'],
                        'strict': strict, 'loose': loose,
                        'response': text})
    total = max(len(rows), 1)
    inst_total = max(inst_total, 1)
    return {'prompt_level_strict': round(prompt_strict / total, 4),
            'prompt_level_loose': round(prompt_loose / total, 4),
            'instruction_level_strict': round(inst_strict / inst_total, 4),
            'instruction_level_loose': round(inst_loose / inst_total, 4),
            'n': len(rows), 'n_instructions': inst_total}, details


                                                                               

def resolve_model(name):
    """基座在 models/ 下，训练后的模型在 TrainModels/ 下"""
    for root in (common.MODELS_DIR, common.TRAINED_DIR):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    raise RuntimeError('模型不存在：%s' % name)


def save_result(model_name, metrics, details, date, seconds, counts):
    """结果覆盖写入 TestResults/<模型名>_disaster_<日期>_{sum,detail}.json"""
                                        
    retention = [metrics[t][PRIMARY_METRIC[t]] for t in TASKS if t in metrics]
    summary = {
        'benchmark': 'general capability suite (catastrophic forgetting axis)',
        'algorithm': 'disaster',
        'model': model_name,
        'datasets': list(TASKS),
        'primary_metric': PRIMARY_METRIC,
        'protocol': {
            'mc_loglik': 'continuation loglikelihood, acc / acc_norm; tasks: %s'
                         % ', '.join(MC_TASKS),
            'truthfulqa': '0-shot multiple_choice, MC1 / MC2',
            'mmlu': '%d-shot, letter logprob over A/B/C/D (top-%d)'
                    % (MMLU_SHOTS, LETTER_TOP_LOGPROBS),
            'mmlu_pro': '%d-shot, letter logprob over A-J (top-%d)'
                        % (MMLU_PRO_SHOTS, LETTER_TOP_LOGPROBS),
            'winogrande': '%d-shot, partial evaluation loglikelihood'
                          % WINOGRANDE_SHOTS,
            'gsm8k': 'greedy generation, flexible numeric extraction (#### or last '
                     'number), max_new_tokens=%d' % GSM8K_MAX_NEW_TOKENS,
            'ifeval': 'greedy generation with chat template, max_new_tokens=%d, '
                      'in-house reimplementation of the 25 instruction checkers'
                      % IFEVAL_MAX_NEW_TOKENS,
        },
        'decode': {'temperature': 0.0, 'greedy': True},
        'max_model_len': common.MAX_MODEL_LEN,
        'engine': {'gpu_memory_utilization': GPU_MEM_UTIL,
                   'max_num_seqs': MAX_NUM_SEQS,
                   'max_num_batched_tokens': MAX_NUM_BATCHED_TOKENS,
                   'enable_prefix_caching': True},
        'num_requests': counts,
        'tasks': metrics,
        'retention_mean': round(sum(retention) / max(len(retention), 1), 4),
        'elapsed_seconds': round(seconds, 1),
        'elapsed': common.format_duration(seconds),
        'created_date': date,
    }
    os.makedirs(common.RESULTS_DIR, exist_ok=True)
    stem = '%s_disaster_%s' % (model_name, date)
    summary_path = os.path.join(common.RESULTS_DIR, stem + '_sum.json')
    detail_path = os.path.join(common.RESULTS_DIR, stem + '_detail.json')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(detail_path, 'w', encoding='utf-8') as handle:
        json.dump({'model': model_name, 'algorithm': 'disaster',
                   'created_date': date, 'details': details},
                  handle, ensure_ascii=False, indent=2)
    return summary_path, detail_path, summary


def main():
    started = time.time()
    data = {
        'hellaswag': load_hellaswag(),
        'piqa': load_piqa(),
        'openbookqa': load_openbookqa(),
        'commonsenseqa': load_commonsenseqa(),
        'arc_easy': load_arc_easy(),
        'arc_challenge': load_arc_challenge(),
        'sciq': load_sciq(),
        'logiqa': load_logiqa(),
        'boolq': load_boolq(),
        'truthfulqa': load_truthfulqa(),
        'mmlu': load_mmlu(),
        'mmlu_pro': load_mmlu_pro(),
        'winogrande': load_winogrande(),
        'gsm8k': load_gsm8k(),
        'ifeval': load_ifeval(),
    }
    sizes = {k: (len(v[0]) if isinstance(v, tuple) else len(v))
             for k, v in data.items()}
    print('数据加载完成（15 集）：%s' % sizes, flush=True)

    requests = build_requests(data)
    counts = {}
    for request in requests:
        counts[request['task']] = counts.get(request['task'], 0) + 1
    print('请求总数 %d：%s' % (len(requests), counts), flush=True)

    gpu_ids = common.available_gpus()
    if os.environ.get('EVAL_GPU_IDS'):
        gpu_ids = [int(x) for x in os.environ['EVAL_GPU_IDS'].split(',') if x.strip()]
    print('本次评测使用 GPU：%s' % gpu_ids, flush=True)
    date = datetime.now().strftime('%Y%m%d')

    for model_name in MODEL_NAMES:
        model_path = resolve_model(model_name)
        model_started = time.time()
        print('\n===== %s =====' % model_name, flush=True)
        results = run_requests(requests, gpu_ids, model_path)
        print('推理完成，用时 %.1fs；开始打分'
              % (time.time() - model_started), flush=True)

        metrics, details = {}, []
                                                       
        for task in MC_TASKS:
            metrics[task], part = score_mc(task, data[task], requests, results)
            details.extend(part)
            print('  %-14s %s' % (task, metrics[task]), flush=True)
        for scorer, key, payload in (
                (score_truthfulqa, 'truthfulqa', data['truthfulqa']),
                (score_mmlu, 'mmlu', data['mmlu'][0]),
                (score_mmlu_pro, 'mmlu_pro', data['mmlu_pro']),
                (score_winogrande, 'winogrande', data['winogrande'][0]),
                (score_gsm8k, 'gsm8k', data['gsm8k']),
                (score_ifeval, 'ifeval', data['ifeval'])):
            metrics[key], part = scorer(payload, requests, results)
            details.extend(part)
            print('  %-14s %s' % (key, metrics[key]), flush=True)

        summary_path, detail_path, summary = save_result(
            model_name, metrics, details, date,
            time.time() - model_started, counts)
        print('结果已覆盖写入：%s 和 %s（耗时 %s，retention_mean=%.4f）'
              % (summary_path, detail_path, summary['elapsed'],
                 summary['retention_mean']), flush=True)

    print('\n运行总耗时：%s' % common.format_duration(time.time() - started),
          flush=True)



if __name__ == '__main__':
    main()
