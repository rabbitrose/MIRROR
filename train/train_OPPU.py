"""OPPU 基线（One PEFT Per User, Tan et al., EMNLP 2024）：训练 + 评测一体。

两级结构与原文一致：
  1. base PEFT：在训练用户上学任务本身。这里直接复用同任务、同基座的 contextSFT
     合并权重（OPPU_BASE_PEFT），它与其他方法共享同一套 800 条样本 / 250 步配置。
  2. personal PEFT：给每个测试用户单独训一个 LoRA，数据只来自该用户自己的历史。
     历史第 j 条变成一条 (question_j, answer_j) 监督样本，question 用与测试题相同的
     任务指令，前面拼上该用户在第 j 条之前的最近 OPPU_CTX_ITEMS 条历史（OPPU 的
     PEFT + retrieval 设定）。每个用户从同一初值重新初始化 LoRA，互不共享。
推理：用户私有 LoRA 挂在 base PEFT 上，用 vLLM 多 LoRA 推理；prompt 与 eval_lamp_base
的 full user context 逐字节一致，解码参数（温度 0.4、种子、512 token）也相同。

测试用户逐个训 LoRA 成本高，默认在固定种子抽取的 OPPU_MAX_USERS 个测试用户上评测
（0 = 全部）。结果文件里记录了抽到的 sample_id，其他方法可在同一子集上对比。

直接 `python train/train_OPPU.py` 运行，配置走环境变量：
  OPPU_BASE_PEFT   TrainModels 下的 base PEFT 模型名（必填）
  OPPU_OUTPUT      结果用的模型名（必填，例如 Qwen3-1.7B-OPPU-long）
  EVAL_TASKS       单个任务名
  OPPU_MAX_USERS   评测用户数（默认 500，0=全部）
  OPPU_ITEMS       每个用户最多用最近多少条历史训练（默认 16）
  OPPU_EPOCHS      每个用户的训练轮数（默认 2）
  OPPU_CTX_ITEMS   每条个人训练样本前拼几条更早的历史（默认 2）
"""
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HUAYI_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(HUAYI_DIR, 'eval'))
sys.path.insert(0, BASE_DIR)
import eval_common as evalbase
import eval_lamp_base as evalfull
import train_sft as common

BASE_PEFT = os.environ['OPPU_BASE_PEFT']
OUTPUT_NAME = os.environ['OPPU_OUTPUT']
MAX_USERS = int(os.environ.get('OPPU_MAX_USERS', '500'))
ITEMS = int(os.environ.get('OPPU_ITEMS', '16'))
EPOCHS = int(os.environ.get('OPPU_EPOCHS', '2'))
CTX_ITEMS = int(os.environ.get('OPPU_CTX_ITEMS', '2'))
LR = common.LEARNING_RATE
GRAD_ACCUM = common.GRAD_ACCUM
PERSONAL_TOKEN_BUDGET = int(os.environ.get('OPPU_TOKEN_BUDGET', '4096'))
SUBSET_SEED = 20261113
# 全量补跑时按分片并行：OPPU_SHARD=k/N 取剩余用户的第 k 份（0 起）；
# OPPU_EXCLUDE_DETAIL 指向已评测过的 detail 文件，里面的用户不再重跑
SHARD = os.environ.get('OPPU_SHARD', '')
EXCLUDE_DETAIL = os.environ.get('OPPU_EXCLUDE_DETAIL', '')

# 历史条目 -> 与测试题同一指令格式的 question / answer
ITEM_QA = {
    'abstract_generation': (lambda p: 'Generate an abstract for the title "%s"' % p['title'],
                            lambda p: p['abstract']),
    'news_headline': (lambda p: 'Generate a headline for the following article: %s' % p['text'],
                      lambda p: p['title']),
    'scholarly_title': (lambda p: 'Generate a title for the following abstract of a paper: %s'
                        % p['abstract'], lambda p: p['title']),
}


def personal_samples(task, profile):
    """用户历史 -> 个人监督样本 [(prefix, question, answer)]，只用最近 ITEMS 条。"""
    q_of, a_of = ITEM_QA[task]
    start = max(0, len(profile) - ITEMS)
    out = []
    for j in range(start, len(profile)):
        item = profile[j]
        answer = (a_of(item) or '').strip()
        if not answer:
            continue
        ctx = profile[max(0, j - CTX_ITEMS):j]
        prefix = evalbase.render_profiles(task, ctx) if ctx else ''
        out.append((prefix, q_of(item), answer))
    return out


def pick_users(n):
    idx = list(range(n))
    if MAX_USERS > 0 and MAX_USERS < n:
        idx = sorted(random.Random(SUBSET_SEED).sample(idx, MAX_USERS))
    if EXCLUDE_DETAIL:
        with open(EXCLUDE_DETAIL, encoding='utf-8') as f:
            done = set(int(d['sample_id'].rsplit('-', 1)[1]) for d in json.load(f)['details'])
        idx = [i for i in idx if i not in done]
    if SHARD:
        k, total = (int(x) for x in SHARD.split('/'))
        idx = idx[k::total]
    return idx


def encode(tokenizer, prefix, question, answer):
    content = evalfull.clip_content(tokenizer, prefix, question, PERSONAL_TOKEN_BUDGET)[0]
    prompt_ids = tokenizer(common.render(tokenizer, content),
                           add_special_tokens=False)['input_ids']
    answer_ids = common.answer_token_ids(tokenizer, answer)
    return prompt_ids + answer_ids, [-100] * len(prompt_ids) + answer_ids


def train_personal_adapters(task, rows, users, base_path, adapter_root):
    """逐用户：LoRA 复位到同一初值 -> 在该用户历史上训 EPOCHS 轮 -> 只存 adapter。"""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(common.SEED)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_path, dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    model.config.use_cache = False
    lora = LoraConfig(r=common.LORA_R, lora_alpha=common.LORA_ALPHA,
                      lora_dropout=common.LORA_DROPOUT,
                      target_modules=common.LORA_TARGETS, bias='none',
                      task_type='CAUSAL_LM')
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    lora_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    init_state = {n: p.detach().clone() for n, p in lora_params}

    stats = []
    started = time.time()
    for k, ui in enumerate(users):
        row = rows[ui]
        samples = personal_samples(task, row.get('profile') or [])
        with torch.no_grad():
            for n, p in lora_params:
                p.copy_(init_state[n])
        model.train()
        opt = torch.optim.AdamW([p for _, p in lora_params], lr=LR, weight_decay=0.0)
        rng = random.Random('%s-%d' % (task, ui))
        losses, micro = [], 0
        for _ in range(EPOCHS):
            order = list(range(len(samples)))
            rng.shuffle(order)
            for j in order:
                ids, labels = encode(tokenizer, *samples[j])
                ids_t = torch.tensor([ids], device='cuda')
                lab_t = torch.tensor([labels], device='cuda')
                loss = model(input_ids=ids_t, labels=lab_t).loss
                (loss / GRAD_ACCUM).backward()
                losses.append(float(loss))
                micro += 1
                if micro % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_([p for _, p in lora_params],
                                                   common.MAX_GRAD_NORM)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
        if micro % GRAD_ACCUM:
            torch.nn.utils.clip_grad_norm_([p for _, p in lora_params], common.MAX_GRAD_NORM)
            opt.step()
            opt.zero_grad(set_to_none=True)
        model.save_pretrained(os.path.join(adapter_root, 'u%06d' % ui))
        stats.append({'row': ui, 'items': len(samples), 'steps': -(-micro // GRAD_ACCUM),
                      'loss_first': round(losses[0], 4) if losses else None,
                      'loss_last': round(losses[-1], 4) if losses else None})
        if k % 25 == 0 or k == len(users) - 1:
            el = time.time() - started
            print('[%d/%d] row=%d items=%d loss %.3f -> %.3f  %.1fs/用户'
                  % (k + 1, len(users), ui, len(samples),
                     losses[0] if losses else 0, losses[-1] if losses else 0,
                     el / (k + 1)), flush=True)
    del model
    torch.cuda.empty_cache()
    return stats, time.time() - started
def _lora_infer(gpu_id, base_path, prompts, adapters, out_path):
    """vLLM 多 LoRA 推理：每条 prompt 挂它自己用户的 adapter，解码参数同 eval_common。"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ.setdefault('VLLM_LOGGING_LEVEL', 'WARNING')
    # 同卡可能并行两个分片，固定端口会 EADDRINUSE，这里取一个空闲端口
    import socket
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        os.environ['VLLM_PORT'] = str(s.getsockname()[1])
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=base_path, dtype='bfloat16', max_model_len=evalbase.MAX_MODEL_LEN,
              gpu_memory_utilization=evalbase.GPU_MEM_UTIL, enable_prefix_caching=True,
              tensor_parallel_size=1, enable_lora=True, max_lora_rank=common.LORA_R,
              max_loras=8)
    reqs = [LoRARequest('u%d' % i, i + 1, path) for i, path in enumerate(adapters)]
    outs = llm.generate(prompts, SamplingParams(
        temperature=evalbase.DECODE_TEMPERATURE, max_tokens=evalbase.MAX_NEW_TOKENS,
        seed=evalbase.DECODE_SEED), lora_request=reqs)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump([o.outputs[0].text for o in outs], f)


def main():
    import multiprocessing as mp
    import tempfile
    from transformers import AutoTokenizer
    import nltk
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        if not nltk.download('wordnet', quiet=True):
            raise RuntimeError('METEOR 依赖的 NLTK WordNet 不可用')

    started = time.time()
    tasks = evalbase.EVAL_TASKS
    if len(tasks) != 1 or tasks[0] not in ITEM_QA:
        raise RuntimeError('OPPU 一次只跑一个任务，EVAL_TASKS=%s' % tasks)
    task = tasks[0]
    base_path = os.path.join(evalbase.TRAINED_DIR, BASE_PEFT)
    if not os.path.isfile(os.path.join(base_path, 'config.json')):
        raise RuntimeError('base PEFT 不存在：%s' % base_path)
    gpu_ids = [int(x) for x in os.environ.get('EVAL_GPU_IDS', '0').split(',') if x.strip()]

    rows = evalbase.load_task(task)
    users = pick_users(len(rows))
    print('OPPU %s：base PEFT=%s，测试 %d/%d 个用户，每用户最近 %d 条历史 × %d 轮'
          % (task, BASE_PEFT, len(users), len(rows), ITEMS, EPOCHS), flush=True)

    out_dir = os.path.join(evalbase.TRAINED_DIR, OUTPUT_NAME)
    adapter_root = os.path.join(out_dir, 'users')
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(adapter_root)
    stats, train_seconds = train_personal_adapters(task, rows, users, base_path, adapter_root)

    # 推理 prompt 与 eval_lamp_base 的 full user context 完全一致
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    prompts, clipped, kept_tokens, sample_ids = [], [], [], []
    for ui in users:
        row = rows[ui]
        prefix, question, _ = evalfull.full_context(task, row['input'], row.get('profile') or [])
        content, kept = evalfull.clip_content(tokenizer, prefix, question,
                                              evalbase.USER_TOKEN_BUDGET)
        clipped.append(content)
        kept_tokens.append(kept)
        prompts.append(evalfull.render_prompt(tokenizer, content))
        sample_ids.append('%s-%06d' % (task, ui))
    adapters = [os.path.join(adapter_root, 'u%06d' % ui) for ui in users]

    infer_started = time.time()
    tmp = tempfile.mkdtemp(prefix='oppu_infer_')
    path = os.path.join(tmp, 'out.json')
    p = mp.get_context('spawn').Process(
        target=_lora_infer, args=(gpu_ids[0], base_path, prompts, adapters, path))
    p.start()
    p.join()
    if p.exitcode != 0 or not os.path.isfile(path):
        raise RuntimeError('OPPU 推理失败 exitcode=%s' % p.exitcode)
    with open(path, encoding='utf-8') as f:
        texts = json.load(f)
    shutil.rmtree(tmp, ignore_errors=True)
    infer_seconds = time.time() - infer_started

    preds = [evalbase.postprocess(t) for t in texts]
    refs = [rows[ui]['output'] for ui in users]
    metrics = evalbase.run_metrics({task: {'preds': preds, 'refs': refs}}, gpu_ids)
    details = [{
        'sample_id': sid, 'task': task, 'input': rows[ui]['input'],
        'model_input': content, 'model_output': pred, 'golden_truth': ref,
        'profile_entries_total': len(rows[ui].get('profile') or []),
        'personal_items': st['items'], 'prefix_tokens_kept': kept,
    } for sid, ui, content, pred, ref, st, kept in zip(
        sample_ids, users, clipped, preds, refs, stats, kept_tokens)]
    date = datetime.now().strftime('%Y%m%d')
    summary_path, detail_path, results = evalbase.save_evaluation_result(
        model_name=OUTPUT_NAME, algorithm='base',
        method='OPPU: base PEFT (%s) + one LoRA per test user trained on that user\'s '
               'history, full user context at inference (thinking=off)' % BASE_PEFT,
        retrieval_seconds=0.0, model_seconds=time.time() - started,
        infer_seconds=infer_seconds, metrics=metrics, created_date=date,
        details=details,
        context_policy='question_reserved_first_then_history_head_fills_budget',
        base_peft=BASE_PEFT, users_evaluated=len(users), users_total=len(rows),
        subset_seed=SUBSET_SEED, shard=SHARD, exclude_detail=os.path.basename(EXCLUDE_DETAIL),
        personal_items=ITEMS, personal_epochs=EPOCHS,
        personal_ctx_items=CTX_ITEMS, personal_lr=LR, personal_grad_accum=GRAD_ACCUM,
        personal_token_budget=PERSONAL_TOKEN_BUDGET,
        lora={'r': common.LORA_R, 'alpha': common.LORA_ALPHA,
              'dropout': common.LORA_DROPOUT, 'target_modules': common.LORA_TARGETS},
        personal_train_seconds=round(train_seconds, 1),
        max_model_len=evalbase.MAX_MODEL_LEN, user_token_budget=evalbase.USER_TOKEN_BUDGET,
        decode_temperature=evalbase.DECODE_TEMPERATURE, decode_seed=evalbase.DECODE_SEED,
        max_new_tokens=evalbase.MAX_NEW_TOKENS, thinking='off')
    os.makedirs(common.LOG_DIR, exist_ok=True)
    with open(os.path.join(common.LOG_DIR, '%s_train.json' % OUTPUT_NAME), 'w',
              encoding='utf-8') as f:
        json.dump({'output_model': OUTPUT_NAME, 'base_peft': BASE_PEFT, 'task': task,
                   'per_user': stats, 'train_seconds': round(train_seconds, 1),
                   'created_date': date}, f, ensure_ascii=False, indent=2)
    # 推理完成后 adapter 只是中间产物，删掉省盘（几百个 × 几十 MB）
    if os.environ.get('OPPU_KEEP_ADAPTERS', '0') == '0':
        shutil.rmtree(adapter_root, ignore_errors=True)
    print('结果：%s  %s' % (summary_path, json.dumps(metrics.get(task), ensure_ascii=False)),
          flush=True)


if __name__ == '__main__':
    main()