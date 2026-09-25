"""纯 OPD：on-policy 混合 KL 蒸馏（generalized JSD），**不含 SFT loss**。

loss = (1/n) Σ_i JSD_β( P_T ‖ P_S )

teacher 在 base_gold 上下文（full history + gold answer 作参考答案 + 引导词），
student 在 full user context + question 上 on-policy 采样，
每步把 student 的分布朝 teacher 拉。无 SFT 项、无个性化权重。

每个训练步：
  1. student 基于 full user context + question 自采样生成 y_1..y_n（on-policy，无梯度）；
  2. teacher 在 base_gold 上下文 + 同一串 y_<i 上给出 P_T(·|hist, x, gold, y_<i)；
  3. loss = (1/n) Σ_i JSD_β( P_T ‖ P_S )，只更新 student。

为什么用混合 KL 而不是单边：
  - 正向 KL(q‖p)（mode-covering）会让 student 去覆盖 teacher 的全部支撑，1.7B
    容量不够时会摊平分布、生成变啰嗦；
  - 逆向 KL(p‖q)（mode-seeking）会让 student 塌到 teacher 的单个峰上，多样性崩，
    也最容易训崩；
  - generalized JSD 以混合分布 m = βq + (1-β)p 为锚，两个方向各占一半，且有界
    （β=0.5 时 ≤ log2），梯度不会因为某个 token 上 p→0 而爆炸。这是"别训崩"
    最直接的保障。

teacher 的 base_gold 一档在 LongLaMP 上实测 ROUGE-1 0.6079/0.7747/0.7837，
是目前最强的一档，所以这里严格复刻它的上下文配置，thinking 两侧都关。

如果需要混合 KL + SFT loss 的版本，请用 train_opdsft.py。
如果需要带个性化权重（token 级 causal preference weighting），请用 train_copd.py / train_psopd.py。

按 CLAUDE.md 约定：
  - 直接 `python train_opd.py` 运行，不接收命令行参数
  - OPD 相关训练默认用混合 KL loss，除非单独说明才加 SFT loss
  - 所有 OPD 变体按 150 步跑

产物：
  TrainModels/<OPD_OUTPUT>                        合并 LoRA 后的模型
  trainlogs/<OPD_OUTPUT>_train.json               训练记录
  figs/<OPD_OUTPUT>_curves.png                    loss / KL 曲线（单张图）
"""
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime

                                                                   
                                              
if int(os.environ.get('LOCAL_RANK', '-1')) < 0:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
                                                                    
                                       
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HUAYI_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(HUAYI_DIR, 'eval'))
sys.path.insert(0, BASE_DIR)
import eval_lamp_base as evalfull                      
import eval_common as evalbase                    
import train_sft as common                             
                

def _resolve_model(name):
    """OPD_STUDENT/OPD_TEACHER 支持基座名(models/)或训练后模型名(TrainModels/)，
    也支持绝对路径。这样 student/teacher 可以直接指向 contextSFT 等已训练模型。"""
    if os.path.isabs(name):
        return name
    for root in ('models', 'TrainModels'):
        cand = os.path.join(HUAYI_DIR, root, name)
        if os.path.isdir(cand):
            return cand
    return os.path.join(HUAYI_DIR, 'models', name)                  


STUDENT_MODEL = _resolve_model(os.environ.get('OPD_STUDENT', 'Qwen3-1.7B'))
TEACHER_MODEL = _resolve_model(os.environ.get('OPD_TEACHER', 'Qwen3-4B'))
                                                         
                        
OUTPUT_NAME = evalbase.with_dataset_suffix(
    os.environ.get('OPD_OUTPUT', 'Qwen3-1.7B-4bopd'), os.environ.get('TRAIN_TASKS'))
TRAINED_DIR = common.TRAINED_DIR
LOG_DIR = common.LOG_DIR
FIG_DIR = os.environ.get(
    'OPD_FIG_DIR', os.path.join(HUAYI_DIR, 'figs'))

                                                     
                                                         
                 
THINKING = 'off'

                                          
USER_TOKEN_BUDGET = evalbase.USER_TOKEN_BUDGET             
MAX_SEQ_LEN = evalbase.MAX_MODEL_LEN                       
GOLD_TOKEN_BUDGET = int(os.environ.get('OPD_GOLD_BUDGET', '2048'))
REFERENCE_HEAD = os.environ.get('OPD_REF_HEAD', 'Here is a reference solution:\n')
                                                           
                                                                 
REFERENCE_HINT = os.environ.get('OPD_REF_HINT',
                 '\nAfter understanding the reference solution, please try to '
                 'solve this problem using your own approach below:\n')

                                
                                             
                                                            
                                                             
                                                         
                                                   
                       
 
                                                 
                                                          
                                                         
                      
ROLLOUT_LENGTH_RATIO = float(os.environ.get('OPD_LEN_RATIO', '1.3'))
ROLLOUT_MIN_TOKENS = int(os.environ.get('OPD_LEN_MIN', '64'))
ROLLOUT_TASK_CAP = {
    'abstract_generation': int(os.environ.get('OPD_CAP_ABSTRACT', '400')),
    'product_review': int(os.environ.get('OPD_CAP_PRODUCT', '1152')),
    'topic_writing': int(os.environ.get('OPD_CAP_TOPIC', '1280')),
}
                                                           
ROLLOUT_HARD_CAP = int(os.environ.get('OPD_LEN_HARD_CAP', '1280'))
                                                 
                                       
GEN_TEMPERATURE = float(os.environ.get('OPD_GEN_TEMP', '1.0'))
GEN_TOP_P = 1.0


def rollout_budget(task, gold_tokens):
    """逐样本 rollout 上限：gold 长度 × ratio，再按任务 CAP 和全局硬上限收口。"""
    want = int(round(gold_tokens * ROLLOUT_LENGTH_RATIO))
    cap = min(ROLLOUT_TASK_CAP.get(task, ROLLOUT_HARD_CAP), ROLLOUT_HARD_CAP)
    return max(ROLLOUT_MIN_TOKENS, min(want, cap))

                 
                                                            
                                                
JSD_BETA = float(os.environ.get('OPD_JSD_BETA', '0.5'))

                    
                                                      
                                                
                                          
SAMPLES_PER_TASK = int(os.environ.get('OPD_SAMPLES_PER_TASK', '800'))
EPOCHS = 1
                              
MAX_STEPS = int(os.environ.get('OPD_MAX_STEPS', '150'))                       
                                                                 
SAVE_STEPS = [int(x) for x in os.environ.get('OPD_SAVE_STEPS', '').replace(' ', '').split(',') if x]
LEARNING_RATE = float(os.environ.get('OPD_LR', '1e-4'))
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 1
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.05
SEED = 20260910

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = common.LORA_TARGETS
              


def render(tokenizer, content):
    """渲染对话模板，thinking=off 时模板会预填空的 <think></think>。"""
    messages = [{'role': 'system', 'content': evalbase.SYSTEM_PROMPT},
                {'role': 'user', 'content': content}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=(THINKING == 'on'))
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def build_pair(tokenizer, task, row):
    """构造 (student prompt, teacher prompt, task, gold_tokens)。

    student：full history + question，与 eval_lamp_base 的 base 口径逐字节一致。
    teacher：full history + question + gold 参考答案 + 提示语，严格复刻 base_gold。
             额度分配顺序 gold -> question -> 历史，所以 teacher 的历史会比 student
             少约 200~300 token（实测 base_gold 历史中位 10536 vs base 10779）。
             这是为了让 teacher 与已实测的那一档完全同配置，代价可接受。
    gold_tokens：gold 参考答案的 token 数，只用于按样本定 rollout 上限。纯 KL 档
             不需要 gold 的 token ids 本身（那是 train_opdsft.py 的 SFT 半才用）。
    """
    prefix, question, _ = evalfull.full_context(
        task, row['input'], row.get('profile') or [])
    student_content, _ = evalfull.clip_content(
        tokenizer, prefix, question, USER_TOKEN_BUDGET)

    gold_ids = tokenizer(row['output'].strip(), add_special_tokens=False,
                         truncation=True, max_length=GOLD_TOKEN_BUDGET)['input_ids']
    gold = tokenizer.decode(gold_ids, skip_special_tokens=True)
    tail = '%s%s%s' % (REFERENCE_HEAD, gold, REFERENCE_HINT)
    room = (USER_TOKEN_BUDGET
            - len(tokenizer(tail, add_special_tokens=False)['input_ids'])
            - len(tokenizer(question, add_special_tokens=False)['input_ids']))
    if room > 0 and prefix:
        hist_ids = tokenizer(prefix, add_special_tokens=False)['input_ids'][-room:]
        history = tokenizer.decode(hist_ids, skip_special_tokens=True)
    else:
        history = ''
    teacher_content = history + question + tail
    return (render(tokenizer, student_content), render(tokenizer, teacher_content),
            task, len(gold_ids))


class OPDDataset(torch.utils.data.Dataset):
    """每条样本给出 student / teacher 两侧的 prompt 以及该样本的 rollout 上限。"""

    def __init__(self, tokenizer, samples):
        self.tokenizer = tokenizer
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        student_prompt, teacher_prompt, task, gold_tokens = self.samples[i]
        max_new = rollout_budget(task, gold_tokens)
                                                    
        room = MAX_SEQ_LEN - max_new - 8
        s = self.tokenizer(student_prompt, add_special_tokens=False)['input_ids'][-room:]
        t = self.tokenizer(teacher_prompt, add_special_tokens=False)['input_ids'][-room:]
        return {'student_ids': s, 'teacher_ids': t, 'max_new': max_new,
                'task': task, 'gold_tokens': gold_tokens}


def collate(features):
    """per_device_batch=1，不需要 padding，直接透传避免引入 pad 干扰 KV cache。"""
    return {'student_ids': [f['student_ids'] for f in features],
            'teacher_ids': [f['teacher_ids'] for f in features],
            'max_new': [f['max_new'] for f in features],
            'task': [f['task'] for f in features],
            'gold_tokens': [f['gold_tokens'] for f in features]}
                 


def completion_logits(model, prompt_ids, completion_ids, device):
    """只在 completion 位置取 logits。

    全序列取 logits 会是 14000×151936，bf16 就有 4GB 以上，必须用 logits_to_keep
    只保留末尾 n+1 个位置：第 -(n+1) 个位置预测 completion 的第 1 个 token，
    所以取 [-(n+1):-1] 正好是 n 个预测位。
    """
    n = len(completion_ids)
    ids = torch.tensor([prompt_ids + completion_ids], device=device)
    out = model(input_ids=ids, logits_to_keep=n + 1)
                                                              
                                   
    return out.logits[0, -(n + 1):-1, :]


def mixed_kl(student_logits, teacher_logits, beta=JSD_BETA, chunk=64):
    """generalized JSD：以 m = beta*q + (1-beta)*p 为锚的双向 KL 混合。

    返回 (jsd, forward_kl, reverse_kl)，其中
      forward_kl = KL(q_teacher ‖ p_student)   mode-covering 方向
      reverse_kl = KL(p_student ‖ q_teacher)   mode-seeking 方向
    后两个只作诊断用，不参与反传。

    按序列位置分块：一次性算会让 logp/logq/p/q/logm 五份 (n,151936) fp32 张量同时
    驻留（n=320 时约 1GB），分成 64 一块后峰值降到约 200MB，梯度照常通过每块回传。
    """
    n = student_logits.shape[0]
    log_b = float(torch.log(torch.tensor(beta)))
    log_1b = float(torch.log(torch.tensor(1.0 - beta)))
    jsd_sum = 0.0
    fwd_sum = torch.zeros((), device=student_logits.device)
    rev_sum = torch.zeros((), device=student_logits.device)
    for i in range(0, n, chunk):
        logp = F.log_softmax(student_logits[i:i + chunk].float(), dim=-1)
        logq = F.log_softmax(teacher_logits[i:i + chunk].float(), dim=-1)
        p, q = logp.exp(), logq.exp()
        logm = torch.logaddexp(logq + log_b, logp + log_1b)
        jsd_sum = jsd_sum + (beta * (q * (logq - logm)).sum(-1)
                             + (1.0 - beta) * (p * (logp - logm)).sum(-1)).sum()
        with torch.no_grad():
            fwd_sum += (q * (logq - logp)).sum(-1).sum()
            rev_sum += (p * (logp - logq)).sum(-1).sum()
    return jsd_sum / n, fwd_sum / n, rev_sum / n
                  


class OPDTrainer(__import__('transformers').Trainer):
    """on-policy 蒸馏：先用当前 student 采样，再对齐 teacher 分布。"""

    def __init__(self, teacher, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher
        self.tok = tokenizer
        self.curve = []

    @torch.no_grad()
    def rollout(self, model, prompt_ids, device, max_new):
        """用当前 student（含 LoRA）自采样，这是 on-policy 的关键——不能用基座。

        max_new 由 rollout_budget 按样本 gold 长度给出，不再是全局常量。
        """
        was_training = model.training
        model.eval()
        base = self.accelerator.unwrap_model(model)
        base.config.use_cache = True
        ids = torch.tensor([prompt_ids], device=device)
        out = base.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            do_sample=GEN_TEMPERATURE > 0,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            max_new_tokens=max_new,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id)
        base.config.use_cache = False
        if was_training:
            model.train()
        completion = out[0, len(prompt_ids):].tolist()
        hit_cap = len(completion) >= max_new and self.tok.eos_token_id not in completion
                                             
        if self.tok.eos_token_id in completion:
            completion = completion[:completion.index(self.tok.eos_token_id) + 1]
        return completion, hit_cap

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device
        s_prompt = inputs['student_ids'][0]
        t_prompt = inputs['teacher_ids'][0]
        max_new = inputs['max_new'][0]
        completion, hit_cap = self.rollout(model, s_prompt, device, max_new)
        if len(completion) < 2:
                                                  
            zero = sum(p.sum() for p in model.parameters() if p.requires_grad) * 0.0
            return (zero, None) if return_outputs else zero

                                                                 
        with torch.no_grad():
            t_logits = completion_logits(self.teacher, t_prompt, completion, device)
        s_logits = completion_logits(model, s_prompt, completion, device)
        loss, fwd, rev = mixed_kl(s_logits, t_logits)

        self.curve.append({
            'step': self.state.global_step,
            'loss': float(loss.detach()),
            'jsd': float(loss.detach()),
            'forward_kl': float(fwd),
            'reverse_kl': float(rev),
            'gen_tokens': len(completion),
            'max_new': int(max_new),
            'gold_tokens': int(inputs['gold_tokens'][0]),
            'task': inputs['task'][0],
            'hit_cap': bool(hit_cap),
        })
        return (loss, None) if return_outputs else loss
              


def smooth(xs, window=15):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        out.append(sum(xs[lo:i + 1]) / (i + 1 - lo))
    return out


def plot_curves(curve, log_history, path):
    """loss / KL 曲线画在一张图里（2x2 子图）。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    steps = [c['step'] for c in curve]
    jsd = [c['jsd'] for c in curve]
    fwd = [c['forward_kl'] for c in curve]
    rev = [c['reverse_kl'] for c in curve]
    gen = [c['gen_tokens'] for c in curve]

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle('%s  on-policy distillation (teacher=%s, mixed KL beta=%.2f)'
                 % (OUTPUT_NAME, os.path.basename(TEACHER_MODEL), JSD_BETA))

    ax[0][0].plot(steps, jsd, alpha=.25, color='C0')
    ax[0][0].plot(steps, smooth(jsd), color='C0', label='JSD (train loss)')
    ax[0][0].axhline(0.6931, ls='--', c='r', lw=.8, label='upper bound log2')
    ax[0][0].set_title('training loss = mixed KL (JSD)')
    ax[0][0].set_xlabel('step'); ax[0][0].set_ylabel('per-token JSD')
    ax[0][0].legend(); ax[0][0].grid(alpha=.3)

    ax[0][1].plot(steps, smooth(fwd), color='C1', label='forward KL(q_T||p_S)')
    ax[0][1].plot(steps, smooth(rev), color='C2', label='reverse KL(p_S||q_T)')
    ax[0][1].set_title('two KL directions (diagnostic)')
    ax[0][1].set_xlabel('step'); ax[0][1].set_ylabel('nats/token')
    ax[0][1].legend(); ax[0][1].grid(alpha=.3)

    loss_hist = [(h['step'], h['loss']) for h in log_history if 'loss' in h]
    gnorm = [(h['step'], h['grad_norm']) for h in log_history
             if h.get('grad_norm') is not None]
    if loss_hist:
        ax[1][0].plot([s for s, _ in loss_hist], [v for _, v in loss_hist],
                      marker='o', ms=3, color='C0', label='Trainer logged loss')
    if gnorm:
        twin = ax[1][0].twinx()
        twin.plot([s for s, _ in gnorm], [v for _, v in gnorm],
                  color='C3', lw=.9, label='grad_norm')
        twin.set_ylabel('grad_norm')
        twin.legend(loc='upper right')
    ax[1][0].set_title('optimizer-step loss & grad norm')
    ax[1][0].set_xlabel('step'); ax[1][0].legend(loc='upper left')
    ax[1][0].grid(alpha=.3)

    ax[1][1].plot(steps, smooth(gen), color='C4', label='rollout length')
    ax[1][1].plot(steps, smooth([c['max_new'] for c in curve]), color='gray',
                  ls='--', lw=.9, label='per-sample cap (gold x %.1f)'
                  % ROLLOUT_LENGTH_RATIO)
    hit = 100.0 * sum(1 for c in curve if c.get('hit_cap')) / max(1, len(curve))
    ax[1][1].set_title('on-policy rollout length (cap-hit %.1f%%)' % hit)
    ax[1][1].set_xlabel('step'); ax[1][1].set_ylabel('tokens')
    ax[1][1].legend(); ax[1][1].grid(alpha=.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
              


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              TrainingArguments, set_seed)

    started = time.time()
    for path in (STUDENT_MODEL, TEACHER_MODEL):
        if not os.path.isdir(path):
            raise RuntimeError('模型不存在：%s' % path)
    set_seed(SEED)
    print('OPD（纯混合 KL，无 SFT loss、无个性化权重）：'
          'student=%s  teacher=%s（base_gold 上下文，thinking=%s）'
          % (os.path.basename(STUDENT_MODEL), os.path.basename(TEACHER_MODEL),
             THINKING), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    loaded = common.load_train_rows()
    if not loaded:
        raise RuntimeError('没有可用的训练数据')
    samples, per_task = [], {}
    for task, rows in loaded.items():
        picked = common.sample_rows(task, rows, SAMPLES_PER_TASK)
        for row in picked:
            samples.append(build_pair(tokenizer, task, row))
        per_task[task] = len(picked)
    random.Random(SEED).shuffle(samples)
    print('参与训练的样本共 %d 条：%s' % (len(samples), per_task), flush=True)
    dataset = OPDDataset(tokenizer, samples)

    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL, dtype=torch.bfloat16, attn_implementation='sdpa')
    student.config.use_cache = False
    student = get_peft_model(student, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, bias='none', task_type='CAUSAL_LM'))
                                             
                                                      
    student.enable_input_require_grads()
    student.print_trainable_parameters()

    local_rank = max(0, int(os.environ.get('LOCAL_RANK', '0')))
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, dtype=torch.bfloat16, attn_implementation='sdpa')
    teacher.config.use_cache = False
    teacher.eval().requires_grad_(False)
    teacher.to('cuda:%d' % local_rank)

    args = TrainingArguments(
        output_dir=os.path.join(LOG_DIR, OUTPUT_NAME + '_checkpoints'),
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type='cosine',
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=10,
        save_strategy=('steps' if SAVE_STEPS else 'no'),
        save_steps=(__import__('math').gcd(*SAVE_STEPS) if len(SAVE_STEPS) > 1 else (SAVE_STEPS[0] if SAVE_STEPS else 500)),
        save_total_limit=None,
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=SEED,
        ddp_find_unused_parameters=False,
    )
    trainer = OPDTrainer(teacher=teacher, tokenizer=tokenizer, model=student,
                         args=args, train_dataset=dataset, data_collator=collate)
    train_output = trainer.train()
    train_seconds = time.time() - started
                   
    if not trainer.is_world_process_zero():
        return

    fig_path = os.path.join(FIG_DIR, '%s_curves.png' % OUTPUT_NAME)
    plot_curves(trainer.curve, trainer.state.log_history, fig_path)
    print('曲线已保存：%s' % fig_path, flush=True)

    print('训练完成，开始合并 LoRA 权重', flush=True)
    merged = trainer.accelerator.unwrap_model(student).merge_and_unload()
    merged.config.use_cache = True
    output_dir = os.path.join(TRAINED_DIR, OUTPUT_NAME)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    if not SAVE_STEPS:
        shutil.rmtree(args.output_dir, ignore_errors=True)
    else:
        print('保留中间 checkpoint 供逐档评测：%s' % args.output_dir, flush=True)

    tail = trainer.curve[-max(1, len(trainer.curve) // 10):]
    record = {
        'student_model': os.path.basename(STUDENT_MODEL),
        'teacher_model': os.path.basename(TEACHER_MODEL),
        'output_model': OUTPUT_NAME,
        'output_dir': output_dir,
        'method': 'cross-model on-policy distillation; teacher context = base_gold '
                  '(full history + gold answer as reference + rewrite hint); '
                  'loss = generalized JSD (mixed forward/reverse KL) only, '
                  'no SFT term, no personalization weights',
        'thinking': {'student': THINKING, 'teacher': THINKING},
        'reference_head': REFERENCE_HEAD,
        'reference_hint': REFERENCE_HINT,
        'jsd_beta': JSD_BETA,
        'rollout_length_ratio': ROLLOUT_LENGTH_RATIO,
        'rollout_task_cap': ROLLOUT_TASK_CAP,
        'rollout_min_tokens': ROLLOUT_MIN_TOKENS,
        'rollout_hard_cap': ROLLOUT_HARD_CAP,
        'rollout_cap_hit_ratio': round(
            sum(1 for c in trainer.curve if c.get('hit_cap')) / max(1, len(trainer.curve)), 4),
        'gen_temperature': GEN_TEMPERATURE,
        'user_token_budget': USER_TOKEN_BUDGET,
        'max_seq_len': MAX_SEQ_LEN,
        'gold_token_budget': GOLD_TOKEN_BUDGET,
        'dataset': 'LongLaMP (user-based train)',
        'samples_per_task': per_task,
        'total_samples': len(dataset),
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'per_device_batch_size': PER_DEVICE_BATCH,
        'gradient_accumulation_steps': GRAD_ACCUM,
        'max_grad_norm': MAX_GRAD_NORM,
        'warmup_ratio': WARMUP_RATIO,
        'lora': {'r': LORA_R, 'alpha': LORA_ALPHA, 'dropout': LORA_DROPOUT,
                 'target_modules': LORA_TARGETS},
        'seed': SEED,
        'train_metrics': train_output.metrics,
        'final_tail_mean': {
            'jsd': round(sum(c['jsd'] for c in tail) / len(tail), 4),
            'forward_kl': round(sum(c['forward_kl'] for c in tail) / len(tail), 4),
            'reverse_kl': round(sum(c['reverse_kl'] for c in tail) / len(tail), 4),
            'gen_tokens': round(sum(c['gen_tokens'] for c in tail) / len(tail), 1),
        },
        'curve': trainer.curve,
        'log_history': trainer.state.log_history,
        'figure': fig_path,
        'train_seconds': round(train_seconds, 1),
        'train_elapsed': evalbase.format_duration(train_seconds),
        'created_date': datetime.now().strftime('%Y%m%d'),
    }
    log_path = os.path.join(LOG_DIR, '%s_train.json' % OUTPUT_NAME)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as output:
        json.dump(record, output, ensure_ascii=False, indent=2)
    print('模型已保存：%s' % output_dir, flush=True)
    print('训练记录：%s（耗时 %s）'
          % (log_path, record['train_elapsed']), flush=True)


if __name__ == '__main__':
    main()






