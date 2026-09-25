"""LongLaMP 训练集上的 PerCE 训练（arXiv:2603.06595，
"Rethinking Personalization in Large Language Models at the Token Level"）。

在 train_contextsft.py 的基础上只做一处改动：把标准 SFT 的均匀 token 权重换成
PerContrast 估计的偏好权重，最终 SFT 损失就是论文的 PerCE loss。

论文机制（Eq.2 / Eq.6 / Eq.7 / Eq.8）：
  PIR(y_i) = log P_θ(y_i | p_u, x, y_<i) - log P_θ(y_i | x, y_<i)
  w(y_i)   = clip(PIR(y_i), m, M)
  PerCE    = -(1/n) Σ_i w(y_i) · log P_θ(y_i | p_u, x, y_<i)

PIR 由被训练模型自己在线估计（论文的 online-EM / bootstrap：E 步 detach 估权重，
M 步用该权重做加权交叉熵），因此不需要任何外部冻结模型。

与 contextSFT 的唯一差别就是 token 权重（PerCE 的 PIR 加权 vs 均匀权重）：history
同样是**全量用户历史**，按数据集原始顺序拼接，与 eval_lamp_base.py 的 full user
context 口径一致，不做检索或重排。这样 contextSFT / NextQuill / PerCE / PSOPD 四者
的输入分布完全相同，消融只剩"权重怎么算"这一个变量。
（论文原设定是 Contriever top-4 的 RAG；本项目为了与其余基线严格可比改成全量历史。）

按 CLAUDE.md 约定：直接 `python train_PerCE.py` 运行，不接收命令行参数。
合并 LoRA 后的模型覆盖写入 TrainModels/Llama-3.2-3B-PerCE。
"""
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime

                                                 
                                            
                                                        
                                                 
if int(os.environ.get('LOCAL_RANK', '-1')) < 0:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HUAYI_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(HUAYI_DIR, 'eval'))
sys.path.insert(0, BASE_DIR)
import eval_lamp_base as evalfull
import eval_lamp_sparse as evalbase
import train_sft as common

OUTPUT_SUFFIX = '-PerCE'
CLIP_MIN = 0.8                                     
CLIP_MAX = 5.0                                     
                                                    
                                                     
MAX_STEPS = int(os.environ.get('PERCE_MAX_STEPS', '300'))


def build_context_question(task, row):
    """全量用户历史 + 问题，与 train_contextsft.py / eval_lamp_base.py 完全一致。"""
    prefix, question, _ = evalfull.full_context(
        task, row['input'], row.get('profile') or [])
    return prefix + question


class ContrastPairDataset(torch.utils.data.Dataset):
    """每条样本给出两条序列：带全量 history 的和只有问题的，answer token 相同。

    PIR 需要对比 (p_u, x, y_<i) 与 (x, y_<i) 两种条件下同一个 gold token 的
    对数概率，所以必须构造这两条输入。
    """

    def __init__(self, tokenizer, samples):
        self.tokenizer = tokenizer
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def _encode(self, question, answer_ids):
        prompt = self.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': evalbase.SYSTEM_PROMPT},
             {'role': 'user', 'content': question}],
            tokenize=False, add_generation_prompt=True)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
        room = common.MAX_SEQ_LEN - len(answer_ids)
        if len(prompt_ids) > room:
            prompt_ids = prompt_ids[-room:]
        return (prompt_ids + answer_ids,
                [-100] * len(prompt_ids) + list(answer_ids))

    def __getitem__(self, i):
        question_with_history, question_only, answer = self.samples[i]
        answer_ids = self.tokenizer(
            answer, add_special_tokens=False, truncation=True,
            max_length=common.ANSWER_TOKEN_BUDGET)['input_ids']
        answer_ids = answer_ids + [self.tokenizer.eos_token_id]
        ids_p, labels_p = self._encode(question_with_history, answer_ids)
        ids_x, labels_x = self._encode(question_only, answer_ids)
        return {'input_ids': ids_p, 'labels': labels_p,
                'input_ids_no_persona': ids_x, 'labels_no_persona': labels_x}


def collate(features, pad_token_id):
    """两条序列各自动态 padding"""
    batch = {}
    for suffix in ('', '_no_persona'):
        ids_key, labels_key = 'input_ids' + suffix, 'labels' + suffix
        width = max(len(f[ids_key]) for f in features)
        ids, mask, labels = [], [], []
        for f in features:
            pad = width - len(f[ids_key])
            ids.append(f[ids_key] + [pad_token_id] * pad)
            mask.append([1] * len(f[ids_key]) + [0] * pad)
            labels.append(f[labels_key] + [-100] * pad)
        batch['input_ids' + suffix] = torch.tensor(ids, dtype=torch.long)
        batch['attention_mask' + suffix] = torch.tensor(mask, dtype=torch.long)
        batch['labels' + suffix] = torch.tensor(labels, dtype=torch.long)
    return batch


def answer_logprobs(model, input_ids, attention_mask, labels):
    """gold answer token 上的 log P_θ(y_i | ...)。

    causal LM 第 t 个位置预测第 t+1 个 token，所以标签左移一位对齐。
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    keep = shifted_labels != -100
    selected = shifted_logits[keep]
    targets = shifted_labels[keep]
    log_probs = torch.log_softmax(selected.float(), dim=-1)
    return log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)


def build_trainer_class():
    from transformers import Trainer

    class PerCETrainer(Trainer):
        """PerCE loss：只有一次额外前向（不带 persona 的那次）。

        带 persona 的前向本身就是 M 步要用的，PIR 直接复用它 detach 后的结果，
        对应论文里 "only a single additional forward pass"。

        归一化必须用 num_items_in_batch（整个梯度累积窗口的 answer token 总数）：
        Llama 的 forward 带 **kwargs，Trainer 因此认为损失已按 token 数归一化，
        不会再除以 gradient_accumulation_steps。若这里只按当前 microbatch 求均值，
        实际梯度会放大 8 倍（= GRAD_ACCUM），和 contextSFT 基线不可比。
        """

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None, **kwargs):
            log_p_persona = answer_logprobs(
                model, inputs['input_ids'], inputs['attention_mask'],
                inputs['labels'])
            with torch.no_grad():
                log_p_plain = answer_logprobs(
                    model, inputs['input_ids_no_persona'],
                    inputs['attention_mask_no_persona'],
                    inputs['labels_no_persona'])
                pir = log_p_persona.detach() - log_p_plain               
                weights = pir.clamp(CLIP_MIN, CLIP_MAX)                       
            token_ce = -log_p_persona
            denominator = num_items_in_batch or token_ce.numel()
            loss = (weights * token_ce).sum() / denominator               
            if self.state.global_step % 10 == 0:
                above = float((pir > CLIP_MIN).float().mean())
                self.log({'perce_loss_token_mean':
                              round(float((weights * token_ce).mean()), 4),
                          'plain_ce_loss': round(float(token_ce.mean()), 4),
                          'mean_pir': round(float(pir.mean()), 4),
                          'mean_weight': round(float(weights.mean()), 4),
                          'personal_token_ratio': round(above, 4)})
            return (loss, None) if return_outputs else loss


    return PerCETrainer


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              TrainingArguments, set_seed)

    started = time.time()
    set_seed(common.SEED)

    base_name = os.path.basename(common.BASE_MODEL)
    output_name = evalbase.with_dataset_suffix(
        base_name.replace('-Instruct', '') + OUTPUT_SUFFIX,
        os.environ.get('TRAIN_TASKS'))
    output_dir = os.path.join(common.TRAINED_DIR, output_name)
    os.makedirs(common.LOG_DIR, exist_ok=True)
    os.makedirs(common.TRAINED_DIR, exist_ok=True)
    print('训练方法：PerCE（PerContrast 估计的 token 偏好权重加权交叉熵）', flush=True)
    print('输出模型：%s；显卡：CUDA_VISIBLE_DEVICES=%s'
          % (output_name, os.environ.get('CUDA_VISIBLE_DEVICES', 'all')),
          flush=True)

    loaded = common.load_train_rows()
    if not loaded:
        raise RuntimeError('没有可用的训练数据')

    samples, per_task = [], {}
    for task, rows in loaded.items():
        chosen = common.sample_rows(task, rows)
        for row in chosen:
                                                           
                                                        
            samples.append((build_context_question(task, row),
                            row['input'], row['output']))
        per_task[task] = len(chosen)
    random.Random(common.SEED).shuffle(samples)
    print('参与训练的样本共 %d 条：%s' % (len(samples), per_task), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(common.BASE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = ContrastPairDataset(tokenizer, samples)

    model = AutoModelForCausalLM.from_pretrained(
        common.BASE_MODEL, dtype=torch.bfloat16, attn_implementation='sdpa')
    model.config.use_cache = False
    lora = LoraConfig(r=common.LORA_R, lora_alpha=common.LORA_ALPHA,
                      lora_dropout=common.LORA_DROPOUT,
                      target_modules=common.LORA_TARGETS, bias='none',
                      task_type='CAUSAL_LM')
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    args = TrainingArguments(
        output_dir=os.path.join(common.LOG_DIR, output_name + '_checkpoints'),
        num_train_epochs=common.EPOCHS,
                                                      
                                                                            
        max_steps=MAX_STEPS,
        per_device_train_batch_size=common.PER_DEVICE_BATCH,
        gradient_accumulation_steps=common.GRAD_ACCUM,
        learning_rate=common.LEARNING_RATE,
        lr_scheduler_type='cosine',
        warmup_ratio=0.04,
        logging_steps=10,
        save_strategy='no',
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=common.SEED,
    )
    trainer = build_trainer_class()(
        model=model, args=args, train_dataset=dataset,
        data_collator=lambda f: collate(f, tokenizer.pad_token_id))
    train_output = trainer.train()
    train_seconds = time.time() - started

                                                
                                                    
    if not trainer.is_world_process_zero():
        return

    print('训练完成，开始合并 LoRA 权重', flush=True)
    merged = trainer.accelerator.unwrap_model(model).merge_and_unload()
    merged.config.use_cache = True
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    shutil.rmtree(args.output_dir, ignore_errors=True)

    record = {
        'base_model': base_name,
        'output_model': output_name,
        'output_dir': output_dir,
        'method': 'PerCE: PerContrast(PIR)-weighted cross entropy',
        'paper': 'arXiv:2603.06595',
        'weighting': 'w_i = clip(log P(y_i|history,x,y<i) - log P(y_i|x,y<i), m, M)',
        'weights_estimated_by': 'the model being trained (online EM, no frozen model)',
        'history_source': 'full user history (dataset order, same as eval_lamp_base)',
        'clip_min': CLIP_MIN,
        'clip_max': CLIP_MAX,
        'dataset': 'LongLaMP (user-based train)',
        'samples_per_task': per_task,
        'total_samples': len(dataset),
        'max_steps': MAX_STEPS,
        'epochs': common.EPOCHS,
        'learning_rate': common.LEARNING_RATE,
        'per_device_batch_size': common.PER_DEVICE_BATCH,
        'gradient_accumulation_steps': common.GRAD_ACCUM,
        'max_seq_len': common.MAX_SEQ_LEN,
        'answer_token_budget': common.ANSWER_TOKEN_BUDGET,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'all'),
        'lora': {'r': common.LORA_R, 'alpha': common.LORA_ALPHA,
                 'dropout': common.LORA_DROPOUT,
                 'target_modules': common.LORA_TARGETS},
        'seed': common.SEED,
        'train_metrics': train_output.metrics,
        'log_history': trainer.state.log_history,
        'train_seconds': round(train_seconds, 1),
        'train_elapsed': evalbase.format_duration(train_seconds),
        'total_seconds': round(time.time() - started, 1),
        'total_elapsed': evalbase.format_duration(time.time() - started),
        'created_date': datetime.now().strftime('%Y%m%d'),
    }
    log_path = os.path.join(common.LOG_DIR, '%s_train.json' % output_name)
    with open(log_path, 'w', encoding='utf-8') as output:
        json.dump(record, output, ensure_ascii=False, indent=2)
    print('模型已保存：%s' % output_dir, flush=True)
    print('训练记录已覆盖写入：%s（耗时 %s）' %
          (log_path, record['total_elapsed']), flush=True)


if __name__ == '__main__':
    main()
