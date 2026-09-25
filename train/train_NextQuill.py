"""LongLaMP 训练集上的 NextQuill 训练（arXiv:2506.02368）。

在 train_contextsft.py 的基础上（训练对仍是 (history + question, answer)）做两处改动：

1. 偏好加权的 next-token 损失（论文 Eq.4~6）：用冻结的 Llama-3.2-3B-contextSFT
   估计每个 gold token 的 data-side causal effect
       DCE_t = p_frozen(y_t | x, h, y_<t) - p_frozen(y_t | x, ∅, y_<t)
   DCE_t > δ 的 token 判为 preference-driven，权重 λ，否则权重 ε。
2. 因果偏好对齐损失 MCE（论文 Eq.3、Eq.7）：用被训练模型自己带历史与不带历史的
   两次前向之差作为 model-side causal effect
       MCE_t = f_θ(x, h, y_<t) - f_θ(x, ∅, y_<t)
   再对 MCE_t 做同样加权的交叉熵，把模型内部的偏好效应对齐到 gold token 上。

总损失（论文 Eq.8）：L = L_n + α · L_p

按 CLAUDE.md 约定：直接 `python train_NextQuill.py` 运行，不接收命令行参数。
合并 LoRA 后的模型覆盖写入 TrainModels/Llama-3.2-3B-NextQuill。
"""
import json
import os
import shutil
import sys
import time
from datetime import datetime

                                                        
                                                 
if int(os.environ.get('LOCAL_RANK', '-1')) < 0:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
                                                                   
                                        
                                                 
                                                        
                                                           
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HUAYI_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(HUAYI_DIR, 'eval'))
sys.path.insert(0, BASE_DIR)
import eval_lamp_base as evalfull
import eval_lamp_sparse as evalbase
import train_sft as common

                                            
                                                     
FROZEN_MODEL = os.path.join(
    common.TRAINED_DIR,
    os.environ.get('NEXTQUILL_FROZEN_MODEL',
                   os.path.basename(common.BASE_MODEL).replace('-Instruct', '')
                   + '-contextSFT'))
OUTPUT_SUFFIX = os.environ.get('NEXTQUILL_OUTPUT_SUFFIX', '-NextQuill')

DCE_THRESHOLD = 0.0                                       
PREFERENCE_WEIGHT = 1.0                                   
OTHER_WEIGHT = 0.1                         
ALIGNMENT_ALPHA = 1.0                       

                   
                                                                
                                                                   
                                         
LEARNING_RATE = float(os.environ.get('NEXTQUILL_LR', str(common.LEARNING_RATE)))
MAX_STEPS = int(os.environ.get('NEXTQUILL_MAX_STEPS', '0'))


def build_context_question(task, row):
    """与 train_contextsft.py 完全一致的带历史输入"""
                                                     
                       
    prefix, question, _ = evalfull.full_context(
        task, row['input'], row.get('profile') or [])
    content = prefix + question
    return content


class CausalPairDataset(torch.utils.data.Dataset):
    """同时给出带历史和不带历史两条序列，answer token 完全相同。

    DCE 与 MCE 都需要对比 (x, h, y_<t) 与 (x, ∅, y_<t) 两种条件下对同一个
    gold token 的预测，因此每个样本要构造两条输入。
    """

    def __init__(self, tokenizer, samples):
        self.tokenizer = tokenizer
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def _encode(self, question, answer_ids):
                                                                 
                                                         
        messages = [{'role': 'system', 'content': evalbase.SYSTEM_PROMPT},
                    {'role': 'user', 'content': question}]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=(common.THINKING == 'on'))
        except (TypeError, ValueError):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
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
        ids_h, labels_h = self._encode(question_with_history, answer_ids)
        ids_x, labels_x = self._encode(question_only, answer_ids)
        return {'input_ids': ids_h, 'labels': labels_h,
                'input_ids_no_history': ids_x, 'labels_no_history': labels_x}


def collate(features, pad_token_id):
    """两条序列各自动态 padding"""
    batch = {}
    for ids_key, labels_key in (('input_ids', 'labels'),
                                ('input_ids_no_history', 'labels_no_history')):
        width = max(len(f[ids_key]) for f in features)
        ids, mask, labels = [], [], []
        for f in features:
            pad = width - len(f[ids_key])
            ids.append(f[ids_key] + [pad_token_id] * pad)
            mask.append([1] * len(f[ids_key]) + [0] * pad)
            labels.append(f[labels_key] + [-100] * pad)
        suffix = '_no_history' if 'no_history' in ids_key else ''
        batch['input_ids' + suffix] = torch.tensor(ids, dtype=torch.long)
        batch['attention_mask' + suffix] = torch.tensor(mask, dtype=torch.long)
        batch['labels' + suffix] = torch.tensor(labels, dtype=torch.long)
    return batch


def answer_logits(model, input_ids, attention_mask, labels):
    """取出 answer 位置上对应 gold token 的 logits 序列。

    causal LM 的第 t 个位置预测第 t+1 个 token，所以标签要左移一位对齐。
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    keep = shifted_labels != -100
    selected = shifted_logits[keep]                                   
    targets = shifted_labels[keep]
    return selected, targets


class NextQuillTrainer:
    """封装 DCE 权重与两项损失，供自定义 Trainer 调用。"""

    def __init__(self, frozen_model):
        self.frozen_model = frozen_model

    @torch.no_grad()
    def preference_weights(self, inputs):
        """论文 Eq.5 + Eq.4：冻结模型估计 DCE，再离散成 λ / ε 权重。"""
        logits_h, targets = answer_logits(
            self.frozen_model, inputs['input_ids'],
            inputs['attention_mask'], inputs['labels'])
        logits_x, _ = answer_logits(
            self.frozen_model, inputs['input_ids_no_history'],
            inputs['attention_mask_no_history'], inputs['labels_no_history'])
        probs_h = torch.softmax(logits_h.float(), dim=-1).gather(
            1, targets.unsqueeze(1)).squeeze(1)
        probs_x = torch.softmax(logits_x.float(), dim=-1).gather(
            1, targets.unsqueeze(1)).squeeze(1)
        dce = probs_h - probs_x
        weights = torch.where(dce > DCE_THRESHOLD,
                              torch.full_like(dce, PREFERENCE_WEIGHT),
                              torch.full_like(dce, OTHER_WEIGHT))
        return weights, dce

    def losses(self, model, inputs, weights):
        """L_n（Eq.6，加权 next-token）与 L_p（Eq.7，MCE 对齐）"""
        logits_h, targets = answer_logits(
            model, inputs['input_ids'],
            inputs['attention_mask'], inputs['labels'])
        logits_x, _ = answer_logits(
            model, inputs['input_ids_no_history'],
            inputs['attention_mask_no_history'], inputs['labels_no_history'])
        token_ce = torch.nn.functional.cross_entropy(
            logits_h.float(), targets, reduction='none')
                                                      
        causal_effect = logits_h.float() - logits_x.float()
        alignment_ce = torch.nn.functional.cross_entropy(
            causal_effect, targets, reduction='none')
        denominator = weights.sum().clamp_min(1e-6)
        weighted_loss = (weights * token_ce).sum() / denominator
        alignment_loss = (weights * alignment_ce).sum() / denominator
        return weighted_loss, alignment_loss


def build_trainer_class(helper):
    from transformers import Trainer

    class NextQuillHFTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False,
                         **kwargs):
            weights, dce = helper.preference_weights(inputs)
            weighted_loss, alignment_loss = helper.losses(model, inputs, weights)
            loss = weighted_loss + ALIGNMENT_ALPHA * alignment_loss
            if self.state.global_step % 10 == 0:
                driven = float((weights > OTHER_WEIGHT).float().mean())
                self.log({'weighted_next_token_loss': round(float(weighted_loss), 4),
                          'causal_alignment_loss': round(float(alignment_loss), 4),
                          'preference_driven_ratio': round(driven, 4),
                          'mean_dce': round(float(dce.mean()), 6)})
            return (loss, None) if return_outputs else loss

    return NextQuillHFTrainer


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              TrainingArguments, set_seed)

    started = time.time()
    if not os.path.isdir(FROZEN_MODEL):
        raise RuntimeError('DCE 需要的冻结模型不存在：%s' % FROZEN_MODEL)
    set_seed(common.SEED)

    base_name = os.path.basename(common.BASE_MODEL)
    output_name = evalbase.with_dataset_suffix(
        base_name.replace('-Instruct', '') + OUTPUT_SUFFIX,
        os.environ.get('TRAIN_TASKS'))
    output_dir = os.path.join(common.TRAINED_DIR, output_name)
    os.makedirs(common.LOG_DIR, exist_ok=True)
    os.makedirs(common.TRAINED_DIR, exist_ok=True)
    print('训练方法：NextQuill（DCE 加权 + MCE 对齐）；输出模型：%s' % output_name,
          flush=True)
    print('DCE 冻结模型：%s' % FROZEN_MODEL, flush=True)

    loaded = common.load_train_rows()
    if not loaded:
        raise RuntimeError('没有可用的训练数据')
    samples, per_task = [], {}
    for task, rows in loaded.items():
        picked = common.sample_rows(task, rows)
        for row in picked:
            samples.append((build_context_question(task, row),
                            common.build_question(task, row), row['output']))
        per_task[task] = len(picked)
    import random
    random.Random(common.SEED).shuffle(samples)
    print('参与训练的样本共 %d 条：%s' % (len(samples), per_task), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(common.BASE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = CausalPairDataset(tokenizer, samples)

                                                    
    _local_rank = os.environ.get('LOCAL_RANK', '0')
    frozen = AutoModelForCausalLM.from_pretrained(
        FROZEN_MODEL, dtype=torch.bfloat16,
        attn_implementation='sdpa').to('cuda:%s' % _local_rank)
    frozen.eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)

    model = AutoModelForCausalLM.from_pretrained(
        common.BASE_MODEL, dtype=torch.bfloat16, attn_implementation='sdpa')
    model.config.use_cache = False
    lora = LoraConfig(r=common.LORA_R, lora_alpha=common.LORA_ALPHA,
                      lora_dropout=common.LORA_DROPOUT,
                      target_modules=common.LORA_TARGETS, bias='none',
                      task_type='CAUSAL_LM')
    model = get_peft_model(model, lora)
                                                
                                                      
                                            
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=os.path.join(common.LOG_DIR, output_name + '_checkpoints'),
        num_train_epochs=common.EPOCHS,
        max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
        per_device_train_batch_size=common.PER_DEVICE_BATCH,
        gradient_accumulation_steps=common.GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type='cosine',
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=10,
        save_strategy='no',
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=common.SEED,
    )
    print('学习率=%.2e  max_steps=%s  任务=%s' % (
        LEARNING_RATE, MAX_STEPS if MAX_STEPS > 0 else '(按 epoch)',
        os.environ.get('TRAIN_TASKS', 'longlamp(默认)')), flush=True)
    trainer_class = build_trainer_class(NextQuillTrainer(frozen))
    trainer = trainer_class(
        model=model, args=args, train_dataset=dataset,
        data_collator=lambda f: collate(f, tokenizer.pad_token_id))
    train_output = trainer.train()
    train_seconds = time.time() - started

                                                
                                                    
    if not trainer.is_world_process_zero():
        return

    print('训练完成，开始合并 LoRA 权重', flush=True)
    del frozen
    torch.cuda.empty_cache()
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
        'method': 'NextQuill: DCE-weighted next-token loss + MCE-DCE alignment',
        'paper': 'arXiv:2506.02368',
        'frozen_model_for_dce': os.path.basename(FROZEN_MODEL),
        'dataset': 'LongLaMP (user-based train)',
        'samples_per_task': per_task,
        'total_samples': len(dataset),
        'epochs': common.EPOCHS,
        'max_steps': MAX_STEPS if MAX_STEPS > 0 else None,
        'learning_rate': LEARNING_RATE,
        'per_device_batch_size': common.PER_DEVICE_BATCH,
        'gradient_accumulation_steps': common.GRAD_ACCUM,
        'max_seq_len': common.MAX_SEQ_LEN,
        'answer_token_budget': common.ANSWER_TOKEN_BUDGET,
        'dce_threshold': DCE_THRESHOLD,
        'preference_weight_lambda': PREFERENCE_WEIGHT,
        'other_weight_epsilon': OTHER_WEIGHT,
        'alignment_alpha': ALIGNMENT_ALPHA,
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
