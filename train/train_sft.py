"""LongLaMP 训练集上的 LoRA SFT：训练数据是 (question, answer) pair。

question 直接用样本的 input，不拼任何用户历史；answer 是 gold output。
只在 answer 的 token 上算 loss，prompt 部分标签置为 -100。

按 CLAUDE.md 约定：直接 `python train_sft.py` 运行，不接收命令行参数。
合并 LoRA 后的模型覆盖写入 TrainModels/<模型名>-SFT，
训练记录写入 trainlogs/<模型名>-SFT_train.json。
"""
import glob
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime

                                                      
                                       
                                                        
                                                 
if int(os.environ.get('LOCAL_RANK', '-1')) < 0:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HUAYI_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(HUAYI_DIR, 'eval'))
import eval_lamp_sparse as evalbase

                                                           
                                                      
BASE_MODEL = os.path.join(HUAYI_DIR, 'models',
                          os.environ.get('SFT_BASE_MODEL', 'Llama-3.2-3B-Instruct'))
TRAINED_DIR = os.path.join(HUAYI_DIR, 'TrainModels')
LOG_DIR = os.path.join(HUAYI_DIR, 'logs', 'train')

SAMPLES_PER_TASK = 2000                        
MAX_SEQ_LEN = 4096                                  
ANSWER_TOKEN_BUDGET = 512                            
EPOCHS = 1
LEARNING_RATE = 1e-4
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 8
                                                
_DDP_WORLD = int(os.environ.get('WORLD_SIZE', '1'))
if _DDP_WORLD > 1:
    GRAD_ACCUM = max(1, GRAD_ACCUM // _DDP_WORLD)
SEED = 20260904
                                                                   
                                                     
                                                   
                                           
THINKING = os.environ.get('SFT_THINKING', 'off')
if THINKING not in ('off', 'on'):
    raise RuntimeError('SFT_THINKING 只能取 off/on：%s' % THINKING)

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj']


def load_train_rows():
    """读取所选任务 user-based 划分的 train parquet。

    通过环境变量 TRAIN_TASKS 自由选择数据集/任务：
      默认（不设）保持历史口径 = 'longlamp'（abstract_generation/product_review/
      topic_writing 三任务）；
      'all'      = 同时训练全部 6 个任务（+ Amazon book/movie/cd_review）；
      'amazon'   = 仅三个 Amazon 任务；
      也可给具体任务名，逗号分隔。缺 parquet 的任务自动跳过。
    """
    import pyarrow.parquet as pq
    tasks = evalbase.resolve_tasks(os.environ.get('TRAIN_TASKS'), 'longlamp')
    loaded = {}
    for task in tasks:
        files = sorted(glob.glob(os.path.join(
            evalbase.task_dir(task), 'train-*.parquet')))
        if not files:
            print('跳过 %s：train parquet 缺失' % task, flush=True)
            continue
        rows = []
        for path in files:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                rows.extend(batch.to_pylist())
        loaded[task] = rows
        print('%s: %d 条训练样本' % (task, len(rows)), flush=True)
    return loaded


def sample_rows(task, rows, limit=SAMPLES_PER_TASK):
    """固定种子下采样，避免每次训练用到不同子集"""
    if len(rows) <= limit:
        return rows
    return random.Random('%s-%d' % (task, SEED)).sample(rows, limit)


def build_question(task, row):
    """SFT 的输入只有问题本身，不带任何用户历史"""
    return row['input']


class PairDataset(torch.utils.data.Dataset):
    """(question, answer) 监督数据：prompt 段标签置 -100，只在 answer 上算 loss。"""

    def __init__(self, tokenizer, samples):
        self.tokenizer = tokenizer
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def _render(self, question):
        """渲染对话模板；不认 enable_thinking 的模板自动退回无参渲染。"""
        messages = [{'role': 'system', 'content': evalbase.SYSTEM_PROMPT},
                    {'role': 'user', 'content': question}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=(THINKING == 'on'))
        except (TypeError, ValueError):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    def __getitem__(self, i):
        question, answer = self.samples[i]
        prompt = self._render(question)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
        answer_ids = self.tokenizer(
            answer, add_special_tokens=False,
            truncation=True, max_length=ANSWER_TOKEN_BUDGET)['input_ids']
        answer_ids = answer_ids + [self.tokenizer.eos_token_id]
                                           
        room = MAX_SEQ_LEN - len(answer_ids)
        if len(prompt_ids) > room:
            prompt_ids = prompt_ids[-room:]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + list(answer_ids)
        return {'input_ids': input_ids, 'labels': labels}


def collate(features, pad_token_id):
    """动态 padding，短样本不用补到 MAX_SEQ_LEN"""
    width = max(len(f['input_ids']) for f in features)
    batch = {'input_ids': [], 'attention_mask': [], 'labels': []}
    for f in features:
        pad = width - len(f['input_ids'])
        batch['input_ids'].append(f['input_ids'] + [pad_token_id] * pad)
        batch['attention_mask'].append([1] * len(f['input_ids']) + [0] * pad)
        batch['labels'].append(f['labels'] + [-100] * pad)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def build_samples(loaded, question_builder):
    """按任务采样并构造 (question, answer) pair"""
    samples, per_task = [], {}
    for task, rows in loaded.items():
        picked = sample_rows(task, rows)
        for row in picked:
            samples.append((question_builder(task, row), row['output']))
        per_task[task] = len(picked)
    random.Random(SEED).shuffle(samples)
    return samples, per_task


def train_lora(suffix, method, question_builder):
    """公共训练流程：LoRA 微调 -> 合并权重 -> 保存模型和训练记录"""
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, set_seed)

    started = time.time()
    if not os.path.isdir(BASE_MODEL):
        raise RuntimeError('基座模型不存在：%s' % BASE_MODEL)
    set_seed(SEED)

    base_name = os.path.basename(BASE_MODEL)
    output_name = evalbase.with_dataset_suffix(
        base_name.replace('-Instruct', '') + suffix,
        os.environ.get('TRAIN_TASKS'))
    output_dir = os.path.join(TRAINED_DIR, output_name)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TRAINED_DIR, exist_ok=True)
    print('训练方法：%s；输出模型：%s；思考=%s'
          % (method, output_name, THINKING), flush=True)

    loaded = load_train_rows()
    if not loaded:
        raise RuntimeError('没有可用的训练数据')
    samples, per_task = build_samples(loaded, question_builder)
    print('参与训练的样本共 %d 条：%s' % (len(samples), per_task), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = PairDataset(tokenizer, samples)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    model.config.use_cache = False
    lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                      target_modules=LORA_TARGETS, bias='none',
                      task_type='CAUSAL_LM')
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=os.path.join(LOG_DIR, output_name + '_checkpoints'),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type='cosine',
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy='no',
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=SEED,
                                               
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset,
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
        'method': method,
        'thinking': THINKING,
        'dataset': 'LongLaMP (user-based train)',
        'samples_per_task': per_task,
        'total_samples': len(dataset),
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'per_device_batch_size': PER_DEVICE_BATCH,
        'gradient_accumulation_steps': GRAD_ACCUM,
        'max_seq_len': MAX_SEQ_LEN,
        'answer_token_budget': ANSWER_TOKEN_BUDGET,
        'lora': {'r': LORA_R, 'alpha': LORA_ALPHA, 'dropout': LORA_DROPOUT,
                 'target_modules': LORA_TARGETS},
        'seed': SEED,
        'train_metrics': train_output.metrics,
        'log_history': trainer.state.log_history,
        'train_seconds': round(train_seconds, 1),
        'train_elapsed': evalbase.format_duration(train_seconds),
        'total_seconds': round(time.time() - started, 1),
        'total_elapsed': evalbase.format_duration(time.time() - started),
        'created_date': datetime.now().strftime('%Y%m%d'),
    }
    log_path = os.path.join(LOG_DIR, '%s_train.json' % output_name)
    with open(log_path, 'w', encoding='utf-8') as output:
        json.dump(record, output, ensure_ascii=False, indent=2)
    print('模型已保存：%s' % output_dir, flush=True)
    print('训练记录已覆盖写入：%s（耗时 %s）' %
          (log_path, record['total_elapsed']), flush=True)


def main():
    train_lora(suffix='-SFT', method='LoRA SFT on (question, answer) pairs',
               question_builder=build_question)


if __name__ == '__main__':
    main()
