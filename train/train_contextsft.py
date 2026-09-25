"""LongLaMP 训练集上的 LoRA ContextSFT：训练数据是 (history + question, answer) pair。

history 的拼接方式与 eval_lamp_base.py 完全一致——按数据集原始顺序纳入用户历史，
不做检索或重排，这样训练输入分布和 base 评测时的输入分布对齐。

按 CLAUDE.md 约定：直接 `python train_contextsft.py` 运行，不接收命令行参数。
合并 LoRA 后的模型覆盖写入 TrainModels/<模型名>-contextSFT。
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HUAYI_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(HUAYI_DIR, 'eval'))
sys.path.insert(0, BASE_DIR)
import eval_lamp_base as evalfull
import train_sft as common


def build_context_question(task, row):
    """把用户历史拼在问题前面，与 base 评测的 full user context 一致"""
                                                     
                       
    prefix, question, _ = evalfull.full_context(
        task, row['input'], row.get('profile') or [])
    content = prefix + question
    return content


def main():
    common.train_lora(
        suffix='-contextSFT',
        method='LoRA ContextSFT on (history + question, answer) pairs',
        question_builder=build_context_question,
    )


if __name__ == '__main__':
    main()
