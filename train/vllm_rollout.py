"""给 train_mirror.py / train_mirrorf.py 增加 vLLM rollout 后端（OPD_ROLLOUT=vllm）。

动机：MIRROR 每步的 rollout 用 HF generate 逐 token 解码，占 1.7B 约 30%、4B 约 25% 的步时间。
vLLM 的 paged attention + 连续批处理在同样采样参数下快数倍。

正确性：vLLM 只负责"采样出 ŷ"，loss 仍用 HF 前向在 ŷ 上计算，梯度路径不变。
采样分布与 HF 在数学上相同（同 temperature/top_p），只是 kernel 不同，属于 on-policy
采样的实现差异，不改变目标函数与超参。

LoRA 同步：每个优化步把当前 adapter 存成一个新目录，用新的 LoRARequest id 交给 vLLM，
保证采样用的是"当前 student"而不是基座。
"""
import os
import shutil
import tempfile

ROLLOUT_BACKEND = os.environ.get('OPD_ROLLOUT', 'hf')
VLLM_MEM_UTIL = float(os.environ.get('OPD_VLLM_MEM_UTIL', '0.25'))
VLLM_MAX_LEN = int(os.environ.get('OPD_VLLM_MAX_LEN', '16384'))


class VllmRollout:
    """包一层 vLLM 离线引擎，按步热加载当前 LoRA adapter。"""

    def __init__(self, model_path, tokenizer, lora_rank, temperature, top_p, seed):
        from vllm import LLM, SamplingParams
        self.SamplingParams = SamplingParams
        self.tok = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.step_id = 0
        self.adapter_root = tempfile.mkdtemp(prefix='opd_lora_')
        self.llm = LLM(model=model_path, dtype='bfloat16', max_model_len=VLLM_MAX_LEN,
                       gpu_memory_utilization=VLLM_MEM_UTIL, enable_prefix_caching=False,
                       tensor_parallel_size=1, enable_lora=True, max_lora_rank=lora_rank,
                       max_loras=1, enforce_eager=True)

    def _publish(self, peft_model):
        """把当前 LoRA 权重落盘，返回 vLLM 用的 LoRARequest。"""
        from vllm.lora.request import LoRARequest
        self.step_id += 1
        path = os.path.join(self.adapter_root, 'step%d' % self.step_id)
        peft_model.save_pretrained(path, safe_serialization=True)
        prev = os.path.join(self.adapter_root, 'step%d' % (self.step_id - 1))
        shutil.rmtree(prev, ignore_errors=True)
        return LoRARequest('opd-%d' % self.step_id, self.step_id, path)

    def generate(self, peft_model, prompt_ids, max_news):
        """一次采样一批；返回 [(completion_ids, hit_cap)]。"""
        lora = self._publish(peft_model)
        reqs = [{'prompt_token_ids': ids} for ids in prompt_ids]
        params = [self.SamplingParams(temperature=self.temperature, top_p=self.top_p,
                                      max_tokens=int(n), seed=self.seed + self.step_id)
                  for n in max_news]
        outs = self.llm.generate(reqs, params, lora_request=lora, use_tqdm=False)
        results = []
        for out, cap in zip(outs, max_news):
            ids = list(out.outputs[0].token_ids)
            hit_cap = len(ids) >= cap and self.tok.eos_token_id not in ids
            if self.tok.eos_token_id in ids:
                ids = ids[:ids.index(self.tok.eos_token_id) + 1]
            elif not hit_cap:
                ids = ids + [self.tok.eos_token_id]
            results.append((ids, hit_cap))
        return results

    def close(self):
        shutil.rmtree(self.adapter_root, ignore_errors=True)
