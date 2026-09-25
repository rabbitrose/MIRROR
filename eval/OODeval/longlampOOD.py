"""LongLaMP OOD 评测：product_review + topic_writing 测试集。

模型未在这两个任务上训练（域内只有 abstract_generation），用于检验泛化能力。
复用 eval_lamp_sparse 公共设施 + eval_lamp_base 的 full context 口径，
仅任务列表硬编码为 product_review / topic_writing。

按 CLAUDE.md 约定：直接 `python longlampOOD.py` 运行，不接收命令行参数。
可用环境变量：EVAL_MODEL_NAMES / EVAL_GPU_IDS / EVAL_THINKING / EVAL_FULL_CONTEXT_CHARS。
结果覆盖写入 TestResults/<模型名>_longlampOOD_<日期>_sum.json 与 _detail.json。
"""
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))                                         
import eval_lamp_sparse as common

                                                               
OOD_TASKS = ['product_review', 'topic_writing']

MODEL_NAMES = ["Qwen3-1.7B"]
if os.environ.get("EVAL_MODEL_NAMES"):
    MODEL_NAMES = [n.strip() for n in os.environ["EVAL_MODEL_NAMES"].split(",") if n.strip()]
TRAINED_DIR = common.TRAINED_DIR
                                       
FULL_CONTEXT_CHAR_BUDGET = int(os.environ.get("EVAL_FULL_CONTEXT_CHARS", "60000"))
EVAL_THINKING = os.environ.get("EVAL_THINKING", "off")
if EVAL_THINKING not in ("off", "on"):
    raise RuntimeError("EVAL_THINKING 只能取 off/on：%s" % EVAL_THINKING)
_ALGO_BASE = "longlampOOD"
if abs(common.DECODE_TEMPERATURE - 0.4) >= 1e-9:
    _ALGO_BASE = "%s_t%s" % (_ALGO_BASE, ("%g" % common.DECODE_TEMPERATURE).replace(".", "p"))
ALGORITHM = _ALGO_BASE


def resolve_model(name):
    """基座模型在 models 下，训练后的模型在 TrainModels 下"""
    for root in (common.MODELS_DIR, common.TRAINED_DIR):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    raise RuntimeError("模型不存在：%s" % name)


def full_context(task, inp, profile):
    """按原始顺序纳入尽可能多的历史，不做检索或重排；预算不够时保留最近的条目。

    与 eval_lamp_base 同口径：question 完整保留，历史从尾部倒着选再翻回时间正序。
    """
    selected = []
    used = 0
    for item in reversed(profile):
        rendered = common.render_profiles(task, [item])
        if selected and used + len(rendered) > FULL_CONTEXT_CHAR_BUDGET:
            break
        selected.append(item)
        used += len(rendered)
    selected.reverse()                              
    prefix = common.render_profiles(task, selected) if selected else ""
    return prefix, inp, len(selected)


def clip_content(tokenizer, prefix, question, budget):
    """question 必须完整保留，剩余额度才留给历史前缀。"""
    question_ids = tokenizer(question, add_special_tokens=False)['input_ids']
    if len(question_ids) >= budget:
        kept = tokenizer.decode(question_ids[:budget], skip_special_tokens=True)
        return kept, 0
    room = budget - len(question_ids)
    if not prefix:
        return question, 0
    prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
    if len(prefix_ids) > room:
        prefix_ids = prefix_ids[-room:]
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    return prefix_text + question, len(prefix_ids)


def render_prompt(tokenizer, content):
    """渲染对话模板；不认 enable_thinking 的模板自动退回无参渲染。"""
    messages = [{"role": "system", "content": common.SYSTEM_PROMPT},
                {"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=(EVAL_THINKING == "on"))
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def prepare_inputs():
    """构造 full user context 输入，多个模型共用同一份"""
    loaded = {}
    for task in OOD_TASKS:
        rows = common.load_task(task)
        if rows:
            loaded[task] = rows
            print("%s: %d 条测试样本" % (task, len(rows)), flush=True)
    started = time.time()
    prefixes, questions, index, refs, sample_ids = [], [], [], [], []
    profile_counts, included_counts = [], []
    for task, rows in loaded.items():
        for row_number, row in enumerate(rows):
            profile = row.get("profile") or []
            prefix, question, included = full_context(task, row["input"], profile)
            prefixes.append(prefix)
            questions.append(question)
            index.append(task)
            refs.append(row["output"])
            sample_ids.append("%s-%06d" % (task, row_number))
            profile_counts.append(len(profile))
            included_counts.append(included)
    return {
        "loaded": loaded, "prefixes": prefixes, "questions": questions,
        "index": index, "refs": refs, "raw_inputs": questions,
        "sample_ids": sample_ids, "profile_counts": profile_counts,
        "included_counts": included_counts,
        "preprocessing_seconds": time.time() - started,
    }


def evaluate(model_name, shared, gpu_ids, date):
    model_path = resolve_model(model_name)
    print("\n===== %s =====" % model_name, flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    clipped, prefix_tokens = [], []
    for prefix, question in zip(shared["prefixes"], shared["questions"]):
        content, kept = clip_content(tokenizer, prefix, question,
                                     common.USER_TOKEN_BUDGET)
        clipped.append(content)
        prefix_tokens.append(kept)
    prompts = [render_prompt(tokenizer, content) for content in clipped]

    model_started = time.time()
    infer_started = time.time()
    texts = common.run_inference(prompts, gpu_ids, model_path)
    infer_seconds = time.time() - infer_started
    print("推理完成，用时 %.1fs；开始计算指标" % infer_seconds, flush=True)

    per_task = {task: {"preds": [], "refs": []} for task in shared["loaded"]}
    for task, ref, text in zip(shared["index"], shared["refs"], texts):
        per_task[task]["preds"].append(common.postprocess(text))
        per_task[task]["refs"].append(ref)
    metrics = common.run_metrics(per_task, gpu_ids)

    details = [{
        "sample_id": sample_id,
        "task": task,
        "input": raw_input,
        "model_input": model_input,
        "model_output": common.postprocess(text),
        "golden_truth": ref,
        "profile_entries_total": total,
        "profile_entries_included_before_token_truncation": included,
        "prefix_tokens_kept": kept,
    } for sample_id, task, raw_input, model_input, text, ref, total, included, kept in zip(
        shared["sample_ids"], shared["index"], shared["raw_inputs"], clipped,
        texts, shared["refs"], shared["profile_counts"], shared["included_counts"],
        prefix_tokens)]

    model_seconds = time.time() - model_started
    summary_path, detail_path, results = common.save_evaluation_result(
        model_name=model_name,
        algorithm=ALGORITHM,
        method="Full user context without retrieval, OOD tasks (thinking=%s)" % EVAL_THINKING,
        retrieval_seconds=shared["preprocessing_seconds"],
        model_seconds=model_seconds,
        infer_seconds=infer_seconds,
        metrics=metrics,
        created_date=date,
        details=details,
        context_policy="question_reserved_first_then_history_head_fills_budget",
        max_model_len=common.MAX_MODEL_LEN,
        user_token_budget=common.USER_TOKEN_BUDGET,
        decode_temperature=common.DECODE_TEMPERATURE,
        decode_seed=common.DECODE_SEED,
        max_new_tokens=common.MAX_NEW_TOKENS,
        thinking=EVAL_THINKING,
        benchmark="LongLaMP-OOD (user-based test, product_review+topic_writing)",
    )
    print("结果已覆盖写入：%s 和 %s（耗时 %s）" %
          (summary_path, detail_path, results["elapsed"]), flush=True)


def main():
    started = time.time()
    import nltk
    try:
        nltk.data.find("corpora/wordnet.zip")
    except LookupError:
        if not nltk.download("wordnet", quiet=True):
            raise RuntimeError("METEOR 依赖的 NLTK WordNet 不可用")
    shared = prepare_inputs()
    if not shared["loaded"]:
        raise RuntimeError("没有可评测的数据")
    gpu_ids = common.available_gpus()
    if os.environ.get("EVAL_GPU_IDS"):
                                             
                                                             
        gpu_ids = [int(x) for x in os.environ["EVAL_GPU_IDS"].split(",") if x.strip()]
    print("本次评测使用 GPU：%s" % gpu_ids, flush=True)
    date = datetime.now().strftime("%Y%m%d")
    for model_name in MODEL_NAMES:
        evaluate(model_name, shared, gpu_ids, date)
    print("运行总耗时：%s" % common.format_duration(time.time() - started), flush=True)


if __name__ == "__main__":
    main()
