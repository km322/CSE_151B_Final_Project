
import json, os, re, sys
from pathlib import Path

import torch

# peft 0.19 imports EmbeddingParallel; transformers 4.57 dropped it
import transformers.integrations.tensor_parallel as _tp
if not hasattr(_tp, 'EmbeddingParallel'):
    _tp.EmbeddingParallel = _tp.ReplicateParallel

from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'notebooks'))
from judger import Judger

MODEL_ID      = 'Qwen/Qwen3-4B-Thinking-2507'
FILTERED_PATH = REPO / 'data' / 'public_filtered.jsonl'
OUTPUT_DIR    = REPO / 'models' / 'grpo_lora'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRPO_MAX_PROMPT_TOKENS = 2048
SAMPLING_CFG = {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0}

judger = Judger(strict_extract=False)

# Prompts MUST match the notebook (otherwise reward signal is biased).
SYSTEM_PROMPT_MATH = 'You are an expert mathematician solving competition problems across algebra, calculus, statistics, probability, linear algebra, geometry, number theory, and discrete math. Reason carefully, then emit a final answer the automated grader can parse.\n\n# Method\n- Identify the problem type and the most direct solution method.\n- State the key formula or theorem before computing.\n- Keep every quantity in EXACT symbolic form throughout. Do not round, truncate, or convert to decimal at any intermediate step.\n- Verify by substitution, a limit/edge case, or solving a simpler instance before committing. If a check fails, restart from the broken step.\n\n# Final-answer format (STRICT — grader uses regex + sympy with 1e-8 relative tolerance)\n- Place the final answer inside \\boxed{...} at the very end of your response.\n- ALWAYS prefer exact symbolic forms over decimals.\n    Use \\boxed{\\frac{1}{3}}    NOT \\boxed{0.333}\n    Use \\boxed{\\sqrt{2}}      NOT \\boxed{1.414}\n    Use \\boxed{\\frac{\\pi}{4}} NOT \\boxed{0.7854}\n    Use \\boxed{\\ln 2}         NOT \\boxed{0.6931}\n    Use \\boxed{e^{2}}          NOT \\boxed{7.389}\n- Decimal only if the exact value is itself a finite decimal or the problem asks for one. Never write "\\approx" inside the box.\n- Inside \\boxed{}: bare value(s) only. No units, no "x =", no \\text{...}.\n\n# Multiple sub-answers\nPut ALL sub-answers in a SINGLE \\boxed{} separated by commas, in order:\n    \\boxed{580, 660, 80}\n    \\boxed{\\frac{1}{2}, \\sqrt{3}, \\pi}\nDo not split sub-answers across multiple boxes.'
SYSTEM_PROMPT_MCQ  = 'You are an expert mathematician answering a multiple-choice competition problem. Solve rigorously, then output ONE letter.\n\n# Method\n- Compute the answer in EXACT symbolic form first.\n- Convert to decimal only at the very end to match against the listed options.\n- If your value matches no option, RECOMPUTE — don\'t pick the visually closest blindly.\n- Sanity check (sign, magnitude) before committing.\n\n# Final-answer format (STRICT)\n- Output exactly one \\boxed{X} where X is a single capital letter (A, B, C, ...).\n- Examples: \\boxed{C}, \\boxed{E}.\n- No "Option C", no extra characters — just the letter.'
USER_PROMPT_SUFFIX = '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'


def build_prompt(question, options):
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = '\n'.join(f'{lbl}. {opt.strip()}' for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f'{question}\n\nOptions:\n{opts_text}{USER_PROMPT_SUFFIX}'
    return SYSTEM_PROMPT_MATH, f'{question}{USER_PROMPT_SUFFIX}'


def extract_letter(text):
    m = re.search(r'\\boxed\{([A-Za-z])\}', text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r'\b([A-Z])\b', text.upper())
    return matches[-1] if matches else ''


def score_one(item, response):
    if item.get('options'):
        return extract_letter(response) == str(item['answer']).strip().upper()
    gold = item['answer']
    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        return bool(judger.auto_judge(pred=response, gold=gold_list,
                                      options=[[]] * len(gold_list)))
    except Exception:
        return False


def reward_fn(prompts, completions, **kwargs):
    items = [json.loads(j) for j in kwargs['item_json']]
    rewards = []
    for comp, item in zip(completions, items):
        text = comp if isinstance(comp, str) else (
            comp[0]['content'] if isinstance(comp, list) and comp else str(comp)
        )
        try:
            rewards.append(1.0 if score_one(item, text) else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def main():
    if not FILTERED_PATH.exists():
        raise FileNotFoundError(f'Add {FILTERED_PATH} before running GRPO.')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    rows = [json.loads(l) for l in open(FILTERED_PATH)]
    print(f'Filtered set: {len(rows)} items')

    def to_record(item):
        sys_p, usr_p = build_prompt(item['question'], item.get('options'))
        prompt = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': sys_p}, {'role': 'user', 'content': usr_p}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        if len(ids) > GRPO_MAX_PROMPT_TOKENS:
            prompt = tokenizer.decode(ids[-GRPO_MAX_PROMPT_TOKENS:], skip_special_tokens=False)
        return dict(prompt=prompt, item_json=json.dumps(item))

    dataset = Dataset.from_list([to_record(r) for r in rows])

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    # pin 4-bit model to this rank's GPU
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map={'': local_rank},
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )

    grpo_cfg = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        learning_rate=1e-6,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        bf16=True,
        logging_steps=1,
        save_steps=25,
        max_grad_norm=1.0,
        beta=0.04,
        num_generations=4,
        max_completion_length=2048,
        temperature=SAMPLING_CFG['temperature'],
        top_p=SAMPLING_CFG['top_p'],
        top_k=SAMPLING_CFG['top_k'],
        min_p=SAMPLING_CFG['min_p'],
        use_vllm=True,
        vllm_mode='colocate',
        vllm_gpu_memory_utilization=0.35,
        vllm_max_model_length=8192,
        report_to='none',
        ddp_find_unused_parameters=False,
    )

    trainer = GRPOTrainer(
        model=model, args=grpo_cfg,
        train_dataset=dataset, reward_funcs=[reward_fn],
        peft_config=peft_cfg, processing_class=tokenizer,
    )

    # resume from latest checkpoint if any
    resume = any(p.is_dir() and p.name.startswith('checkpoint-')
                 for p in OUTPUT_DIR.iterdir()) if OUTPUT_DIR.exists() else False
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(OUTPUT_DIR))
    print(f'GRPO adapter saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
