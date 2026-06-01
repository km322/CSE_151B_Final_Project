"""Full inference pipeline (SC voting + GRPO LoRA) -> csv/submission.csv."""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

HF_ADAPTER_REPO = "k1mittal/cse151b-grpo-lora"

MODEL_ID       = "Qwen/Qwen3-4B-Thinking-2507"
LOCAL_ADAPTER   = Path(__file__).parent / "models" / "grpo_lora"
GPU_IDS        = os.environ.get("GPU_IDS", "0,1")
TENSOR_PARALLEL = int(os.environ.get("TENSOR_PARALLEL", "2"))
MAX_MODEL_LEN  = 24576
SC_K           = 3          # self-consistency samples per item

SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS


# prompts (must match training)
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician solving competition problems across algebra, "
    "calculus, statistics, probability, linear algebra, geometry, number theory, "
    "and discrete math. Reason carefully, then emit a final answer the automated "
    "grader can parse.\n"
    "\n"
    "# Method\n"
    "- Identify the problem type and the most direct solution method.\n"
    "- State the key formula or theorem before computing.\n"
    "- Keep every quantity in EXACT symbolic form throughout. Do not round, "
    "truncate, or convert to decimal at any intermediate step.\n"
    "- Verify by substitution, a limit/edge case, or solving a simpler instance "
    "before committing. If a check fails, restart from the broken step.\n"
    "\n"
    "# Final-answer format (STRICT — grader uses regex + sympy with 1e-8 relative tolerance)\n"
    "- Place the final answer inside \\boxed{...} at the very end of your response.\n"
    "- ALWAYS prefer exact symbolic forms over decimals.\n"
    "    Use \\boxed{\\frac{1}{3}}    NOT \\boxed{0.333}\n"
    "    Use \\boxed{\\sqrt{2}}      NOT \\boxed{1.414}\n"
    "    Use \\boxed{\\frac{\\pi}{4}} NOT \\boxed{0.7854}\n"
    "    Use \\boxed{\\ln 2}         NOT \\boxed{0.6931}\n"
    "    Use \\boxed{e^{2}}          NOT \\boxed{7.389}\n"
    "- Decimal only if the exact value is itself a finite decimal or the problem asks for one. "
    "Never write \"\\approx\" inside the box.\n"
    "- Inside \\boxed{}: bare value(s) only. No units, no \"x =\", no \\text{...}.\n"
    "\n"
    "# Multiple sub-answers\n"
    "Put ALL sub-answers in a SINGLE \\boxed{} separated by commas, in order:\n"
    "    \\boxed{580, 660, 80}\n"
    "    \\boxed{\\frac{1}{2}, \\sqrt{3}, \\pi}\n"
    "Do not split sub-answers across multiple boxes."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician answering a multiple-choice competition "
    "problem. Solve rigorously, then output ONE letter.\n"
    "\n"
    "# Method\n"
    "- Compute the answer in EXACT symbolic form first.\n"
    "- Convert to decimal only at the very end to match against the listed options.\n"
    "- If your value matches no option, RECOMPUTE — don't pick the visually closest blindly.\n"
    "- Sanity check (sign, magnitude) before committing.\n"
    "\n"
    "# Final-answer format (STRICT)\n"
    "- Output exactly one \\boxed{X} where X is a single capital letter (A, B, C, ...).\n"
    "- Examples: \\boxed{C}, \\boxed{E}.\n"
    "- No \"Option C\", no extra characters — just the letter."
)

USER_PROMPT_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def build_prompt(question, options):
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts = "\n".join(f"{l}. {o.strip()}" for l, o in zip(labels, options))
        user = f"{question}\n\nOptions:\n{opts}{USER_PROMPT_SUFFIX}"
        sys_p = SYSTEM_PROMPT_MCQ
    else:
        user = f"{question}{USER_PROMPT_SUFFIX}"
        sys_p = SYSTEM_PROMPT_MATH
    return sys_p, user


def extract_letter(text):
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def modal(keys):
    counts = Counter(k for k in keys if k)
    if not counts:
        return ""
    top = max(counts.values())
    winners = [k for k, c in counts.items() if c == top]
    for k in keys:
        if k in winners:
            return k
    return winners[0]


def generate_one(llm, tokenizer, item, lora_request=None):
    from vllm import SamplingParams

    sys_p, usr_p = build_prompt(item["question"], item.get("options"))
    messages = [{"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    max_new = MAX_MODEL_LEN - n_prompt - 64
    if max_new < 256:
        return ""
    sp = SamplingParams(
        temperature=SAMPLING["temperature"],
        top_p=SAMPLING["top_p"],
        top_k=SAMPLING["top_k"] if SAMPLING["top_k"] > 0 else -1,
        min_p=SAMPLING["min_p"],
        max_tokens=max_new,
        repetition_penalty=1.0,
    )
    out = llm.generate([prompt], sampling_params=sp, lora_request=lora_request)
    return out[0].outputs[0].text


def run_inference(
    private_path="data/private.jsonl",
    output_path="csv/submission.csv",
    k=SC_K,
    adapter_path=None,
):
    """Run the pipeline on private_path and write CSV to output_path."""
    from transformers import AutoTokenizer
    from vllm import LLM
    from tqdm import tqdm

    private_path = Path(private_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # adapter: local first, fall back to HF Hub
    if adapter_path:
        adapter_dir = Path(adapter_path)
    elif LOCAL_ADAPTER.exists() and (LOCAL_ADAPTER / "adapter_config.json").exists():
        adapter_dir = LOCAL_ADAPTER
    else:
        if LOCAL_ADAPTER.exists():
            cps = sorted(
                [d for d in LOCAL_ADAPTER.iterdir()
                 if d.is_dir() and d.name.startswith("checkpoint-")
                 and (d / "adapter_config.json").exists()],
                key=lambda d: int(d.name.split("-")[1]),
            )
            if cps:
                adapter_dir = cps[-1]
            else:
                adapter_dir = None
        else:
            adapter_dir = None

        if adapter_dir is None:
            from huggingface_hub import snapshot_download
            print(f"Downloading adapter from {HF_ADAPTER_REPO}...")
            adapter_dir = Path(snapshot_download(HF_ADAPTER_REPO))

    has_adapter = adapter_dir is not None and (adapter_dir / "adapter_config.json").exists()
    print(f"Adapter: {adapter_dir} (loaded={has_adapter})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = dict(
        model=MODEL_ID,
        dtype="bfloat16",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.75,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        max_num_seqs=128,
        max_num_batched_tokens=16384,
        tensor_parallel_size=TENSOR_PARALLEL,
    )
    if has_adapter:
        llm_kwargs.update(enable_lora=True, max_lora_rank=32, max_loras=1)

    llm = LLM(**llm_kwargs)

    lora_request = None
    if has_adapter:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("grpo", 1, str(adapter_dir))
        print(f"LoRA adapter loaded from {adapter_dir}")

    items = [json.loads(l) for l in open(private_path)]
    print(f"Private set: {len(items)} items, SC k={k}")

    sys.path.insert(0, str(Path(__file__).parent / "notebooks"))
    from judger import Judger
    judger = Judger(strict_extract=False)

    def vote_key(text, is_mcq):
        if is_mcq:
            return extract_letter(text)
        extracted = judger.extract_ans(text) or ""
        parts = [judger.norm_ans_str(p) for p in judger.split_by_comma(extracted)]
        if not parts:
            return ""
        return parts[0] if len(parts) == 1 else "(" + ", ".join(parts) + ")"

    results = []
    for item in tqdm(items, desc="Inference"):
        is_mcq = bool(item.get("options"))
        responses, keys = [], []
        for _ in range(k):
            try:
                text = generate_one(llm, tokenizer, item, lora_request=lora_request)
            except Exception as e:
                text = f"[error: {e!r}]"
            responses.append(text)
            keys.append(vote_key(text, is_mcq))

        modal_key = modal(keys)
        chosen = next(
            (r for r, kk in zip(responses, keys) if kk == modal_key),
            responses[0] if responses else "",
        )
        results.append({"id": item["id"], "response": chosen})

    results.sort(key=lambda r: r["id"])
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])
        for r in results:
            writer.writerow([r["id"], r["response"]])

    print(f"Wrote {len(results)} rows to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSE 151B — full inference pipeline")
    parser.add_argument("--private-path", default="data/private.jsonl")
    parser.add_argument("--output", default="csv/submission.csv")
    parser.add_argument("--k", type=int, default=SC_K, help="self-consistency samples")
    parser.add_argument("--adapter-path", default=None, help="override adapter directory")
    args = parser.parse_args()
    run_inference(
        private_path=args.private_path,
        output_path=args.output,
        k=args.k,
        adapter_path=args.adapter_path,
    )
