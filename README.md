# CSE 151B Competition: Math Reasoning

## Hardware

2x NVIDIA A40 (46 GB each). vLLM tensor-parallel=2, BF16.

Approx. runtime:
- Full 943-item private set, SC k=3: ~4 days
- 200-item subset: ~17 hours
- Single item: ~6 minutes

## Setup

Base model `Qwen/Qwen3-4B-Thinking-2507` is pulled from HuggingFace at first run.

The GRPO LoRA adapter is at `k1mittal/cse151b-grpo-lora` on HuggingFace Hub. `run_inference.py` downloads it on its own if `models/grpo_lora/` is empty. To grab it manually:

```python
from huggingface_hub import snapshot_download
snapshot_download("k1mittal/cse151b-grpo-lora", local_dir="models/grpo_lora")
```

## Run

```bash
pip install sympy numpy 'transformers>=4.51,<5' 'vllm>=0.8.5' 'torch>=2.5' tqdm huggingface_hub bitsandbytes antlr4-python3-runtime==4.11.1
python run_inference.py
```

Or call the function directly from Python:

```python
from run_inference import run_inference
run_inference(private_path="data/private.jsonl", output_path="csv/submission.csv")
```

Other CLI options:
```bash
python run_inference.py --private-path data/private.jsonl --output csv/submission.csv
python run_inference.py --k 1   # no voting
python run_inference.py --adapter-path models/grpo_lora/checkpoint-561
```

Single GPU:
```bash
GPU_IDS=0 TENSOR_PARALLEL=1 python run_inference.py
```

## Pipeline

`run_inference()`:
1. Loads Qwen3-4B-Thinking via vLLM (BF16, TP=2) and the GRPO LoRA adapter.
2. For each private item, generates one response per sample with the math/MCQ system prompt.
3. k=3 samples per item, majority vote. MCQ: extracted letter. Free-form: judger-normalized symbolic form.
4. Writes `csv/submission.csv` with columns id, response.

## Final hyperparameters

temperature 0.6, top_p 0.95, top_k 20, min_p 0.0. SC k=3. GRPO LoRA r=16, alpha=32, lr 1e-6, KL beta 0.04, num_generations=4, max_completion_length=2048, 1 epoch (561 steps).

## Files

- `run_inference.py` - entry point
- `notebooks/judger.py`, `notebooks/utils.py` - scoring
- `data/private.jsonl` - verifier supplies this
- `models/grpo_lora/` - auto-downloaded from HF if absent
- `csv/submission.csv` - final submission
