# dpinfer Lab — Hardware-Aware Coding LLM Inference

> **Status: early research / proof of concept.** Working local baseline: Qwen3-4B with PyTorch on an RTX 4090. Hybrid CPU/GPU attention is a standalone mathematical prototype; it has **not** demonstrated an end-to-end speedup or improved coding accuracy.

## 1. Why this project exists

`dpinfer` investigates **model–system co-design** for coding-oriented language models on memory-asymmetric machines: a fast GPU with limited VRAM alongside a CPU with larger, slower memory. The long-term goal is to make a coding model **more accurate on software-engineering tasks and faster in actual inference**, potentially by training it to use hardware-friendly attention, memory representations, or expert routing.

The first research question is deliberately narrower:

> For real coding prompts, is attention to historical code concentrated and reusable enough that a CPU/GPU KV-memory hierarchy can reduce GPU memory pressure and/or decode latency without harming answer quality?

A related systems question is whether it is cheaper to **compute attention over CPU-resident KV on the CPU** and transfer only compact softmax statistics, or to copy that KV over PCIe and compute on the GPU.

These are **hypotheses to test**, not established results. Changing memory placement alone does not improve model accuracy; improved task quality would require separate fine-tuning and evaluation.

## 2. Hardware and software

### Local development machine (current)

| Component | Configuration / status |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, `24564 MiB` reported VRAM |
| Host RAM | 64 GB (user-reported) |
| NVIDIA driver | `580.126.09` |
| Driver-reported CUDA capability | `13.0` (not necessarily installed toolkit version) |
| Conda environment | `dpinfer` |
| Python | 3.14 |
| PyTorch | `2.14.1+cu130` |
| PyTorch CUDA runtime | `13.0` |
| Transformers | `5.23.0` |
| Verification | CUDA tensor allocation and matrix multiplication passed |

The desktop and CosyVoice processes can occupy GPU memory. Stop unnecessary GPU workloads before collecting comparable timings.

**Interpreter gotcha:** pyenv shims shadow conda here, so `conda run -n dpinfer python` resolves to a pyenv Python without torch. Activate the env in an interactive shell, or call the interpreter directly: `/home/wukong/miniconda3/envs/dpinfer/bin/python`.

### Larger follow-up machine (not the local baseline)

- One NVIDIA L40, approximately **44.39 GiB usable VRAM** reported by PyTorch.
- Two Intel Xeon Platinum 8362 processors, **64 physical cores** total.
- Approximately **1 TiB host RAM** according to earlier observations; verify actual available capacity.
- Recorded pinned-memory CPU→GPU throughput: **22.63 GiB/s**; GPU→CPU: **24.45 GiB/s**.
- Recorded effective GPU copy throughput: **612.2 GiB/s**, counting read and write traffic.

Source measurements and background are in `l40-report.md` and `research.md` if copied into this repository. Measurements on the L40 **must not** be treated as RTX 4090 benchmarks.

## 3. Research stages

| Phase | Experiment | Main result sought | Status |
|---|---|---|---|
| 0 | Run Qwen3-4B and establish coding-generation baseline | Repeatable latency and throughput | **Baseline obtained** |
| 1 | Collect coding attention locality traces | Attention mass per KV page; hot-page reuse across steps | **Planned** |
| 2 | Compare full-GPU, CPU-resident, synchronous offload, exact hybrid attention | Correctness and cost crossover | **Standalone PoC available; optimized comparison pending** |
| 3 | Integrate a hybrid attention backend into model inference | End-to-end decode latency and memory under equal workloads | **Planned** |
| 4 | Study sparse / compressed / adaptive-rank KV | Quality versus memory and transfer budget | **Planned** |
| 5 | Coding-aware LoRA or architecture adaptation | **Both** coding-quality improvement and measured speed improvement | **Planned** |
| 6 | Scale to Qwen hybrid-attention 27B and larger MoE systems | Transferability to larger models / CPU-rich hardware | **Future** |

**Important:** Qwen3-4B is being used as a relatively straightforward dense GQA control model. The later hybrid DeltaNet/attention model is a *different architecture*; do not assume its cache layout or Attention layers behave identically. MoE expert routing experiments apply only to MoE models, not this dense 4B baseline.

## 4. Quick start — local RTX 4090

### 4.1 Environment

A known working user configuration was **Python 3.14 + PyTorch 2.14.1 CUDA 13.0** on NVIDIA driver 580.126.09.

```bash
conda create -n dpinfer python=3.14 -y
conda activate dpinfer
python -m pip install -U pip
python -m pip install torch==2.14.1 torchvision==0.29.1 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install transformers accelerate datasets safetensors \
  huggingface_hub sentencepiece numpy scipy pandas matplotlib tqdm
```

If an exact wheel or dependency is unavailable on another host, use the supported compatibility matrix for that host rather than forcing versions. Keep vLLM in a **separate** environment if it requires different compiled PyTorch/CUDA extensions.

Verify GPU operation:

```bash
python - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device='cuda')
_ = x @ x
torch.cuda.synchronize()
print('CUDA computation successful')
PY
```

### 4.2 Download the first model

The control model is [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), loaded locally as BF16 by Transformers.

```bash
mkdir -p ~/dpinfer-lab/{models,data,results,profiles}
hf download Qwen/Qwen3-4B \
  --local-dir ~/dpinfer-lab/models/Qwen3-4B
ls ~/dpinfer-lab/models/Qwen3-4B/config.json
```

Use an **absolute path** when loading weights. Running code from another working directory with `./models/Qwen3-4B` previously caused a model path lookup error.

### 4.3 Verify text generation

This is `scratch/hf_smoke.py` (built on the shared loader in `src/dpinfer/common.py`). Standalone equivalent:

```python
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = str(Path('~/dpinfer-lab/models/Qwen3-4B').expanduser())
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16, device_map='cuda',
    local_files_only=True,
).eval()

messages = [{'role': 'user', 'content': 'Write a C++17 implementation of binary search.'}]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    enable_thinking=False, return_tensors='pt', return_dict=True,
).to('cuda')

with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)

new_tokens = output[0, inputs['input_ids'].shape[-1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
```

```bash
python scratch/hf_smoke.py
```

**Why `return_dict=True` / `**inputs`?** `apply_chat_template` may return a `BatchEncoding`; passing that whole object as the first positional argument to `generate()` caused an `AttributeError: shape`. `model.generate(**inputs)` passes `input_ids` and `attention_mask` correctly. `enable_thinking=False` makes the initial coding tests easier to compare by disabling the long thinking-style output.

## 5. Observed baseline (do not overwrite with estimates)

The following numbers were measured by the user on the **local RTX 4090** using Transformers with Qwen3-4B; input was a short binary-search coding prompt, batch size 1, output 256 tokens:

**End-to-end `generate()`** (includes prefill and generation overhead):

| Run | Input tokens | Output tokens | Total `generate()` time | Output tokens / total second |
|---|---:|---:|---:|---:|
| `bench/hf_generate_timed.py` (formerly `perf-test.py`) | 23 | 256 | 3.370 s | 75.96 |
| `bench/hf_generate_median.py` (formerly `perf-decode.py`; 5-run median) | 23 | 256 | 3.360814 s | 76.17 |

The two runs differ by about **0.28%**. `perf-decode.py` was renamed because it timed the full `generate()` call, not decode.

**Prefill vs decode separated** (`bench/hf_prefill_decode.py`, 2026-11-17): manual greedy forward loop over a `DynamicCache`, each forward timed with CUDA events, 5 runs, 256 decode steps after prefill; output token-identical to greedy `generate()`. Full write-up: `results/2026-11-17_hf-prefill-decode_qwen3-4b_23tok.md`.

| Metric | Value |
|---|---|
| Prefill (23 tokens) | 14.73 ms median (14.50–14.83) |
| Decode per token | 13.25 ms median; p10 13.05, p90 13.38, p99 13.60 |
| Decode throughput | 75.55 tok/s median (75.52–75.60) |
| Peak VRAM allocated | 7.54 GiB |

At 23 input tokens, end-to-end throughput ≈ decode throughput, and prefill costs about one decode step (both weight-read bound). Decode reaches ~60% of the memory-bandwidth floor (~8 GB weights per step at ~1 TB/s → ~8 ms). Whether the remaining ~5 ms/token is kernel-launch overhead is an **untested hypothesis** (test: static cache + CUDA graphs). About 1.9 GiB of VRAM was held by other processes (likely the desktop) during this run. No 1K–32K long-context benchmark has yet been reported.

Still to add for long contexts: time to first token at realistic prompt lengths, and repeated configurations with fixed settings.

Suggested local matrix:

| Prompt length | Generated tokens | Purpose |
|---:|---:|---|
| ~1K | 256 | Short-context control |
| ~4K | 256 | Initial attention cost |
| ~8K | 256 | KV scale |
| ~16K | 256 | Hybrid/offload pressure |
| ~32K | 256 | Longer-context stress, if model/config permits |

Use tokenizer-counted lengths of **real code**, not only repeated filler text. Control for prefix caching, generation mode, batch size, model version, precision, and background GPU load.

## 6. Download research data

### RepoQA — attention locality / code retrieval

Repository: <https://github.com/evalplus/repoqa>

```bash
cd ~/dpinfer-lab/data
git clone https://github.com/evalplus/repoqa.git
git clone https://github.com/evalplus/repoqa_release.git
```

Start with **50–100** function-retrieval examples. The target is to correlate attention with code locations such as functions, definitions and relevant modules, while measuring function-retrieval correctness. Check the repositories' current layouts before writing an adapter; this README does not assume a stable dataset file path.

### SWE-smith — software-engineering tasks (later)

Dataset: <https://huggingface.co/datasets/SWE-bench/SWE-smith>

```python
from datasets import load_dataset
import json

ds = load_dataset('SWE-bench/SWE-smith', split='train', streaming=True)
with open('swe_smith_100.jsonl', 'w') as f:
    for item in ds.take(100):
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
```

SWE-smith tasks may be useful for training; **agent trajectories** are a distinct artifact and should not be assumed to be in the task rows. Use a separately curated trajectory dataset if studying multi-turn tool use.

### Hold out for evaluation

- [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified): end-to-end repository repair quality.
- [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench): algorithmic coding quality.

Keep task-level and repository-level splits disjoint where possible; don't train on held-out evaluation patches or expected answers. Track dataset revision and license before redistribution.

## 7. First mathematical PoC — exact split GQA attention

A standalone script, `hybrid_gqa_poc.py`, was previously provided in the **dpinfer local PoC archive**. Copy it into the repo root or adjust the path below. It is not created automatically by the model download.

```bash
python hybrid_gqa_poc.py \
  --seq 8192 --gpu-frac 0.25 --threads 16 --repeats 10

for seq in 2048 8192 16384; do
  for frac in 0.10 0.25 0.50 0.75; do
    python hybrid_gqa_poc.py \
      --seq "$seq" --gpu-frac "$frac" --repeats 10
  done
done
```

This script uses **synthetic FP32 Q/K/V**, PyTorch reference operations and a thread pool. It evaluates **one decode-step GQA attention** split between CPU and GPU, and checks its output against full GPU attention. It **does not** load Qwen, measure coding accuracy, or implement a production FlashAttention backend.

For scores `s_i = q·k_i / sqrt(d)`, each KV partition `P` computes

\[
m_P=\max_{i\in P}s_i,\quad
\ell_P=\sum_{i\in P}e^{s_i-m_P},\quad
u_P=\sum_{i\in P}e^{s_i-m_P}v_i.
\]

With GPU partition `G` and CPU partition `C`, define `m = max(m_G,m_C)`. Then the exact fused attention is

\[
O=\frac{e^{m_G-m}u_G+e^{m_C-m}u_C}
        {e^{m_G-m}\ell_G+e^{m_C-m}\ell_C}.
\]

This is standard **stable log-sum-exp fusion**, not a novel mathematical result. CPU sends compact partial statistics rather than the complete historical KV; numerical error should be small under FP32 reference computation.

**Benchmark caveat:** the PoC creates a full-GPU reference using a preparatory CPU→GPU copy excluded from its hybrid timing. Its CPU and GPU kernels are unoptimized reference kernels. Do **not** claim a speedup over optimized GPU attention, vLLM, or FlashAttention from this script alone. For a fair benchmark, compare identical data layouts, transfer accounting, kernel quality, synchronization, and memory budgets.

The separate `hybrid_attention_lab.py` from the **dpinfer lab starter archive** reports per-path timings and an *ideal-overlap estimate*, but that estimate is **not observed simultaneous execution latency**. Its arguments differ: `--gpu-fraction --heads --kv-heads --head-dim` instead of the local PoC's `--gpu-frac --q-heads --kv-heads --dim`.

## 8. Experiment E1 — real coding attention locality (not yet implemented)

Once the synthetic exact-attention reference passes, collect **real Q/K tensors** from the same Qwen3-4B checkpoint used in the baseline. Do not assume synthetic Q/K reflect real coding locality.

For each selected **decode** step, attention layer and query head, compute normalized attention mass of logical KV page `b`:

\[
A_{l,h,b}=\sum_{i\in b}
\operatorname{softmax}\!\left(\frac{q_{l,h}K_{l,h}^{\top}}{\sqrt{d}}\right)_i.
\]

Capture and record:

- Model commit/revision, tokenizer, exact prompt and task ID.
- Logical token-to-page mapping and page size (try 64/128/256 tokens).
- Layer, query head, decode step, and context length.
- Top 5%/10%/20% page **attention mass**; page reuse between adjacent decoding steps.
- RepoQA correctness and relevant code-span positions (if available).

**Sampling method:** use a dedicated debug path (e.g., instrument attention projections plus the correct RoPE and causal mask, or reconstruct scores offline from exported tensors). GPU-efficient attention kernels often **do not expose complete attention weights**. Paged caches must be read in logical token order. A generic PyTorch profiler trace does not provide `A_{l,h,b}`.

**Important:** High retained attention mass is *not* proof of preserved coding accuracy. Validate output deviations and real task success separately.

## 9. Experiment E2 — fair CPU/GPU execution comparison

Compare these alternatives on matched Q/K/V and identical numerical precision:

1. **Full GPU**: all KV already GPU-resident (quality/performance reference, if memory permits).
2. **Sync offload**: CPU-resident KV copied to GPU before attention.
3. **CPU attention**: CPU processes CPU-resident KV, sending only partial softmax statistics.
4. **Exact hybrid**: GPU processes local KV concurrently with CPU historical attention, followed by stable fusion.
5. **Sparse hybrid** (later): approximate by selecting important blocks; measure accuracy loss explicitly.

Required measurements: max/relative output error, end-to-end attention latency, CPU compute time, GPU kernel time, actual H2D/D2H bytes, exposed transfer time, GPU/host memory footprint, and latency distributions. Vary context length, CPU thread count, page size, GPU-resident fraction, and memory budget. Avoid overlapping a GPU renderer or CosyVoice workload with measurements.

A simple cost model is:

\[
T_{\mathrm{hybrid}} \approx T_{\mathrm{sync}}+\max(T_{\mathrm{CPU\ path}},T_{\mathrm{GPU\ path}})+T_{\mathrm{fusion}},
\]

but **only a measured runtime** can establish actual overlap and critical-path latency. Test CPU attention against *fetch + GPU attention* before assuming the CPU path wins. An exact hybrid method cannot improve model quality by itself.

## 10. Experimental interpretation and gates

- **Gate A — correctness:** exact split attention matches a trusted reference within a stated numeric tolerance, using identical masks, scales, positions and head mapping.
- **Gate B — CPU/GPU crossover:** a measurable region exists where CPU-compute-plus-small-result-transfer is competitive with transferring KV to the GPU.
- **Gate C — locality:** real code prompts show strong enough page concentration and temporal reuse to justify sparse/priority placement.
- **Gate D — practical advantage:** optimizations improve *end-to-end* decode latency or fit otherwise infeasible contexts at an explicit memory budget.
- **Gate E — model quality:** any sparsification, compression or fine-tuning is evaluated independently on held-out coding tasks.

If Gate B fails, explore vectorized / quantized CPU attention or compressed KV. If Gate C fails, do not force a sparsity story; consider representational compression or alternate architecture. A system result under constrained VRAM is not automatically faster than a full-GPU baseline when all KV fits.

## 11. From model profiling to architectural research

Only after measuring a real workload should the project attempt:

- **Adaptive precision/rank KV:** allocate representation budget by layer/head/code segment, optimizing downstream attention quality versus memory and hardware cost.
- **Hardware-aware KV placement:** dynamically choose GPU-resident, CPU-computed, or GPU-fetched KV blocks.
- **Coding-aware fine-tuning:** supervised code tasks + teacher distillation + a calibrated efficiency proxy, with ablations against ordinary coding SFT.
- **Hybrid architectures:** eventually test models that mix recurrent/DeltaNet state and full attention; the GQA control experiment is not a substitute for this evaluation.
- **MoE expert placement:** separate future research on genuine MoE models; the local Qwen3-4B **has no routed experts**.

Research objective, not an achieved result:

\[
\min_{\theta,\pi}\;\mathcal L_{\mathrm{coding}}(\theta,\pi)
+\lambda\widehat T_{\mathrm{decode}}(\theta,\pi)
\quad \text{s.t.}\quad M_{\mathrm{GPU}}\le M_{\max}.
\]

Latency is generally not directly differentiable; practical training would need a differentiable surrogate, relaxed placement policy, or alternating optimization. Always separately test whether software changes and model changes improve runtime and coding accuracy.

## 12. Suggested project structure

```text
dpinfer-lab/
├── README.md                      # This file
├── models/                        # Downloaded checkpoints (not committed)
│   └── Qwen3-4B/
├── data/                          # RepoQA / SWE-smith (not committed)
├── results/                       # Benchmark write-ups, JSON/CSV, figures
├── profiles/                      # Profiler traces (not committed)
├── logs/                          # Run logs (not committed)
├── bench/
│   ├── hf_generate_timed.py       # One timed generate() run + output text
│   ├── hf_generate_median.py      # Median end-to-end generate() over 5 runs
│   └── hf_prefill_decode.py       # Prefill vs per-token decode, CUDA-event timed
├── scratch/
│   └── hf_smoke.py                # Load + generate smoke test (§4.3)
├── src/dpinfer/
│   └── common.py                  # Shared model loading / inputs / warmup
├── hybrid_gqa_poc.py              # Future: copy from dpinfer local PoC archive
└── traces/                        # Future: Q/K traces and page metadata
```

Bench and scratch scripts add `src/` to `sys.path` themselves, so they run from any directory. Entries marked **future** do not exist yet. Do not commit model weights, bulky traces or evaluation answers unless distribution rights permit.

## 13. Reproducibility checklist

Before publishing any performance claim, record:

- [ ] Hardware, driver, CUDA runtime, PyTorch/Transformers/vLLM versions.
- [ ] Model ID, immutable revision and precision; tokenizer and chat template.
- [ ] Data source, revision, prompt list and exact token counts.
- [ ] Warmup, repetitions, seed, batch/concurrency and generation parameters.
- [ ] GPU/CPU memory budget and actual background workloads.
- [ ] Prefill, decode, p50/p95, throughput and memory measured separately.
- [ ] PCIe bytes/time and CPU/GPU kernel measurements.
- [ ] Accuracy tested on disjoint coding tasks; numerical error reported independently.
- [ ] Baselines with **comparable optimized kernels**, not only Python reference code.

## 14. Immediate next steps

1. ~~Preserve the ~76 tokens/s short-prompt baseline and separate prefill from decode.~~ Done for a 23-token prompt (§5).
2. Run `hybrid_gqa_poc.py` for numerical correctness over 2K–16K sequences.
3. Build a longer-input 1K/4K/8K/16K/32K coding benchmark from RepoQA by extending `bench/hf_prefill_decode.py` with a context-length option; have it write JSON to `results/`.
4. Add an **offline trace extractor** for a few layers/heads on 50–100 real RepoQA queries.
5. Plot attention-mass concentration and page reuse before implementing sparse scheduling.
6. Implement and compare optimized CPU attention, synchronous GPU fetch and exact hybrid on matched inputs.
7. Only after positive results: integrate runtime changes and test coding-aware adapters.

---

**Current evidence:** The RTX 4090 PyTorch/CUDA stack runs successfully, and Qwen3-4B loads and generates text with a measured short-input throughput near 76 output tokens per total second; isolated decode is 13.25 ms/token (75.55 tok/s) with a 14.73 ms prefill at 23 input tokens. **Not yet demonstrated:** long-context locality, a hybrid attention speedup, or accuracy gains from fine-tuning.
