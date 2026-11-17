# L40 runbook — reproducing and extending the experiments on the server

Step-by-step commands for the L40 machine. Background and expectations are in
[l40-outlook.md](l40-outlook.md); what the 4090 desktop measured is in [findings.md](findings.md).

Goal on this machine, in priority order:

1. **Gate B on a server CPU:** how fast can the host serve decode attention (per NUMA node, both
   nodes), versus PCIe?
2. **Copy-free baseline** (`full`) decode latency on the L40.
3. **KV-budget curve:** `hybrid` vs `offload` vs `full` at 50/25/10% GPU KV budgets, 8K–32K.

E1 / Gate C does not need to be repeated here: it measures the model's attention, not the machine.

---

## 0. Before you start

- The server may be shared. Check `nvidia-smi` and `top` first; do not run timing experiments while
  another job uses the GPU or saturates memory bandwidth. Record what else is running.
- Results from this machine always go into paths tagged `_l40` (commands below pass `--out`
  explicitly; the scripts' default output paths are dated only and would mix with the 4090 results).
- Big files (model, data, traces, logs) stay out of git, as on the desktop (`.gitignore`).

## 1. Get the code

```bash
git clone git@github.com:xuanxinjushi/dpinfer-lab.git   # or: cd dpinfer-lab && git pull
cd dpinfer-lab
export REPO=$PWD
export DATE=$(date +%F)
export OUT=$REPO/results/l40/$DATE
mkdir -p $OUT logs
```

## 2. Record the machine (commit this file with the results)

```bash
{
  echo "## nvidia-smi";            nvidia-smi
  echo "## topology";              nvidia-smi topo -m
  echo "## lscpu";                 lscpu
  echo "## numa";                  numactl -H
  echo "## memory";                free -g
  echo "## dimms";                 sudo -n dmidecode -t memory 2>/dev/null | grep -E "Speed|Size|Locator" | sort | uniq -c
  echo "## kernel";                uname -a
} > $OUT/machine.txt 2>&1
```

From `nvidia-smi topo -m`, note the **CPU / NUMA affinity of the GPU** — the NUMA node the L40's
PCIe slot hangs off. Set it for the rest of the runbook:

```bash
export GPU_NODE=0   # replace with the NUMA node shown for GPU0
```

`numactl` missing? `sudo apt install numactl`; without root, use `taskset -c <cores of node>`
(memory placement then relies on first-touch from those cores).

## 3. Environment

Check the driver first: `nvidia-smi` prints the highest CUDA version it supports.

- Driver ≥ 580 (CUDA 13.0): use the same stack as the desktop (README §4.1):
  `torch==2.14.1` from the `cu130` index, `transformers==5.23.0`.
- Older driver: pick the newest PyTorch wheel whose CUDA version the driver supports
  (e.g. `cu128`, `cu126`). Keep `transformers==5.23.0`; the code uses its cache / attention
  interfaces (`AttentionInterface`, `DynamicLayer`), which change between major versions.

```bash
conda create -n dpinfer python=3.14 -y        # or 3.12 if 3.14 wheels are unavailable
conda activate dpinfer
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install transformers==5.23.0 accelerate safetensors huggingface_hub \
    numpy pandas matplotlib tqdm ninja
which g++ ninja                                # needed to JIT-build the CPU kernel
```

If `conda run` picks up the wrong Python (it did on the desktop because of pyenv shims), call the
env's interpreter by full path: `$(conda info --base)/envs/dpinfer/bin/python`.

Version reference (desktop): torch 2.14.1+cu130, transformers 5.23.0, accelerate 1.16.0,
safetensors 0.8.0, tokenizers 0.23.3, numpy 2.5.4, pandas 3.1.1, matplotlib 3.11.2.

## 4. Model and data (not in git)

Expected layout (paths are relative to the repo; override the model with `DPINFER_MODEL`):

```
models/Qwen3-4B/                                   7.6 GB
data/repoqa_release/repoqa-2024-06-23.json.gz      12 MB
```

Either download:

```bash
hf download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
git clone https://github.com/evalplus/repoqa_release.git data/repoqa_release
```

or copy from the desktop (works without internet on one side):

```bash
rsync -aP <desktop>:/media/wukong/jackie/git.repo/dpinfer-lab/models/Qwen3-4B models/
rsync -aP <desktop>:/media/wukong/jackie/git.repo/dpinfer-lab/data/repoqa_release data/
```

The model must be the same checkpoint as on the desktop; record its commit:
`cat models/Qwen3-4B/.cache/huggingface/download/*.metadata 2>/dev/null | head` (or the HF revision).

## 5. Smoke tests (≈ 2 min)

```bash
python scratch/hf_smoke.py                                   # loads the model, generates text
python -c "import sys; sys.path.insert(0,'src'); from dpinfer.cpu_kernel import _ext; _ext(); print('kernel ok')"
```

The kernel is built into `build/cpu_attn/` (AVX2 + FMA + F16C, which Ice Lake supports).

## 6. Gate B microbenchmarks (≈ 15–30 min)

Host bandwidth, PCIe, GPU attention, fp32 PyTorch CPU attention and the bf16 kernel, for each NUMA
placement. The GPU parts are identical across runs; the CPU parts are what differ.

```bash
# GPU-local node
numactl --cpunodebind=$GPU_NODE --membind=$GPU_NODE \
  python bench/cpu_attention.py --threads 8 16 32 --reps 5 \
  --out $OUT/gate_b_node$GPU_NODE.json 2>&1 | tee logs/l40_gate_b_node$GPU_NODE.log

# the other node
OTHER=$((1 - GPU_NODE))
numactl --cpunodebind=$OTHER --membind=$OTHER \
  python bench/cpu_attention.py --threads 8 16 32 --reps 5 \
  --out $OUT/gate_b_node$OTHER.json 2>&1 | tee logs/l40_gate_b_node$OTHER.log

# both sockets, memory interleaved
numactl --interleave=all \
  python bench/cpu_attention.py --threads 16 32 64 --reps 5 \
  --out $OUT/gate_b_both.json 2>&1 | tee logs/l40_gate_b_both.log
```

What to look at:

| Line | Desktop (i9-13900KF) | Question on the L40 server |
|---|---|---|
| `host bandwidth … read` | 71 GB/s | per-socket and two-socket read GB/s |
| `pcie … H2D` | 24.0 GB/s | expect ≈ 22–25 GB/s (earlier note: 22.6 GiB/s) |
| `kernel bf16 L=32768` | 51 GB/s (94.5 ms) | GB/s per socket; does it scale with threads? |
| `gpu bf16 L=32768` | 857 GB/s (5.6 ms) | expect somewhat lower (L40 ≈ 864 GB/s peak) |

Gate B on this machine ≈ **kernel GB/s ÷ PCIe GB/s**. The desktop ratio is ~2.1. If the server
reaches ≥ 100 GB/s per socket, hybrid should beat offload by ≥ 4× in the budget curve.

If the kernel stops scaling well below the `host bandwidth` line, it is compute- or
latency-bound on these cores. That is when the AVX-512 variant (not written yet; see
`l40-outlook.md`) is worth doing.

## 7. Copy-free baseline sweep (≈ 40 min)

```bash
python bench/hf_context_sweep.py --cache inplace \
  --out $REPO/results/context_sweep/${DATE}_l40_inplace 2>&1 | tee logs/l40_sweep_inplace.log
python bench/analyze_context_sweep.py results/context_sweep/${DATE}_l40_inplace
```

Optional, to confirm the `DynamicCache` artifact transfers to this GPU (same arch, sm_89):

```bash
python bench/hf_decode_profile.py --repo psf/black --lengths 1024 8192 32768 \
  --out $OUT/decode_profile.json
```

Expected (**estimate**): 32K decode ~21–23 ms/token in-place (desktop: 18.77 ms).

## 8. KV-budget curve (≈ 1 h)

Bind the process to the GPU-local node so both the CPU kernel's memory and the pinned offload
buffers sit next to the GPU:

```bash
for T in 16 32; do
numactl --cpunodebind=$GPU_NODE --membind=$GPU_NODE \
  python bench/hybrid_budget.py --repos psf/black \
  --lengths 8192 16384 32768 --fracs 0.5 0.25 0.1 \
  --modes hybrid offload offload_sync drop \
  --cpu-dtype bfloat16 --threads $T --runs 5 --decode-steps 64 \
  --out $REPO/results/hybrid_budget/${DATE}_l40_t$T 2>&1 | tee logs/l40_hybrid_t$T.log
done
```

Checks while it runs:

- Every exact mode must print `equal_full=True`. If not, stop: it is a correctness bug on this
  stack (driver/torch version), not a performance result.
- `cpu` ms per step for hybrid ≈ cold bytes ÷ the kernel GB/s from step 6. A big gap means
  threads are crossing sockets or competing with something else.

Desktop reference (ms/token, 32K): full 18.8; 10% budget: hybrid 114.5, offload 182.7.

## 9. Save and report

```bash
git add results/l40/$DATE results/context_sweep/${DATE}_l40_inplace results/hybrid_budget/${DATE}_l40_*
git commit -m "L40: Gate B microbenchmarks, in-place baseline, KV-budget curve"
git push
```

Then add an entry to `docs/findings.md` titled "(L40 server, measured)" — never merge L40 and 4090
numbers into one table without labelling the machine.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: torch` under `conda run` | shims; call the env's python by full path |
| `OSError … models/Qwen3-4B` | model not at `<repo>/models/Qwen3-4B`; set `DPINFER_MODEL=/path` |
| kernel build fails: `ninja` not found | `pip install ninja`; also needs `g++` |
| `CUDA driver version is insufficient` | torch wheel's CUDA newer than the driver; reinstall with an older `cuXXX` index |
| hybrid `equal_full=False` | report it; do not tune around it |
| very noisy timings | another process on the GPU or memory bus; check `nvidia-smi`, `top`, rerun |
| `numactl: command not found` | `sudo apt install numactl`, or `taskset -c` with that node's cores |

## Later on this machine (not in this runbook yet)

- Real concurrency: batch > 1 so the KV budget is genuinely binding (needs batched cache support
  in `HybridCache`).
- AVX-512 kernel variant and per-socket KV sharding (layers split across sockets).
- Larger models (14B BF16) where KV placement decides the feasible context.
