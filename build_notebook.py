"""Generate Efficient_AI_Inference.ipynb with the framework embedded via %%writefile,
so the notebook is self-contained on Colab (no repo clone needed)."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n")})


md("""
# Efficient AI Inference System — LLM serving under concurrent load (Colab T4)

**Goal.** Measure how a vLLM server behaves as concurrent users grow (1 → 10 → 25 → 50 → 100) and
quantify what **continuous batching**, **KV caching** and **quantization** buy you, in TTFT,
inter-token latency (ITL), P95/P99 end-to-end latency, throughput, KV-cache usage and VRAM.

**What came from the two reference repos**

| Idea | Source | How it's used here |
|---|---|---|
| Realistic coding prompts; exact output length via `min_tokens = max_tokens` | systalyze `workload/aws-p50p90-v1-mintok` | Prompts are sliced from that file into windows of **exactly 512 tokens**, sent as token IDs; outputs are held to exactly 128 tokens |
| Prefix caching OFF for the main sweep, temperature 0, one launch line per config saved with results | systalyze `configs/*/launch.sh`, `row.json` | `configs/experiments.yaml` + `command.txt` per config |
| Sweep driver over a concurrency × num-prompts grid, results as one file per case, then aggregate | vllm-benchmarking-studio `combos.yaml`, `run_sweep.py`, `aggregate_result.py` | `infbench/runner.py` → `summary.csv` (resumable) |
| Semaphore-bounded concurrency (`--max-concurrency`) + streaming TTFT/ITL measurement | studio `benchmark_serving.py` / `backend_request_func.py` | `infbench/loadgen.py` (asyncio + aiohttp) |

**Run order:** 1 → 6 once per session, then the experiment cells. Every experiment is resumable: if Colab disconnects, re-run setup and the same cell — finished (config, concurrency) pairs are skipped.
""")

md("## 1 · Check the GPU\nYou need a **T4 GPU** runtime (Runtime → Change runtime type).")
code("!nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv")

md("## 2 · Mount Google Drive (results survive disconnects)")
code("""
from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = '/content/drive/MyDrive/efficient-inference'   # code + results live here
RESULTS_DIR = f'{PROJECT_DIR}/results'
import os; os.makedirs(f'{PROJECT_DIR}/infbench', exist_ok=True); os.makedirs(f'{PROJECT_DIR}/configs', exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
print(PROJECT_DIR)
""")

md("""
## 3 · Install
vLLM brings its own PyTorch build. If the cell ends with a message asking to restart, do
**Runtime → Restart session**, then continue from step 2 (skip this cell).

If the latest vLLM fails to start on the T4 (sm 7.5 is an older architecture and support can
change between releases), set `VLLM_VERSION` to a release known to work and re-run.
""")
code("""
VLLM_VERSION = ''   # e.g. '==0.11.0' to pin; '' = latest
!pip install -q "vllm{VLLM_VERSION}" nest_asyncio nvidia-ml-py aiohttp pyyaml
!python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
""")

md("""
## 4 · Write the framework to Drive
Each cell below writes one module. You can edit these files directly in Drive (or push them to
your GitHub and replace this section with a `git clone`).
""")
for rel in ["infbench/__init__.py", "infbench/workload.py", "infbench/server.py",
            "infbench/monitor.py", "infbench/loadgen.py", "infbench/metrics.py",
            "infbench/runner.py", "infbench/analyze.py", "infbench/kv_ablation.py",
            "infbench/hf_server.py",
            "configs/experiments.yaml"]:
    src = (ROOT / rel).read_text()
    code(f"%%writefile {{PROJECT_DIR}}/{rel}\n{src}")

md("## 5 · Import and load the plan")
code("""
import sys, importlib, copy, asyncio
import nest_asyncio
nest_asyncio.apply()          # Colab already runs an event loop
sys.path.insert(0, PROJECT_DIR)

import infbench.runner as runner, infbench.analyze as analyze
importlib.reload(runner); importlib.reload(analyze)

plan = runner.load_plan(f'{PROJECT_DIR}/configs/experiments.yaml')
for name, exp in plan['experiments'].items():
    print(f"{name:20s} levels={exp['levels']}  configs={[c['name'] for c in exp['configs'] if c.get('enabled', True)]}")
""")

md("""
## 6 · Smoke test (~3 min)
One config, two concurrency levels, into a separate `smoke/` folder. Confirms vLLM boots on the
T4, streaming works, `/metrics` is scraped, and every request got exactly 128 output tokens
(`output_len_mismatch` must be 0).
""")
code("""
smoke = copy.deepcopy(plan)
smoke['experiments'] = {'smoke': {**plan['experiments']['concurrency_sweep'], 'levels': [1, 10],
                                  'configs': [{'name': 'cb_fp16'}]}}
csv_path = runner.run_experiment(smoke, 'smoke', RESULTS_DIR)
df = analyze.load(RESULTS_DIR, 'smoke')
df[['config','concurrency','ttft_p95_ms','itl_p95_ms','e2e_p95_ms','output_tok_per_s',
    'kv_usage_pct_max','vram_used_mib_max','kv_capacity_tokens','output_len_mismatch','n_failed']]
""")

md("""
## 7 · Experiment 1 — concurrency sweep: baseline vs optimizations (~50–60 min)

| config | what changes | the question it answers |
|---|---|---|
| `baseline_hf_naive` | plain `transformers` + HTTP, global lock | How does a typical first-version chatbot backend behave when users overlap? |
| `vllm_no_batching` | vLLM, `--max-num-seqs 1` | How much comes from a better **engine** alone (CUDA graphs, fused kernels), with batching still off? |
| `cb_fp16` | default (256 seqs) | How much does **continuous batching + paged KV cache** help? |
| `cb_fp16_small_kv` | `--gpu-memory-utilization 0.35` | What happens when the **KV cache runs out** (queueing, preemption)? |
| `cb_awq_int4` | AWQ 4-bit weights | Does **quantization** help speed, memory, or both — and at which load? |
| `cb_gptq_int8`, `cb_fp16_kv_fp8` | disabled by default | Optional extras: set `enabled: true` in the YAML |
""")
code("runner.run_experiment(plan, 'concurrency_sweep', RESULTS_DIR)")

md("""
## 8 · Experiment 2 — KV-cache reuse: automatic prefix caching (~8 min)
Every request shares a 448-token prefix (think: system prompt or RAG context) + 64 unique tokens.
With prefix caching on, vLLM reuses the KV blocks of the shared prefix instead of recomputing them,
so prefill shrinks from 512 to ~64 tokens per request. Watch **TTFT** and `prefix_hit_rate_pct`.
""")
code("runner.run_experiment(plan, 'prefix_caching', RESULTS_DIR)")

md("""
## 9 · Experiment 3 — static vs continuous batching with variable output lengths (~8 min)
With fixed output lengths every request finishes together, so static and continuous batching look
alike. Real traffic has variable lengths (here sampled from the systalyze distribution). In
**static waves**, a slot freed by a short answer sits idle until the longest answer in the wave
finishes; continuous batching refills it immediately.
""")
code("runner.run_experiment(plan, 'batching_policy', RESULTS_DIR)")

md("""
## 10 · Experiment 4 — what the KV cache itself buys (Hugging Face, single request, ~5 min)
vLLM can't switch its KV cache off, so this uses a hand-written decode loop in `transformers`.
Without a cache, every new token re-processes the whole sequence, so per-token latency grows with
length; with a cache it stays flat. Runs only when no vLLM server holds the GPU.
""")
code("""
import infbench.kv_ablation as kv
kv_df, curves = kv.run(plan['model'], prompt_lens=(128, 512, 1024), new_tokens=128)
kv_df.to_csv(f'{RESULTS_DIR}/kv_ablation.csv', index=False)
kv.plot(curves, save=f'{RESULTS_DIR}/kv_ablation.png')
kv_df
""")

md("""
## 11 · Experiment 5 — generation stalls vs chunked-prefill budget (~15 min)
Experiment 1 showed most token gaps at 30–70 ms but a cluster at ~430 ms (16% of gaps at C=100).
Those are **decode steps that also processed newcomers' prompts**: vLLM packs up to
`max_num_batched_tokens` (2048 here, ≈ four 512-token prompts) into one step, and everyone already
generating waits for it. A smaller budget shortens each stall but spreads prompts over more steps.
Prediction: ITL p95/p99 falls with the budget while TTFT rises — a direct TTFT ↔ smoothness trade-off.
""")
code("runner.run_experiment(plan, 'chunked_prefill', RESULTS_DIR)")

md("""
## 12 · Experiment 6 — prefill cost vs prompt length, vLLM vs HF (~12 min)
At one user, HF's TTFT (74 ms) beat vLLM's (~125 ms) for a 512-token prompt. vLLM on the T4 uses the
`TRITON_ATTN` backend (FlashAttention needs sm80+). If vLLM's gap grows with prompt length, the
attention kernel is the cause; if it stays constant, it's fixed per-request overhead (API, scheduler).
""")
code("runner.run_experiment(plan, 'prompt_length', RESULTS_DIR)")

md("""
## 13 · Experiment 7 — model size: 0.5B vs 1.5B vs 3B (~20 min)
Decode reads every weight once per step, so single-user ITL should grow with weight bytes. The 0.5B
model tests where fixed overheads (kernel launches, scheduling) set a floor that bandwidth can't explain.
""")
code("runner.run_experiment(plan, 'model_size', RESULTS_DIR)")

md("""
## 14 · Experiment 8 — admission control: max concurrent sequences at C=100 (~12 min)
Capping `max_num_seqs` keeps the batch small: admitted users decode faster, the rest queue.
Look at the per-user speed (`per_user_tok_per_s`) vs TTFT vs total throughput.
""")
code("runner.run_experiment(plan, 'admission_control', RESULTS_DIR)")

md("## 15 · Analysis")
code("""
sweep = analyze.steady_state(RESULTS_DIR, 'concurrency_sweep')   # adds steady-state TTFT/E2E columns
analyze.sweep_grid(sweep, title='Qwen2.5-1.5B on T4 — 512 in / 128 out tokens',
                   save=f'{RESULTS_DIR}/sweep_grid.png')
analyze.headline_table(sweep)
""")
code("""
# Multipliers vs the naive HF baseline (>1 = better)
analyze.compare_table(sweep, baseline='baseline_hf_naive')
""")
code("""
# Quantization in isolation: AWQ vs FP16, both with continuous batching
analyze.compare_table(sweep[sweep.config.isin(['cb_fp16', 'cb_awq_int4'])], baseline='cb_fp16')
""")
code("""
# When does the KV cache fill up? (look for 'waiting' > 0 and preemptions)
analyze.timeline(RESULTS_DIR, 'concurrency_sweep', 'cb_fp16_small_kv', 100,
                 save=f'{RESULTS_DIR}/timeline_small_kv_c100.png');
sweep[['config','concurrency','kv_capacity_tokens','kv_usage_pct_max','waiting_max','preemptions']]
""")
code("""
pc = analyze.load(RESULTS_DIR, 'prefix_caching')
analyze.bar_compare(pc, 'ttft_p95_ms', 'TTFT p95 (ms) — shared 448-token prefix',
                    save=f'{RESULTS_DIR}/prefix_ttft.png')
pc[['config','concurrency','ttft_p50_ms','ttft_p95_ms','e2e_p95_ms','output_tok_per_s','prefix_hit_rate_pct']].round(1)
""")
code("""
bp = analyze.load(RESULTS_DIR, 'batching_policy')
analyze.bar_compare(bp, 'output_tok_per_s', 'Output throughput (tok/s) — variable output lengths',
                    save=f'{RESULTS_DIR}/batching_policy_throughput.png')
bp[['config','concurrency','output_tok_per_s','e2e_p95_ms','e2e_p99_ms','ttft_p95_ms']].round(1)
""")
code("""
# Generation stalls: bimodal token-gap distribution (experiment 1)
analyze.itl_histogram(RESULTS_DIR, 'concurrency_sweep', 'cb_fp16', levels=(25, 50, 100),
                      save=f'{RESULTS_DIR}/itl_histogram_cb_fp16.png');
""")
code("""
# Experiment 5: chunked-prefill budget -> TTFT vs ITL tail
cp = analyze.load(RESULTS_DIR, 'chunked_prefill')
cp['budget'] = cp.config.str.replace('mnbt_', '').astype(int)
for c in (50, 100):
    d = cp[cp.concurrency == c]
    analyze.metric_vs(d.melt(id_vars='budget', value_vars=['itl_p99_ms', 'ttft_p95_ms']),
                      'budget', 'value', 'variable', f'C={c}: ITL p99 vs TTFT p95 (ms)',
                      'max_num_batched_tokens', logx=True, save=f'{RESULTS_DIR}/chunked_prefill_c{c}.png')
cp[['config','concurrency','ttft_p50_ms','ttft_p95_ms','itl_p50_ms','itl_p95_ms','itl_p99_ms','output_tok_per_s']].round(1)
""")
code("""
# Experiment 6: TTFT at one user vs prompt length, per engine
pl = analyze.load(RESULTS_DIR, 'prompt_length')
one = pl[pl.concurrency == 1]
analyze.metric_vs(one, 'isl', 'ttft_p50_ms', 'engine', 'TTFT p50 at 1 user (ms)', 'Prompt tokens',
                  logx=True, save=f'{RESULTS_DIR}/prompt_length_ttft.png')
piv = one.pivot_table(index='isl', columns='engine', values='ttft_p50_ms')
piv['vllm_minus_hf_ms'] = piv['vllm'] - piv['hf']
piv.round(1)   # gap growing with isl -> attention kernel; flat gap -> fixed overhead
""")
code("""
# Experiment 7: model size. Expected decode floor = weight bytes / 320 GB/s (T4 peak bandwidth)
ms = analyze.load(RESULTS_DIR, 'model_size')
params_b = {'qwen2.5_0.5b': 0.494, 'qwen2.5_1.5b': 1.544, 'qwen2.5_3b': 3.086}
ms['weights_gb'] = ms.config.map(params_b) * 2           # fp16 = 2 bytes/param
ms['bandwidth_floor_ms'] = ms.weights_gb / 320 * 1000
ms['achieved_GBps'] = ms.weights_gb / (ms.itl_p50_ms / 1000)
ms[['config','concurrency','weights_gb','itl_p50_ms','bandwidth_floor_ms','achieved_GBps',
    'output_tok_per_s','kv_capacity_tokens']].round(1)
""")
code("""
# Experiment 8: admission control at C=100
ac = analyze.load(RESULTS_DIR, 'admission_control')
ac['cap'] = ac.config.str.replace('max_seqs_', '').astype(int)
ac.sort_values('cap')[['config','ttft_p50_ms','ttft_p95_ms','itl_p50_ms','per_user_tok_per_s',
                       'e2e_p95_ms','output_tok_per_s','waiting_max']].round(1)
""")
code("""
# Memory: where the T4's 15 GB went, and why paging matters
import pandas as pd
from pathlib import Path
from infbench.server import VLLMServer
mem = [{'config': d.name, **VLLMServer('x', [], d / 'server.log').kv_capacity()}
       for d in sorted(Path(f'{RESULTS_DIR}/concurrency_sweep').iterdir()) if (d / 'server.log').exists()]
display(pd.DataFrame(mem))     # parsed from each config's server.log

# Fragmentation: a server that reserves max_model_len of KV per request vs vLLM's 16-token pages
bytes_per_tok = 28 * 2 * 2 * 128 * 2       # layers x (K,V) x kv_heads x head_dim x fp16 = 28,672 B
cap = int(sweep[sweep.config == 'cb_fp16'].kv_capacity_tokens.iloc[0])
used, max_len, block = 512 + 128, 2048, 16
print(f"KV per token: {bytes_per_tok/1024:.0f} KiB | KV pool: {cap*bytes_per_tok/2**30:.2f} GiB = {cap:,} tokens")
print(f"Reserve-max-length server: {cap // max_len} requests fit, {100*(1-used/max_len):.0f}% of each reservation unused")
print(f"Paged (vLLM):              {cap // used} requests fit, waste < 1 block = {100*(block-1)/used:.1f}% worst case")
""")

md("""
## 16 · Turning numbers into findings
Answer each with a number from the tables, then the *why*:

1. **Where is the knee?** At what concurrency does `cb_fp16` throughput stop growing while ITL keeps rising?
   (Decode is memory-bandwidth bound: each step reads all weights once, so more sequences per step are
   almost free — until the step becomes compute-bound or the KV cache fills.)
2. **Engine vs batching:** split the total gain into two steps — `baseline_hf_naive` → `vllm_no_batching`
   (engine: CUDA graphs, fused kernels) and `vllm_no_batching` → `cb_fp16` (batching). Where does the
   baselines' latency come from? (TTFT ≈ queueing time — check `waiting_max`.)
3. **KV-cache capacity:** `kv_capacity_tokens` for `cb_fp16` vs `cb_fp16_small_kv`. At C=100 × 640 tokens,
   does demand exceed capacity? What did that do to TTFT p95 and preemptions?
4. **Quantization:** AWQ frees ~2 GB of weights → more KV capacity. Is single-user ITL faster or slower
   than FP16 on the T4 (dequantisation cost vs fewer bytes read)? Does that change at C=100?
5. **Prefix caching:** TTFT reduction and hit rate; why does it help prefill but not ITL?
6. **Static vs continuous:** throughput gap and P99 latency with variable output lengths.
7. **Generation stalls:** what share of token gaps are ~400 ms stalls at C=25/50/100, and how does the
   chunked-prefill budget trade ITL p99 against TTFT?
8. **Prefill kernel:** does vLLM's TTFT gap vs HF grow with prompt length (attention kernel) or stay flat?
9. **Model size:** achieved GB/s per model — does the 0.5B model hit an overhead floor?
10. **Admission control:** which `max_num_seqs` gives the best per-user speed without exploding TTFT?
11. **Memory:** weights vs KV vs CUDA graphs, and how many requests fit with paging vs max-length reservations.
12. **Measurement caveats:** `gpu_util_pct` is ~100% even at 1 user, because NVML counts time *any* kernel
    runs, not how busy the GPU is. Use achieved GB/s and tok/s instead. CPU was ~30%, so not a bottleneck.
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                    "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:   # ipynb stores source as a list of lines
    c["source"] = c["source"].splitlines(True)
out = ROOT / "Efficient_AI_Inference.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
