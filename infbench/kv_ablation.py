"""What does the KV cache actually buy?  (single request, Hugging Face transformers)

vLLM can't turn the KV cache off (PagedAttention *is* a KV cache), so this
micro-benchmark uses a hand-written greedy decode loop in transformers:

  with cache    : each step feeds ONE new token and reuses stored K/V
                  -> per-step cost ~ constant (memory-bound weight reads)
  without cache : each step re-runs attention over the WHOLE sequence
                  -> per-step cost grows with sequence length (O(n^2) total)

Run it while no vLLM server is holding the GPU.
"""
from __future__ import annotations

import time

import torch


@torch.inference_mode()
def decode_step_times(model, input_ids: torch.Tensor, new_tokens: int, use_cache: bool) -> list[float]:
    times = []
    ids = input_ids
    past = None
    nxt = None
    for step in range(new_tokens):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if use_cache:
            out = model(ids if past is None else nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
        else:
            out = model(ids, use_cache=False)
        nxt = out.logits[:, -1:].argmax(-1)
        ids = torch.cat([ids, nxt], dim=1)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def run(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", prompt_lens=(128, 512, 1024),
        new_tokens: int = 128, prompt_ids: list[int] | None = None):
    import pandas as pd
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).cuda().eval()
    rows, curves = [], {}
    for L in prompt_lens:
        ids = torch.tensor([prompt_ids[:L]] if prompt_ids else
                           torch.randint(100, 20000, (1, L)).tolist(), device="cuda")
        decode_step_times(model, ids, 4, True)  # warm-up
        for cache in (True, False):
            torch.cuda.reset_peak_memory_stats()
            t = decode_step_times(model, ids, new_tokens, cache)
            curves[(L, cache)] = t
            decode = t[1:]  # step 0 is the prefill
            rows.append({
                "prompt_len": L, "kv_cache": cache,
                "ttft_ms": t[0] * 1000,
                "mean_step_ms": 1000 * sum(decode) / len(decode),
                "last_step_ms": 1000 * decode[-1],
                "decode_tok_per_s": len(decode) / sum(decode),
                "peak_mem_mib": torch.cuda.max_memory_allocated() / 2**20,
            })
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows).round(2), curves


def plot(curves, save=None):
    import matplotlib.pyplot as plt
    from .analyze import PALETTE

    lens = sorted({L for L, _ in curves})
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, L in enumerate(lens):
        c = PALETTE[i % 8]
        ax.plot([x * 1000 for x in curves[(L, True)][1:]], color=c, label=f"prompt {L} - KV cache")
        ax.plot([x * 1000 for x in curves[(L, False)][1:]], color=c, linestyle="--",
                label=f"prompt {L} - no cache")
    ax.set_xlabel("Generated token index")
    ax.set_title("Per-token decode latency (ms)", loc="left", fontsize=10)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig
