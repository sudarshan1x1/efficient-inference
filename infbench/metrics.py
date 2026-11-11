"""Turn raw per-request timings + monitor samples into one summary row.

Units (per project spec): latencies in milliseconds, throughput in tokens/s.
"""
from __future__ import annotations

import numpy as np

from .loadgen import RequestResult


def _pct(xs, p):
    return float(np.percentile(xs, p)) if len(xs) else float("nan")


def _stats(prefix: str, xs_s: list[float]) -> dict:
    ms = [x * 1000 for x in xs_s]
    return {
        f"{prefix}_mean_ms": float(np.mean(ms)) if ms else float("nan"),
        f"{prefix}_p50_ms": _pct(ms, 50),
        f"{prefix}_p95_ms": _pct(ms, 95),
        f"{prefix}_p99_ms": _pct(ms, 99),
    }


def summarize(results: list[RequestResult], wall_s: float, samples: list[dict],
              counter_delta: dict, extra: dict | None = None) -> dict:
    ok = [r for r in results if r.ok]
    row: dict = dict(extra or {})
    row["n_requests"] = len(results)
    row["n_failed"] = len(results) - len(ok)
    row["duration_s"] = wall_s

    if ok:
        # Throughput over the actual busy window (first send -> last finish)
        t_first = min(r.start for r in ok)
        t_last = max(r.start + r.e2e for r in ok)
        busy = max(t_last - t_first, 1e-9)
        out_tok = sum(r.output_tokens for r in ok)
        in_tok = sum(r.prompt_tokens for r in ok)
        row["req_throughput_rps"] = len(ok) / busy
        row["output_tok_per_s"] = out_tok / busy
        row["total_tok_per_s"] = (out_tok + in_tok) / busy
        row["mean_prompt_tokens"] = in_tok / len(ok)
        row["mean_output_tokens"] = out_tok / len(ok)
        row["output_len_mismatch"] = sum(r.output_tokens != r.expected_output_tokens for r in ok)

        row.update(_stats("ttft", [r.ttft for r in ok]))
        # Steady state: drop the first cohort (all C users send at t=0 and queue
        # behind each other's prefill), which otherwise dominates TTFT p95.
        c = int((extra or {}).get("concurrency") or 0)
        steady = sorted(ok, key=lambda r: r.start)[c:] if c and len(ok) > c else []
        if steady:
            row.update({k.replace("ttft", "ttft_ss"): v for k, v in _stats("ttft", [r.ttft for r in steady]).items()})
            row.update({k.replace("e2e", "e2e_ss"): v for k, v in _stats("e2e", [r.e2e for r in steady]).items()})
        row.update(_stats("itl", [g for r in ok for g in r.itl]))       # pooled token gaps
        row.update(_stats("e2e", [r.e2e for r in ok]))
        tpot = [(r.e2e - r.ttft) / (r.output_tokens - 1) for r in ok if r.output_tokens > 1]
        row.update(_stats("tpot", tpot))                                  # per-request avg
        # What one user "feels": decode speed of their own stream
        row["per_user_tok_per_s"] = float(np.mean([1 / t for t in tpot])) if tpot else float("nan")

    def col(k):
        return [s[k] for s in samples if s.get(k) is not None]

    for k, agg in (("kv_usage", "mean"), ("kv_usage", "max"), ("running", "max"),
                   ("waiting", "max"), ("vram_used_mib", "max"), ("gpu_util_pct", "mean"),
                   ("cpu_pct", "mean"), ("ram_used_mib", "max")):
        v = col(k)
        if v:
            val = float(np.mean(v) if agg == "mean" else np.max(v))
            if k == "kv_usage":
                val *= 100  # 0..1 -> percent
                k = "kv_usage_pct"
            row[f"{k}_{agg}"] = val

    row["preemptions"] = counter_delta.get("preemptions", 0.0)
    q = counter_delta.get("prefix_queries", 0.0)
    row["prefix_hit_rate_pct"] = 100 * counter_delta.get("prefix_hits", 0.0) / q if q else 0.0
    return row
