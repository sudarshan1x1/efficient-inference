"""Experiment driver: config matrix x concurrency levels -> results on Google Drive.

Layout (one folder per experiment, like systalyze's runs/<name>/):
  <results_dir>/<experiment>/summary.csv              one row per (config, concurrency)
  <results_dir>/<experiment>/<config>/command.txt     exact vllm serve line
  <results_dir>/<experiment>/<config>/server.log
  <results_dir>/<experiment>/<config>/c<C>_requests.jsonl   raw per-request timings
  <results_dir>/<experiment>/<config>/c<C>_monitor.csv      1 Hz GPU/CPU/KV samples

Colab sessions die. The runner is resumable: (config, concurrency) pairs
already present in summary.csv are skipped on the next run.
"""
from __future__ import annotations

import asyncio
import csv
import json
import time
from pathlib import Path

import nest_asyncio
import yaml

from . import workload as wl
from .loadgen import run_load, warmup
from .metrics import summarize
from .monitor import ResourceMonitor
from .server import VLLMServer, config_to_args

nest_asyncio.apply()  # Colab already runs an event loop; let asyncio.run() nest inside it


def load_plan(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _done(summary_csv: Path) -> set[tuple[str, int]]:
    if not summary_csv.exists():
        return set()
    with open(summary_csv) as f:
        return {(r["config"], int(r["concurrency"])) for r in csv.DictReader(f)}


def _append(summary_csv: Path, row: dict) -> None:
    rows = []
    if summary_csv.exists():
        with open(summary_csv) as f:
            rows = list(csv.DictReader(f))
    rows.append({k: v for k, v in row.items()})
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _n_requests(c: int, wcfg: dict, cfg: dict) -> int:
    per_user = cfg.get("requests_per_user", wcfg.get("requests_per_user", 2))
    n = max(cfg.get("min_requests", wcfg.get("min_requests", 20)), per_user * c)
    if cfg.get("max_requests"):
        n = min(n, cfg["max_requests"])
    return max(n, 1)


def _build_requests(kind: str, stream, corpus, n: int, wcfg: dict, exp: dict):
    isl, osl = wcfg["isl"], wcfg["osl"]
    if kind == "fixed":
        return wl.build_fixed(stream, n, isl, osl)
    if kind == "shared_prefix":
        return wl.build_shared_prefix(stream, n, exp["prefix_len"], isl - exp["prefix_len"], osl)
    if kind == "variable_osl":
        return wl.build_variable_osl(stream, corpus, n, isl, scale=exp.get("osl_scale", 0.5),
                                     cap=exp.get("osl_cap", 256), seed=n)
    raise ValueError(kind)


def run_experiment(plan: dict, exp_name: str, results_dir: str | Path,
                   only: list[str] | None = None) -> Path:
    from transformers import AutoTokenizer

    exp = plan["experiments"][exp_name]
    wcfg = plan["workload"]
    out = Path(results_dir) / exp_name
    out.mkdir(parents=True, exist_ok=True)
    summary_csv = out / "summary.csv"
    done = _done(summary_csv)

    tok = AutoTokenizer.from_pretrained(plan.get("tokenizer", plan["model"]))
    corpus = wl.load_corpus(Path(results_dir) / "_cache")
    base_stream = wl.TokenStream(tok, corpus, seed=wcfg.get("seed", 0))

    # Give every concurrency level its own, big-enough region of the token stream
    # (sized by how many prompts that level needs), so levels never share prompts.
    all_levels = sorted(set(exp["levels"]) | {l for cf in exp["configs"] for l in cf.get("levels", [])})
    isl_max = max([wcfg["isl"]] + [cf.get("isl", 0) for cf in exp["configs"]])
    offsets, cur = {}, 0
    for c in all_levels:
        offsets[c] = cur
        cur += max(wcfg.get("min_requests", 20), wcfg.get("requests_per_user", 4) * c) * isl_max
    if cur > len(base_stream.ids) - 4096:
        print(f"[runner] note: experiment needs {cur:,} prompt tokens, corpus has "
              f"{len(base_stream.ids):,}; some prompts repeat (harmless unless prefix caching is on)")

    server: VLLMServer | None = None
    server_key = None
    try:
        for cfg_over in exp["configs"]:
            cfg = {**plan.get("defaults", {}), **cfg_over}
            name = cfg["name"]
            if cfg.get("enabled") is False or (only and name not in only):
                continue
            levels = cfg.get("levels", exp["levels"])
            todo = [c for c in levels if (name, c) not in done]
            if not todo:
                print(f"[runner] {exp_name}/{name}: all levels done, skipping")
                continue

            model = cfg.get("model", plan["model"])
            args = config_to_args(cfg)
            cdir = out / name
            cdir.mkdir(exist_ok=True)

            engine = cfg.get("engine", "vllm")
            key = (engine, model, tuple(args))
            if key != server_key:              # reuse the server if nothing changed
                if server:
                    server.stop()
                server = VLLMServer(model, args, cdir / "server.log", port=plan.get("port", 8000),
                                    engine=engine)
                (cdir / "command.txt").write_text(" ".join(server.command) + "\n")
                server.start(timeout_s=plan.get("server_timeout_s", 900))
                server_key = key
                # warm-up prompts come from the END of the stream so they never
                # overlap (and pre-cache) any measured prompt
                n_wu = wcfg.get("warmup_requests", 4)
                wu = base_stream.view(len(base_stream.ids) - n_wu * wcfg["isl"] - 1)
                asyncio.run(warmup(server.base_url, wl.build_fixed(wu, n_wu, wcfg["isl"], 16)))
            kv_cap = server.kv_capacity()

            # a config may override the prompt/output length (prompt-length sweep)
            wc = {**wcfg, **{k: cfg[k] for k in ("isl", "osl") if k in cfg}}
            for c in todo:
                # Every config replays the SAME prompts at a given level -> apples-to-apples
                stream = base_stream.view(offsets[c])
                n = _n_requests(c, wcfg, cfg)
                reqs = _build_requests(exp.get("workload", "fixed"), stream, corpus, n, wc, exp)
                policy = cfg.get("policy", exp.get("policy", "closed"))
                print(f"[runner] {exp_name}/{name}  C={c}  n={n}  policy={policy}")

                async def _level():
                    async with ResourceMonitor(server.base_url, plan.get("monitor_interval_s", 1.0)) as mon:
                        res, wall = await run_load(server.base_url, reqs, c, policy=policy)
                    return res, wall, mon

                res, wall, mon = asyncio.run(_level())
                row = summarize(res, wall, mon.samples, mon.counter_delta, extra={
                    "experiment": exp_name, "config": name, "concurrency": c, "policy": policy,
                    "engine": engine,
                    "model": model, "isl": wc["isl"],
                    "osl": wc["osl"] if exp.get("workload") != "variable_osl" else "variable",
                    "max_num_seqs": cfg.get("max_num_seqs"),
                    "max_num_batched_tokens": cfg.get("max_num_batched_tokens") or "default",
                    "gpu_memory_utilization": cfg.get("gpu_memory_utilization"),
                    "quantization": cfg.get("quantization") or "none",
                    "prefix_caching": bool(cfg.get("prefix_caching")),
                    "kv_cache_dtype": cfg.get("kv_cache_dtype") or "auto",
                    **kv_cap, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                with open(cdir / f"c{c}_requests.jsonl", "w") as f:
                    for r in res:
                        f.write(json.dumps(r.to_row()) + "\n")
                if mon.samples:
                    keys = sorted({k for s in mon.samples for k in s})
                    with open(cdir / f"c{c}_monitor.csv", "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=keys)
                        w.writeheader()
                        w.writerows(mon.samples)
                _append(summary_csv, row)
                _print_row(row)
                if row["n_failed"]:
                    first_err = next(r.error for r in res if not r.ok)
                    print(f"   !! {row['n_failed']} failed requests, e.g. {first_err}")
    finally:
        if server:
            server.stop()
    return summary_csv


def _print_row(r: dict) -> None:
    print(f"   TTFT p95 {r.get('ttft_p95_ms', float('nan')):8.1f} ms | "
          f"ITL p95 {r.get('itl_p95_ms', float('nan')):6.1f} ms | "
          f"E2E p95 {r.get('e2e_p95_ms', float('nan')):8.1f} ms | "
          f"{r.get('output_tok_per_s', float('nan')):7.1f} out tok/s | "
          f"KV max {r.get('kv_usage_pct_max', float('nan')):5.1f}% | "
          f"preempt {r.get('preemptions', 0):.0f}")
