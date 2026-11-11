"""Plots + comparison tables from summary.csv files (matplotlib, Colab-friendly)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Fixed categorical order; a config keeps its color in every figure.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 2, "font.size": 10,
})


def load(results_dir: str | Path, experiment: str) -> pd.DataFrame:
    df = pd.read_csv(Path(results_dir) / experiment / "summary.csv")
    return df.sort_values(["config", "concurrency"])


def _style(configs):
    return {c: (PALETTE[i % 8], MARKERS[i % 8]) for i, c in enumerate(configs)}


def sweep_grid(df: pd.DataFrame, panels=None, title: str = "", save: str | Path | None = None):
    """Small multiples: one panel per metric, x = concurrent users, one line per config."""
    ttft_col = "ttft_ss_p95_ms" if "ttft_ss_p95_ms" in df and df["ttft_ss_p95_ms"].notna().any() else "ttft_p95_ms"
    panels = panels or [
        (ttft_col, "TTFT p95, steady state (ms)" if ttft_col.startswith("ttft_ss") else "TTFT p95 (ms)", True),
        ("itl_p50_ms", "Inter-token latency p50 (ms)", False),
        ("e2e_p95_ms", "End-to-end latency p95 (ms)", True),
        ("output_tok_per_s", "Output throughput (tok/s)", False),
        ("kv_usage_pct_max", "Peak KV-cache usage (%)", False),
        ("vram_used_mib_max", "Peak VRAM used (MiB)", False),
    ]
    configs = list(dict.fromkeys(df["config"]))
    sty = _style(configs)
    ncol = 3
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow))
    for ax, (col, label, logy) in zip(axes.flat, panels):
        if col not in df:
            ax.set_visible(False)
            continue
        dashed_any = False
        for cfg in configs:
            d = df[df.config == cfg]
            color, m = sty[cfg]
            y, ls = d[col], "-"
            if col.startswith("ttft_ss") and d[col].iloc[1:].isna().all():
                # n == C at every level: no steady state exists, show all-request p95 dashed
                y, ls, dashed_any = d["ttft_p95_ms"], "--", True
            ax.plot(d.concurrency, y, marker=m, markersize=6, color=color, label=cfg, linestyle=ls)
        if dashed_any:
            label += "\n(dashed: all requests, no steady state)"
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_xlabel("Concurrent users")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xticks(sorted(df.concurrency.unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        if logy:
            ax.set_yscale("log")
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(configs), 5), frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    if title:
        fig.suptitle(title, y=1.07, fontsize=12)
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


def compare_table(df: pd.DataFrame, baseline: str, metrics=None) -> pd.DataFrame:
    """Each config vs baseline at every concurrency level, as a multiplier.
    Latency: baseline / config (>1 = faster). Throughput: config / baseline."""
    metrics = metrics or {"ttft_p95_ms": "lower", "e2e_p95_ms": "lower",
                          "itl_p95_ms": "lower", "output_tok_per_s": "higher"}
    base = df[df.config == baseline].set_index("concurrency")
    rows = []
    for cfg, d in df[df.config != baseline].groupby("config"):
        d = d.set_index("concurrency")
        for c in d.index.intersection(base.index):
            r = {"config": cfg, "concurrency": c}
            for m, better in metrics.items():
                if m in d and m in base:
                    r[f"{m} x"] = (base.at[c, m] / d.at[c, m]) if better == "lower" \
                        else (d.at[c, m] / base.at[c, m])
            rows.append(r)
    return pd.DataFrame(rows).round(2)


def headline_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["config", "concurrency", "ttft_p50_ms", "ttft_p95_ms", "ttft_ss_p95_ms", "itl_p50_ms",
            "itl_p95_ms", "e2e_p95_ms", "e2e_ss_p95_ms",
            "e2e_p99_ms", "output_tok_per_s", "req_throughput_rps", "per_user_tok_per_s",
            "kv_usage_pct_max", "waiting_max", "preemptions", "prefix_hit_rate_pct",
            "vram_used_mib_max", "gpu_util_pct_mean", "cpu_pct_mean", "n_failed"]
    return df[[c for c in cols if c in df]].round(1)


def timeline(results_dir, experiment, config, concurrency, save=None):
    """KV-cache usage and queue depth over one run - shows *when* the KV cache fills
    and requests start waiting. Two panels, shared x (never a dual y-axis)."""
    p = Path(results_dir) / experiment / config / f"c{concurrency}_monitor.csv"
    m = pd.read_csv(p)
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
    a.plot(m.t, m.kv_usage * 100, color=PALETTE[0])
    a.set_title(f"KV-cache usage (%) - {config}, C={concurrency}", loc="left", fontsize=10)
    a.set_ylim(0, 105)
    b.plot(m.t, m.running, color=PALETTE[0], label="running")
    b.plot(m.t, m.waiting, color=PALETTE[1], label="waiting")
    b.set_title("Requests running vs waiting", loc="left", fontsize=10)
    b.set_xlabel("Seconds since level start")
    b.legend(frameon=False)
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


def bar_compare(df: pd.DataFrame, metric: str, label: str, save=None):
    """Grouped bars: configs side by side at each concurrency (for 2-config experiments)."""
    configs = list(dict.fromkeys(df["config"]))
    levels = sorted(df.concurrency.unique())
    w = 0.8 / len(configs)
    fig, ax = plt.subplots(figsize=(1.8 * len(levels) + 2, 3.4))
    for i, cfg in enumerate(configs):
        d = df[df.config == cfg].set_index("concurrency").reindex(levels)
        xs = [k + (i - (len(configs) - 1) / 2) * w for k in range(len(levels))]
        ax.bar(xs, d[metric], width=w * 0.92, color=PALETTE[i % 8], label=cfg)
    ax.set_xticks(range(len(levels)), [str(c) for c in levels])
    ax.set_xlabel("Concurrent users")
    ax.set_title(label, loc="left", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


def itl_histogram(results_dir, experiment, config, levels=(25, 50, 100), save=None):
    """Distribution of token gaps. A bimodal shape (most gaps ~tens of ms plus a
    cluster at ~400 ms) is the signature of generation stalls: decode steps that
    also had to process newcomers' prompts."""
    import json
    import numpy as np
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.logspace(np.log10(3), np.log10(3000), 70)
    for i, c in enumerate(levels):
        p = Path(results_dir) / experiment / config / f"c{c}_requests.jsonl"
        if not p.exists():
            continue
        g = np.array([x for line in open(p) for x in json.loads(line)["itl"]]) * 1000
        ax.hist(g, bins=bins, histtype="step", linewidth=2, color=PALETTE[i % 8],
                label=f"C={c}  ({100 * (g > 200).mean():.1f}% of gaps > 200 ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gap between consecutive tokens (ms, log scale)")
    ax.set_title(f"Inter-token gap distribution - {config}", loc="left", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


def metric_vs(df: pd.DataFrame, x: str, y: str, group: str, label: str, xlabel: str,
              logx: bool = False, save=None):
    """One line per group (e.g. engine) of metric y against a numeric x (e.g. isl)."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, (g, d) in enumerate(df.sort_values(x).groupby(group, sort=False)):
        ax.plot(d[x], d[y], marker=MARKERS[i % 8], color=PALETTE[i % 8], label=str(g))
    if logx:
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df[x].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(xlabel)
    ax.set_title(label, loc="left", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


def steady_state(results_dir, experiment, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Recompute steady-state TTFT/E2E from raw per-request files, excluding the first
    C requests of each level (the t=0 burst cohort). Works on results collected
    before these columns existed in summary.csv."""
    import json
    import numpy as np
    df = load(results_dir, experiment) if df is None else df.copy()
    for i, r in df.iterrows():
        p = Path(results_dir) / experiment / r["config"] / f"c{int(r['concurrency'])}_requests.jsonl"
        if not p.exists():
            continue
        reqs = sorted((json.loads(l) for l in open(p)), key=lambda x: x["start"])
        steady = [q for q in reqs[int(r["concurrency"]):] if q.get("ok")]
        if not steady:
            continue
        for key, col in (("ttft", "ttft_ss"), ("e2e", "e2e_ss")):
            v = np.array([q[key] for q in steady]) * 1000
            df.loc[i, f"{col}_p50_ms"] = np.percentile(v, 50)
            df.loc[i, f"{col}_p95_ms"] = np.percentile(v, 95)
    return df
