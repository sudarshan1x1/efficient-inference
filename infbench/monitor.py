"""Background resource sampler: GPU (NVML), CPU/RAM (psutil) and vLLM's own
Prometheus /metrics (KV-cache usage, running/waiting queue, preemptions).

Why scrape /metrics instead of only nvidia-smi?  vLLM pre-allocates a fixed
slice of VRAM for the KV cache at startup (gpu_memory_utilization), so VRAM
"used" is nearly flat during a run. The interesting, *dynamic* memory signal
is what fraction of those KV blocks are actually occupied: vllm:kv_cache_usage_perc.
"""
from __future__ import annotations

import asyncio
import time

import aiohttp

GAUGES = {
    "kv_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    "running": ("vllm:num_requests_running", "infbench:num_requests_running"),
    "waiting": ("vllm:num_requests_waiting", "infbench:num_requests_waiting"),
}
COUNTERS = {
    "preemptions": ("vllm:num_preemptions_total", "vllm:num_preemptions"),
    "prefix_hits": ("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"),
    "prefix_queries": ("vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total"),
    "prompt_tokens": ("vllm:prompt_tokens_total",),
    "generation_tokens": ("vllm:generation_tokens_total",),
}


def parse_prometheus(text: str) -> dict[str, float]:
    """Minimal exposition-format parser: sums a metric over all its label sets."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            val = float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + val
    return out


def pick(metrics: dict[str, float], names: tuple[str, ...]) -> float | None:
    for n in names:
        if n in metrics:
            return metrics[n]
    return None


async def scrape(session: aiohttp.ClientSession, base_url: str) -> dict[str, float]:
    try:
        async with session.get(base_url + "/metrics", timeout=aiohttp.ClientTimeout(total=3)) as r:
            return parse_prometheus(await r.text())
    except Exception:  # noqa: BLE001
        return {}


def counter_snapshot(m: dict[str, float]) -> dict[str, float]:
    return {k: (pick(m, names) or 0.0) for k, names in COUNTERS.items()}


class ResourceMonitor:
    def __init__(self, base_url: str, interval_s: float = 1.0):
        self.base_url, self.interval = base_url, interval_s
        self.samples: list[dict] = []
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._nvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
        except Exception:  # noqa: BLE001
            print("[monitor] NVML unavailable - GPU columns will be empty")
        try:
            import psutil
            self._psutil = psutil
            psutil.cpu_percent(None)  # prime the counter
        except Exception:  # noqa: BLE001
            self._psutil = None

    def _hw(self) -> dict:
        s: dict = {}
        if self._nvml:
            nv, h = self._nvml
            mem = nv.nvmlDeviceGetMemoryInfo(h)
            s["vram_used_mib"] = mem.used / 2**20
            s["vram_total_mib"] = mem.total / 2**20
            s["gpu_util_pct"] = nv.nvmlDeviceGetUtilizationRates(h).gpu
        if self._psutil:
            s["cpu_pct"] = self._psutil.cpu_percent(None)
            s["ram_used_mib"] = self._psutil.virtual_memory().used / 2**20
        return s

    async def _loop(self, session):
        t0 = time.perf_counter()
        while not self._stop.is_set():
            m = await scrape(session, self.base_url)
            s = {"t": time.perf_counter() - t0, **self._hw()}
            for k, names in GAUGES.items():
                s[k] = pick(m, names)
            self.samples.append(s)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                pass

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        self.counters_before = counter_snapshot(await scrape(self._session, self.base_url))
        self._task = asyncio.create_task(self._loop(self._session))
        return self

    async def __aexit__(self, *exc):
        self._stop.set()
        await self._task
        after = counter_snapshot(await scrape(self._session, self.base_url))
        self.counter_delta = {k: after[k] - self.counters_before[k] for k in after}
        await self._session.close()
