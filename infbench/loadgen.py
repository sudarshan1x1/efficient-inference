"""Asynchronous load generator (streaming, OpenAI /v1/completions).

Two client policies:

* "closed"  - C virtual users. Each user sends its next request the moment the
              previous one finishes (an asyncio.Semaphore of size C), exactly
              like vLLM's benchmark_serving --max-concurrency used by
              vllm-benchmarking-studio. The server is free to continuously
              batch whatever is in flight.
* "waves"   - emulates STATIC batching from the client: send C requests, wait
              for ALL of them, then send the next C. Slots freed by short
              requests sit idle until the longest one in the wave finishes.

Timing is measured per streamed chunk with time.perf_counter():
  TTFT = first chunk - send time
  ITL  = gaps between consecutive chunks
  E2E  = last byte - send time
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field

import aiohttp

from .workload import Request


@dataclass
class RequestResult:
    idx: int
    ok: bool = False
    error: str = ""
    start: float = 0.0          # seconds since level start
    ttft: float = 0.0           # seconds
    e2e: float = 0.0            # seconds
    itl: list[float] = field(default_factory=list)  # seconds
    prompt_tokens: int = 0
    output_tokens: int = 0
    expected_output_tokens: int = 0

    def to_row(self) -> dict:
        d = asdict(self)
        d["itl"] = [round(x, 6) for x in self.itl]
        return d


async def _send(session: aiohttp.ClientSession, url: str, model: str, req: Request,
                idx: int, t_level0: float) -> RequestResult:
    res = RequestResult(idx=idx, expected_output_tokens=req.output_len)
    payload = {
        "model": model,
        "prompt": req.prompt_ids,          # token ids -> input length is exact
        "max_tokens": req.output_len,
        "min_tokens": req.output_len,      # systalyze "mintok": exact output length
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    res.start = t0 - t_level0
    last = None
    n_chunks = 0
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                res.error = f"HTTP {resp.status}: {(await resp.text())[:300]}"
                return res
            async for raw in resp.content:   # aiohttp yields one SSE line at a time
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                now = time.perf_counter()
                if chunk.get("choices"):
                    # count every token chunk, even if its text is "" (partial UTF-8)
                    if last is None:
                        res.ttft = now - t0
                    else:
                        res.itl.append(now - last)
                    last = now
                    n_chunks += 1
                usage = chunk.get("usage")
                if usage:
                    res.prompt_tokens = usage.get("prompt_tokens", 0)
                    res.output_tokens = usage.get("completion_tokens", 0)
        res.e2e = time.perf_counter() - t0
        if not res.output_tokens:
            res.output_tokens = n_chunks
        res.ok = last is not None
        if not res.ok:
            res.error = "no tokens streamed"
    except Exception as exc:  # noqa: BLE001
        res.error = f"{type(exc).__name__}: {exc}"
        res.e2e = time.perf_counter() - t0
    return res


async def run_load(base_url: str, requests: list[Request], concurrency: int,
                   policy: str = "closed", model: str = "bench",
                   request_timeout_s: int = 1800) -> tuple[list[RequestResult], float]:
    """Fire `requests` at the server with `concurrency` users. Returns (results, wall_s)."""
    url = base_url + "/v1/completions"
    conn = aiohttp.TCPConnector(limit=0)                # no client-side connection cap
    timeout = aiohttp.ClientTimeout(total=request_timeout_s)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        t_level0 = time.perf_counter()
        if policy == "closed":
            sem = asyncio.Semaphore(concurrency)

            async def user(i, r):
                async with sem:
                    return await _send(session, url, model, r, i, t_level0)

            results = await asyncio.gather(*(user(i, r) for i, r in enumerate(requests)))
        elif policy == "waves":
            results = []
            for w in range(0, len(requests), concurrency):
                wave = requests[w:w + concurrency]
                results += await asyncio.gather(
                    *(_send(session, url, model, r, w + j, t_level0) for j, r in enumerate(wave)))
        else:
            raise ValueError(f"unknown policy {policy!r}")
        wall = time.perf_counter() - t_level0
    return list(results), wall


async def warmup(base_url: str, requests: list[Request], model: str = "bench") -> None:
    """A few throwaway requests: first-call costs (CUDA graph replay paths,
    allocator growth, tokenizer threads) shouldn't land in the C=1 numbers."""
    await run_load(base_url, requests, concurrency=len(requests), model=model)
