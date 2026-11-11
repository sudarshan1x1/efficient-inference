"""Naive Hugging Face transformers server - the "how most LLM apps start" baseline.

This is how a typical first-version chatbot backend serves a model: load it with
transformers, put an HTTP endpoint in front, and run generation for ONE request at
a time (a global lock). It does use a KV cache (model forward with past_key_values),
but has none of the serving-engine tricks:

  * no batching       - other users wait in the lock's queue
  * contiguous KV     - each request's cache grows by torch.cat-style reallocation,
                        no paging, no sharing
  * eager PyTorch     - one Python-driven forward per token, no CUDA graphs

Speaks the same streaming /v1/completions protocol as vLLM (token-id prompts,
max_tokens, min_tokens, usage chunk), so infbench's load generator and monitor
work unchanged. Standalone on purpose: launched as a subprocess by server.py.

    python hf_server.py --model Qwen/Qwen2.5-1.5B-Instruct --port 8000
"""
from __future__ import annotations

import argparse
import asyncio
import json

import torch
from aiohttp import web
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--served-model-name", default="bench")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--dtype", default="half", choices=["half", "float32"])
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    dtype = torch.float16 if a.dtype == "half" else torch.float32
    print(f"[hf_server] loading {a.model} ({a.dtype}) on {a.device}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(a.device).eval()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
    eos_list = sorted(eos_ids)
    if a.device == "cuda":
        print(f"[hf_server] weights: {torch.cuda.memory_allocated() / 2**20:.0f} MiB", flush=True)

    lock = asyncio.Lock()   # <- the whole point: one generation at a time
    st = {"running": 0, "waiting": 0, "prompt": 0, "gen": 0}

    @torch.inference_mode()
    def step(ids, past, block_eos: bool):
        """One forward pass + greedy pick. Returns (next-token tensor, its id, new cache)."""
        out = model(input_ids=ids, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        if block_eos and eos_list:          # min_tokens: no EOS before the target length
            logits[:, eos_list] = float("-inf")
        nxt = logits.argmax(-1, keepdim=True)
        return nxt, int(nxt.item()), out.past_key_values

    def sse(obj) -> bytes:
        return f"data: {json.dumps(obj)}\n\n".encode()

    async def completions(req: web.Request):
        body = await req.json()
        prompt = body["prompt"]
        if isinstance(prompt, str):
            prompt = tok.encode(prompt, add_special_tokens=False)
        max_t = int(body.get("max_tokens") or 16)
        min_t = int(body.get("min_tokens") or 0)
        if len(prompt) + max_t > a.max_model_len:
            return web.json_response({"error": "prompt + max_tokens exceeds max_model_len"}, status=400)

        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(req)
        st["waiting"] += 1
        async with lock:
            st["waiting"] -= 1
            st["running"] += 1
            n = 0
            try:
                ids = torch.tensor([prompt], device=a.device)
                past, nxt = None, None
                while n < max_t:
                    # run the GPU work in a thread so the event loop keeps
                    # accepting (and queueing) other users' requests
                    nxt, t, past = await asyncio.to_thread(
                        step, ids if past is None else nxt, past, n < min_t)
                    n += 1
                    st["gen"] += 1
                    await resp.write(sse({"choices": [{"index": 0, "text": tok.decode([t]),
                                                       "finish_reason": None}]}))
                    if t in eos_ids and n >= min_t:
                        break
            finally:
                st["running"] -= 1
                st["prompt"] += len(prompt)
                del past
        await resp.write(sse({"choices": [], "usage": {
            "prompt_tokens": len(prompt), "completion_tokens": n,
            "total_tokens": len(prompt) + n}}))
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def health(_):
        return web.Response(text="ok")

    async def metrics(_):
        return web.Response(text=(
            f"infbench:num_requests_running {st['running']}\n"
            f"infbench:num_requests_waiting {st['waiting']}\n"
            f"infbench:prompt_tokens_total {st['prompt']}\n"
            f"infbench:generation_tokens_total {st['gen']}\n"))

    app = web.Application(client_max_size=64 * 2**20)
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", metrics)
    app.router.add_post("/v1/completions", completions)
    print("[hf_server] ready", flush=True)
    web.run_app(app, host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
