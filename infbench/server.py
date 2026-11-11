"""Start / stop a vLLM OpenAI-compatible server as a subprocess (Colab has no Docker).

One server per configuration, mirroring systalyze's "one launch.sh per config":
the config dict is translated into `vllm serve` flags, the full command is saved
next to the results so every number is reproducible.
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def config_to_args(cfg: dict) -> list[str]:
    """Translate one experiment config into server CLI flags (vLLM or the naive HF server)."""
    if cfg.get("engine", "vllm") == "hf":
        return ["--max-model-len", str(cfg.get("max_model_len", 2048)),
                "--dtype", "half" if cfg.get("dtype", "half") in ("half", "float16") else "float32"]
    a: list[str] = []
    a += ["--dtype", str(cfg.get("dtype", "half"))]  # T4 (sm75) has no bf16
    a += ["--max-model-len", str(cfg.get("max_model_len", 2048))]
    a += ["--gpu-memory-utilization", str(cfg.get("gpu_memory_utilization", 0.85))]
    a += ["--max-num-seqs", str(cfg.get("max_num_seqs", 256))]
    if cfg.get("quantization"):
        a += ["--quantization", str(cfg["quantization"])]
    if cfg.get("kv_cache_dtype"):
        a += ["--kv-cache-dtype", str(cfg["kv_cache_dtype"])]
    # Prefix caching is ON by default in vLLM V1. We turn it OFF for the main
    # sweep (same as systalyze) so it can't hide inside "baseline" numbers.
    a += ["--enable-prefix-caching"] if cfg.get("prefix_caching") else ["--no-enable-prefix-caching"]
    if cfg.get("max_num_batched_tokens"):
        a += ["--max-num-batched-tokens", str(cfg["max_num_batched_tokens"])]
    if cfg.get("enforce_eager"):
        a += ["--enforce-eager"]
    a += [str(x) for x in cfg.get("extra_args", [])]
    return a


class VLLMServer:
    def __init__(self, model: str, args: list[str], log_path: str | Path,
                 port: int = 8000, served_name: str = "bench", engine: str = "vllm"):
        self.model, self.args, self.port, self.served_name = model, args, port, served_name
        self.engine = engine
        self.log_path = Path(log_path)
        self.base_url = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None

    @property
    def command(self) -> list[str]:
        if self.engine == "hf":
            script = str(Path(__file__).with_name("hf_server.py"))
            return [sys.executable, script, "--model", self.model, "--host", "127.0.0.1",
                    "--port", str(self.port), "--served-model-name", self.served_name] + self.args
        exe = shutil.which("vllm")
        head = [exe, "serve", self.model] if exe else \
            [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", self.model]
        return head + ["--host", "127.0.0.1", "--port", str(self.port),
                       "--served-model-name", self.served_name] + self.args

    def start(self, timeout_s: int = 900) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        print("[server] " + " ".join(self.command))
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(self.command, stdout=self._log, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=os.environ.copy())
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with code {self.proc.returncode}.\n"
                                   f"--- last log lines ---\n{self.tail(40)}")
            if self._healthy():
                print(f"[server] ready in {time.time() - t0:.0f}s")
                return
            time.sleep(3)
        self.stop()
        raise TimeoutError(f"vLLM not healthy after {timeout_s}s\n{self.tail(40)}")

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=2) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False

    def stop(self, gpu_release_timeout_s: int = 90) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
        if getattr(self, "_log", None):
            self._log.close()
        _wait_gpu_free(gpu_release_timeout_s)
        print("[server] stopped")

    def tail(self, n: int = 30) -> str:
        try:
            return "".join(self.log_path.read_text(errors="ignore").splitlines(True)[-n:])
        except FileNotFoundError:
            return ""

    def kv_capacity(self) -> dict:
        """vLLM prints its KV-cache budget at startup; that number is the hard ceiling
        on how many tokens can be in flight before requests queue or get preempted."""
        text = self.log_path.read_text(errors="ignore") if self.log_path.exists() else ""
        out = {}
        m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text)
        if m:
            out["kv_capacity_tokens"] = int(m.group(1).replace(",", ""))
        m = re.search(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x", text)
        if m:
            out["max_concurrency_at_max_len"] = float(m.group(2))
        m = re.search(r"# GPU blocks:\s*(\d+)", text)  # older engines
        if m:
            out["gpu_blocks"] = int(m.group(1))
        # Where the VRAM went (best effort: log wording differs between versions)
        for key, pat in (("mem_weights_gib", r"Model loading took\s*([\d.]+)\s*GiB"),
                         ("mem_kv_cache_gib", r"Available KV cache memory:\s*([\d.]+)\s*GiB"),
                         ("mem_cuda_graphs_gib", r"Graph capturing finished.*?([\d.]+)\s*GiB"),
                         ("mem_weights_gib", r"weights:\s*([\d.]+)\s*MiB")):   # hf_server
            m = re.search(pat, text)
            if m and key not in out:
                v = float(m.group(1))
                out[key] = v / 1024 if "MiB" in pat else v
        m = re.search(r"max_num_batched_tokens=(\d+)", text)
        if m:
            out["engine_max_num_batched_tokens"] = int(m.group(1))
        m = re.search(r"Using (\w+) attention backend", text)
        if m:
            out["attention_backend"] = m.group(1)
        return out

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


def _wait_gpu_free(timeout_s: int, threshold_mib: int = 1500) -> None:
    """Wait until the previous server's CUDA context is gone so the next config
    starts from a clean GPU (otherwise its KV budget is silently smaller)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            used = pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**20
            if used < threshold_mib:
                return
            time.sleep(2)
        print(f"[server] warning: GPU still has {used:.0f} MiB in use")
    except Exception:  # noqa: BLE001
        time.sleep(5)
