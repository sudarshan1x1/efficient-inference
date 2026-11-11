"""Workload builder.

Idea borrowed from systalyze/vllm-benchmarks (workload/aws-p50p90-v1-mintok):
  * prompts are realistic coding text, not random tokens
  * every request is held to an exact output length with min_tokens == max_tokens

Adapted for a T4 / controlled experiment:
  * input length is FIXED (same token count for every request) by slicing the
    corpus into windows of exactly `isl` token ids and sending the ids directly
    to /v1/completions, so tokenisation can't change the length.
  * every request gets a different window, so prefix caching can't sneak in
    unless an experiment asks for it (shared_prefix mode).
"""
from __future__ import annotations

import json
import random
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

SYSTALYZE_URL = (
    "https://raw.githubusercontent.com/systalyze/vllm-benchmarks/main/"
    "workload/aws-p50p90-v1-mintok/requests-qwen3-32b.jsonl"
)


@dataclass
class Request:
    prompt_ids: list[int]
    output_len: int


@dataclass
class Corpus:
    texts: list[str]
    output_lengths: list[int] = field(default_factory=list)  # from systalyze, may be empty


def load_corpus(cache_dir: str | Path, url: str = SYSTALYZE_URL) -> Corpus:
    """Download the systalyze workload once (cached), fall back to synthetic text offline."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / Path(url).name
    if not path.exists():
        try:
            print(f"[workload] downloading {url}")
            urllib.request.urlretrieve(url, path)
        except Exception as exc:  # noqa: BLE001
            print(f"[workload] download failed ({exc}); using synthetic corpus")
            return _synthetic_corpus()
    texts, olens = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            olens.append(int(row.get("output_length", 0)))
    print(f"[workload] corpus: {len(texts)} documents from {path.name}")
    return Corpus(texts, olens)


def _synthetic_corpus(n_docs: int = 200, seed: int = 0) -> Corpus:
    rnd = random.Random(seed)
    words = ("def class return import async await tensor cache kernel batch latency token "
             "memory scheduler request queue throughput block page warp thread gpu").split()
    texts = [" ".join(rnd.choice(words) for _ in range(3000)) for _ in range(n_docs)]
    return Corpus(texts, [])


class TokenStream:
    """One long token-id stream built from the corpus; hands out non-overlapping windows."""

    def __init__(self, tokenizer, corpus: Corpus, seed: int = 0):
        docs = list(corpus.texts)
        random.Random(seed).shuffle(docs)
        ids: list[int] = []
        for d in docs:
            ids.extend(tokenizer.encode(d, add_special_tokens=False))
        self.ids = ids
        self.cursor = 0
        print(f"[workload] token stream: {len(ids):,} tokens")

    def view(self, cursor: int) -> "TokenStream":
        """A cheap copy that shares the token ids but starts at `cursor`."""
        v = TokenStream.__new__(TokenStream)
        v.ids, v.cursor = self.ids, cursor % max(len(self.ids), 1)
        return v

    def take(self, n: int) -> list[int]:
        if self.cursor + n > len(self.ids):  # wrap around: reuse is rare and logged
            print("[workload] token stream wrapped; prompts may start repeating")
            self.cursor = 0
        out = self.ids[self.cursor:self.cursor + n]
        self.cursor += n
        return out


def build_fixed(stream: TokenStream, n: int, isl: int, osl: int) -> list[Request]:
    """n requests, each exactly `isl` input tokens and `osl` output tokens, all unique."""
    return [Request(stream.take(isl), osl) for _ in range(n)]


def build_shared_prefix(stream: TokenStream, n: int, prefix_len: int, suffix_len: int,
                        osl: int, prefix: list[int] | None = None) -> list[Request]:
    """n requests sharing one `prefix_len` prefix (think: a long system prompt / RAG
    context) + a unique `suffix_len` tail. Total input length is still fixed."""
    prefix = prefix if prefix is not None else stream.take(prefix_len)
    return [Request(prefix + stream.take(suffix_len), osl) for _ in range(n)]


def build_variable_osl(stream: TokenStream, corpus: Corpus, n: int, isl: int,
                       scale: float = 0.5, cap: int = 256, floor: int = 16,
                       seed: int = 0) -> list[Request]:
    """Fixed input length, but output lengths follow the systalyze distribution
    (p50 ~200 / p90 ~400, scaled down to fit a T4). This is what makes
    continuous batching beat static batching: requests finish at different times."""
    olens = corpus.output_lengths or [128] * 160
    rnd = random.Random(seed)
    reqs = []
    for _ in range(n):
        o = max(floor, min(cap, round(rnd.choice(olens) * scale)))
        reqs.append(Request(stream.take(isl), o))
    return reqs
