"""Stage-1 smoke test for the OpenRouter-served `text-embedding-3-small` arm.

Offline part (no key): asserts the Mem0 embedder config is wired to OpenRouter with the right model
and *without* a `dimensions` param, and that the Qdrant width is 1536. Runs anywhere.

Live part (needs OPENROUTER_API_KEY in the environment): actually embeds through OpenRouter with
BOTH adapters — RE-call's OpenAICompatEmbedder and Mem0's OpenAIEmbedding — and asserts 1536-dim
vectors come back. Auto-skips if the key is absent, so it is safe to run in a keyless shell.

Run from the RE-call repo root:  python -m benchmarks.smoke_router_embedder
"""
from __future__ import annotations

import os
import sys

MODEL = "openai/text-embedding-3-small"
BASE = "https://openrouter.ai/api/v1"


def offline_checks() -> None:
    from benchmarks.systems import mem0_config

    cfg = mem0_config(
        "sk-or-placeholder", "openai/gpt-4o-mini",
        embedder="openai", openai_key="sk-or-placeholder",
        openai_model=MODEL, openai_base_url=BASE, run_id="smoke",
    )
    emb = cfg["embedder"]
    assert emb["provider"] == "openai", emb
    assert emb["config"]["model"] == MODEL, emb
    assert emb["config"]["openai_base_url"] == BASE, emb
    assert "embedding_dims" not in emb["config"], "must NOT set embedding_dims (skips dimensions param)"
    assert cfg["vector_store"]["config"]["embedding_model_dims"] == 1536, cfg["vector_store"]
    print("[offline] Mem0 embedder -> OpenRouter, model=%s, no dimensions param, qdrant dim=1536  OK" % MODEL)


def live_checks(key: str) -> None:
    from benchmarks.systems import resolve_embedder

    re_emb = resolve_embedder(f"router:{MODEL}")
    vecs = re_emb.embed(["hello world", "a second sentence"])
    assert len(vecs) == 2 and len(vecs[0]) == 1536 and len(vecs[1]) == 1536, [len(v) for v in vecs]
    print(f"[live] RE-call OpenAICompatEmbedder: name={re_emb.name} dim={re_emb.dim} n_vecs={len(vecs)}  OK")

    from mem0.configs.embeddings.base import BaseEmbedderConfig
    from mem0.embeddings.openai import OpenAIEmbedding

    m = OpenAIEmbedding(BaseEmbedderConfig(model=MODEL, api_key=key, openai_base_url=BASE))
    v = m.embed("hello from mem0")
    assert len(v) == 1536, len(v)
    print(f"[live] Mem0 OpenAIEmbedding via OpenRouter: dim={len(v)}  OK")


def main() -> int:
    offline_checks()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("[live] SKIPPED: OPENROUTER_API_KEY not set in this shell. Re-run where it is exported.")
        return 0
    live_checks(key)
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
