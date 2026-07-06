# recall — Retrieval-Augmented Self-Recall

RAG over a long-running agent's own memory, engineered to be **honest about what
it doesn't know**: it detects corpus gaps (`gap_warning`), flags stale indexes,
and is meant to be queried *before* the agent re-litigates a settled decision.

Built on **PostgreSQL + pgvector** with hybrid dense + full-text retrieval and
Reciprocal Rank Fusion.

## Quickstart (≈2 minutes, no API key)

```bash
git clone <this-repo> recall && cd recall
docker compose up -d                 # Postgres + pgvector
python -m venv .venv && . .venv/bin/activate    # Windows: .\.venv\Scripts\activate
pip install -e ".[fastembed,dev]"
python -m recall.cli demo
```

You'll see the caching and prompt-injection queries return relevant hits, and a
deliberately-unanswerable query flagged `[GAP]` instead of confidently returning
noise.

## The three honesty guards

- **`gap_warning`** — when the best candidate similarity is below threshold
  (~0.50 cosine), the result says "probable corpus gap — treat as noise".
- **freshness / staleness** — every result reports how old the newest indexed
  content is; a stale index warns instead of silently serving rot.
- **anti-re-litigation** — the intended usage: an agent calls `search()` before
  re-proposing an idea, so closed decisions resurface instead of being redone.

## Usage

```bash
python -m recall.cli index ./path/to/markdown   # index your own docs
python -m recall.cli search "your question"
```

Set `RECALL_DSN` to point at another Postgres. Default embedder is local
FastEmbed (no key); `--embedder hashing` is a fully-offline fallback.

## Tests

```bash
docker compose up -d
pytest -v      # integration tests hit the real pgvector container (no mock DB)
```

## Status

M1 (engine + demo). Next: agentic self-recall MCP server, a reproducible eval
harness with ablations and an honest negative result, and the fine-tuning study.
