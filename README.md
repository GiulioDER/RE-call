# recall — Retrieval-Augmented Self-Recall

[![CI](https://github.com/GiulioDER/recall/actions/workflows/ci.yml/badge.svg)](https://github.com/GiulioDER/recall/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

RAG over a long-running agent's own memory, engineered to be **honest about what
it doesn't know**: it detects corpus gaps (`gap_warning`), flags stale indexes,
and is meant to be queried *before* the agent re-litigates a settled decision.

Built on **PostgreSQL + pgvector** with hybrid dense + full-text retrieval and
Reciprocal Rank Fusion.

**→ [Engineering writeup: the design, and the honest evaluation](docs/WRITEUP.md)** — the problem,
the three honesty guards, and what the ablations (including two negative results) actually showed.

## Quickstart (≈2 minutes, no API key)

```bash
git clone <this-repo> recall && cd recall
docker compose up -d --wait          # Postgres + pgvector (waits until healthy)
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

## MCP self-recall server

Expose memory to an MCP client as tools — `recall_search`, `recall_index`, `recall_stats`:

    pip install -e ".[fastembed,mcp]"
    python -m recall_mcp.server        # stdio server

Example client config (e.g. Claude Desktop):

    {
      "mcpServers": {
        "recall": {
          "command": "python",
          "args": ["-m", "recall_mcp.server"],
          "env": { "RECALL_DSN": "postgresql://recall:recall@localhost:5432/recall" }
        }
      }
    }

Additional server env: `RECALL_EMBEDDER=hashing` selects the fully-offline embedder (default
`fastembed`); `RECALL_INDEX_ROOT` bounds where `recall_index` may read (default: the server's
working directory).

The self-recall pattern: an agent calls `recall_search` before proposing an idea; if a closed
decision or falsified hypothesis surfaces (and it isn't a `gap_warning`), it backs off instead
of re-litigating. See `examples/self_recall_agent.py`.

## Evaluation

A reproducible ablation harness lives in `recall/eval`. With Docker up and the eval extras installed:

    pip install -e ".[fastembed,rerank,eval]"
    make eval        # -> results/RESULTS.md + charts

It scores every `embedder × fusion (dense / hybrid / +rerank)` config against a labeled query set
on a synthetic corpus, using precision@k, recall@k, MRR, nDCG, and a guard-specific
**false-confident rate**. Two honest findings (full writeup in
[results/FINDINGS.md](results/FINDINGS.md)):

- **Hybrid + cross-encoder rerank** lifts MRR from 0.68 to 1.00 on a weak embedder; a strong
  embedder already saturates this corpus (so the gain is real but situational).
- **The gap threshold does not transfer across embedders** — the default 0.50 gives a 0.80
  false-confident rate on FastEmbed (whose cosines cluster high), but per-embedder calibration to
  ~0.70 makes the guard perfect. Calibrate against a small labeled set; don't hard-code.

The Voyage cloud row appears when `VOYAGE_API_KEY` is set — in your shell, or in a gitignored
`recall/.env` (a tiny built-in loader picks it up).

## Tests

```bash
docker compose up -d --wait
pytest -v      # integration tests hit the real pgvector container (no mock DB)
```

## Status

M1 (engine + demo), M2 (self-recall MCP server), M3 (eval harness, ablations, cloud + local embedder
comparison, gap-threshold calibration, and a domain fine-tuning study), and M4 (writeup, LICENSE,
LF normalization, dependency audit) complete.

## License

[MIT](LICENSE).
