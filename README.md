<p align="center">
  <img src="docs/banner.png" alt="RE-call — Retrieval-Augmented Self-Recall" width="900">
</p>

<p align="center">
  RAG for an AI agent's own memory — that <i>knows when it doesn't know</i>.
</p>

<p align="center">
  <a href="https://github.com/GiulioDER/RE-call/actions/workflows/ci.yml"><img src="https://github.com/GiulioDER/RE-call/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/PostgreSQL-pgvector-336791" alt="PostgreSQL + pgvector">
</p>

<p align="center">
  <a href="docs/CASE_STUDY.md"><b>📄 Real-world usage →</b></a>
  &nbsp;·&nbsp;
  <a href="docs/USING_WITH_CLAUDE.md">Use with Claude</a>
  &nbsp;·&nbsp;
  <a href="docs/WRITEUP.md">Engineering writeup</a>
  &nbsp;·&nbsp;
  <a href="docs/RAG_TRAINING_STUDY.md">Fine-tuning study</a>
  &nbsp;·&nbsp;
  <a href="#-quickstart-2-minutes-no-api-key">Quickstart</a>
</p>

---

A long-running agent piles up memory — decisions, closed experiments, incident notes. Three failure
modes follow: it **re-litigates settled decisions**, it **hallucinates over gaps** where the
memory simply has no answer, and it **confidently builds on memories that are no longer true** —
the semantically-closest match wins even after the decision it records was reversed.

**RE-call** is a RAG engine for that memory, built to be *honest about what it doesn't know*: it
retrieves **before** the agent acts, returns **confidence + provenance + validity** with every hit
— not just similarity — and prefers an explicit *"I don't know"* over confident noise.

## ✨ What it does

- 🕳️ **Gap-aware** — when the best match is weak, it returns a `gap_warning` (*"probable corpus gap — treat as noise"*) instead of hallucinating an answer.
- 🧾 **Trust verdicts** — every hit carries a verdict (`ok · superseded · expired · not_yet_valid · low_confidence`), a **calibrated confidence**, and provenance (`indexed_at`, source, chunk). A memory that is superseded or outside its validity window **loses to its successor — or to "I don't know"** — even when it has the top cosine.
- 🎯 **Calibrated abstention** — the confidence threshold is calibrated *per embedder* against a labelled query set (`recall calibrate`); when no valid hit clears it, the result abstains with a reason instead of answering.
- ⏱️ **Freshness-aware** — every result reports how stale the index is, so a rotting memory warns instead of silently serving old facts.
- 🔁 **Anti-re-litigation** — meant to be queried *before* re-proposing an idea, so closed decisions and falsified hypotheses resurface first.
- ⚖️ **Entailment stage (opt-in)** — a QNLI judge catches the **near-miss**: a high-similarity memory that doesn't actually answer the query (which clears any cosine threshold *by construction*). A decision, not another score — nothing to recalibrate per embedder. OFF by default; measured cost in [Finding 5](results/FINDINGS.md).
- 🧹 **`recall lint`** — write-time checks on the supersession graph (dangling/cyclic `supersedes:`, versioned siblings with no edge, closures declared only in prose). No DB needed; exit 1 on errors, CI-ready.
- 🧱 **Production-shaped** — PostgreSQL + pgvector, hybrid dense + full-text retrieval fused with RRF, cross-encoder reranking, and an MCP server. Integration-tested on a real database.

## 🧭 How it works

```mermaid
flowchart LR
    Q([query]) --> E[embed]
    E --> D[dense · pgvector cosine]
    Q --> S[sparse · Postgres full-text]
    D --> F[Reciprocal Rank Fusion]
    S --> F
    F --> R[cross-encoder rerank]
    R --> G{trust layer}
    G --> O([verdict + confidence + provenance per hit, or ABSTAIN])
```

Dense semantic search and sparse keyword search each retrieve candidates; **Reciprocal Rank Fusion**
merges them, a cross-encoder reranks, and the **trust layer** judges every hit — supersession,
validity window, calibrated confidence — before it ever reaches the agent. Validity is plain
frontmatter in the memory itself (`supersedes: old_doc.md`, `valid_until: 2026-06-30`).

## ⚡ See it work

```text
$ python -m recall.cli demo
indexed 5 chunks from 5 files

[ok] query='how many requests per second can a client make?'
  ok             conf=1.00 cos=0.784  rate_limits_v2.md  '# API rate limits (revised)  Rate limiting was tight'
  ok             conf=0.96 cos=0.655  decisions.md       '# Decisions  ## 2026-05-02 — Caching layer We decide'
  ...
  superseded     conf=1.00 cos=0.806  rate_limits_v1.md -> use rate_limits_v2.md  '# API rate limits  Each client key is limited to 100'

[ABSTAIN GAP] query='how do we handle penguins on mars?'
  reason: no hit above the calibrated confidence threshold (probable corpus gap)
  low_confidence conf=0.35 cos=0.468  decisions.md  '# Decisions  ## 2026-05-02 — Caching layer We decide'
```

Look at the first query: the *stale* rate-limit memory has the **highest cosine in the whole result
(0.806)** — plain vector search would hand it back as the answer, and the agent would build on a
limit that no longer exists. The trust layer flags it `superseded`, points at its successor, and
puts the *current* memory on top. And when the memory genuinely has no answer, the result is an
explicit **`ABSTAIN`** with a reason — not the least-irrelevant chunk dressed up as one. That
ordering decision is the whole thesis.

## 📊 Results that matter

A reproducible ablation harness scores every `embedder × fusion` config on a labelled query set —
precision@k, recall@k, MRR, nDCG, and a guard-specific **false-confident rate**.

<p align="center">
  <img src="results/trust_effect.png" width="48%" alt="Superseded-trust rate: plain search vs trust layer">
  &nbsp;
  <img src="results/guard_effect.png" width="48%" alt="Guard effect: false-confident rate on unanswerable queries">
</p>

Six **honest** findings — including what *didn't* work:

- 🧾 **Similarity cannot see supersession — the trust layer can.** On validity-sensitive queries
  (worded deliberately closer to the *stale* memory), plain search returns the superseded/expired
  memory as the answer **83–100% of the time**; with the trust layer that rate is **0.00 on every
  embedder**, ordinary retrieval quality is untouched (identical MRR), and abstention fires on the
  expired-only cases on the calibrated semantic embedder (a weak embedder cannot support
  abstention at any threshold). → *Full table + limits in [FINDINGS §4](results/FINDINGS.md).*
- 🎯 **The gap threshold doesn't transfer across embedders.** The default `0.50` sits below
  FastEmbed's entire cosine distribution — a **1.00** false-confident rate (the guard never fires);
  per-embedder calibration to `~0.70` makes it perfect (0.00). → *Calibrate against a small
  labelled set; don't hard-code.*
- 🔁 **Reranking rescues a weak embedder.** Hybrid + cross-encoder lifts MRR **0.63 → 1.00** on the
  offline embedder — but a strong embedder already saturates this corpus, so the gain is real yet
  situational.
- 🧪 **Fine-tuning the embedder pays off only for a vocabulary gap.** A controlled study: on a rich
  corpus the base already saturates (Δ **+0.00**); on an opaque-jargon corpus it can't decode,
  fine-tuning lifts held-out MRR **0.31 → 0.55 (+79%)** and generalizes to unseen paraphrases.
  → *Measure the base–corpus gap before fine-tuning.* **[Read the study →](docs/RAG_TRAINING_STUDY.md)**
- ⚖️ **Near-misses need a judge — and the judge needs the threshold.** On a held-out set of
  high-similarity-but-wrong queries the calibrated threshold is blind (FCR **0.40–1.00**, by
  construction); an optional QNLI entailment stage cuts it (**1.00→0.60, 0.80→0.50**) with the
  *identical judge on every embedder, zero recalibration* — but judge-alone *degrades* far-gap
  detection (0.00→0.40): they guard **different failure classes, so stack them**. Costs measured:
  ~100× latency, one negation-phrased answer wrongly rejected.
  **[Read the study →](docs/ENTAILMENT_SUPERSESSION_STUDY.md)**
- ⏰ **Timestamps cannot replace declared supersession — even steelmanned.** "Trust the newest
  relevant hit" (stale docs re-synced later, as real corpora constantly do) still trusts the stale
  memory **83–100%** of the time — and on bge-small it's *worse than plain ranking*. Supersession
  is a relation between two documents; a per-document timestamp can't see it. The declared
  `supersedes:` edge stays at **0.00** in the same runs.

> Full methodology + per-embedder tables → **[results/FINDINGS.md](results/FINDINGS.md)**.

✅ **141 tests — the DB-touching ones against a real pgvector container** (no mock DB), verified in CI, with a
dependency audit.

## 🏭 Where this comes from

RE-call isn't a toy — it's extracted from the memory system I run for a **production trading-research
agent** whose own memory outgrew its context window: **≈660 typed markdown memos (~5 MB), re-indexed
daily.** Every guard here is a scar from a real failure — re-litigating an already-falsified
experiment, trusting weak hits on a question the memory couldn't answer, serving a stale fact.

**→ [Read the redacted case study](docs/CASE_STUDY.md)** — the real structure, the guards in action,
and exactly what's public vs private.

## 🚀 Quickstart (≈2 minutes, no API key)

```bash
git clone https://github.com/GiulioDER/RE-call && cd RE-call
docker compose up -d --wait          # Postgres + pgvector (waits until healthy)
python -m venv .venv && . .venv/bin/activate    # Windows: .\.venv\Scripts\activate
pip install -e ".[fastembed,dev]"
python -m recall.cli demo
```

Default embedder is local **FastEmbed** (no key); `--embedder hashing` is a fully-offline fallback.

## 🔧 Use it

```bash
python -m recall.cli index ./path/to/markdown   # index your own docs
python -m recall.cli search "your question"     # -> verdicts + confidence + gap/freshness flags
python -m recall.cli calibrate recall/eval/queries.json   # per-embedder abstention threshold -> calibration.json
python -m recall.cli lint ./path/to/markdown    # supersession-graph completeness (no DB needed)
```

Point `RECALL_DSN` at any Postgres to use your own database. Declare validity in the memory
itself — plain frontmatter, no schema changes (validity is *authored, not verified*: index only
content you trust, because a `supersedes:` claim is honored as written):

```markdown
---
supersedes: rate_limits_v1.md
valid_until: 2026-12-31
---
# API rate limits (revised)
...
```

**Code, not just prose.** The engine is content-agnostic — point it at source and it chunks on
`def` / `class` boundaries, so natural-language questions land the exact function:

```text
$ python -m recall.cli index ./src --glob "**/*.py"   # your codebase
$ python -m recall.cli code                            # demo: search RE-call's OWN source
indexed 67 code chunks from 19 files

[ok] query='where is reciprocal rank fusion implemented?'
  ok             conf=1.00 cos=0.805  retriever.py  'def _rrf(rankings: list[list[str]], k: int = 60) -> '

[ok] query='how are embeddings stored in postgres?'
  ok             conf=1.00 cos=0.788  store.py  'class PgVectorStore:     """The single, production-g'

[ok] query='how does cross-encoder reranking reorder hits?'
  ok             conf=1.00 cos=0.886  rerank.py  'class CrossEncoderReranker:     """Reorder hits by c'
```

## 🔌 Use it with Claude (MCP)

Expose memory to **Claude Code** or **Claude Desktop** as three tools — `recall_search`,
`recall_index`, `recall_stats` — so the agent queries its memory *before* it acts:

```bash
pip install -e ".[fastembed,mcp]"
python -m recall_mcp.server        # stdio server
```

The self-recall pattern: Claude calls `recall_search` **before** proposing an idea; if a closed
decision surfaces (and it isn't a `gap_warning`), it backs off instead of re-litigating. Every hit
now carries `verdict` / `confidence` / `superseded_by` / `indexed_at`, and when `abstained` is
true the advice says so explicitly: *say you don't know — do not answer from these hits.*

**→ [Full guide: config for Claude Code + Desktop, the three tools, and a real redacted loop](docs/USING_WITH_CLAUDE.md)**
&nbsp;·&nbsp; example agent: [`examples/self_recall_agent.py`](examples/self_recall_agent.py).

## 🧪 Reproduce the evaluation

```bash
pip install -e ".[fastembed,rerank,eval]"
make eval        # -> results/RESULTS.md + the charts above
```

The Voyage cloud row appears when `VOYAGE_API_KEY` is set (shell env, or a gitignored `.env`).

## 🧱 Tests

```bash
docker compose up -d --wait
pytest -v      # integration tests hit the real pgvector container — no mock DB
```

## License

[MIT](LICENSE).
