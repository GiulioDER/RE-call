# Retrieval-Augmented Self-Recall — an engineering writeup

> A RAG system engineered to be **honest about what it doesn't know**, built for a use case most
> RAG demos ignore: a long-running agent retrieving over *its own accumulated memory*.

This is the design story behind [**RE-call**](../README.md): the problem it targets, the architecture,
and — most importantly — the evaluation, including the results that *didn't* go the way a demo would
want them to. If you are reviewing this as a work sample, the [Evaluation](#3-evaluation) and
[What this demonstrates](#4-what-this-demonstrates) sections are the ones to read.

---

## 1. The problem: retrieval for an agent's own memory

Most RAG systems answer a human's question over a document corpus. **RE-call** targets a different
setting that shows up in every long-running autonomous agent:

**An agent accumulates memory — decisions it made, hypotheses it falsified, incidents it resolved —
and needs to retrieve from that memory before it acts.** Two failure modes dominate here, and both
are failures of *honesty*, not of ranking:

1. **Re-litigation.** The agent re-proposes an idea it already tried and rejected, because nothing
   surfaced the prior decision. Wasted work, and worse, silent drift away from settled conclusions.
2. **Confident retrieval on a gap.** The agent asks something the memory genuinely doesn't cover.
   A naive top-k retriever *always* returns k chunks — so it hands back the nearest noise with no
   signal that the corpus is empty on this topic. The agent then acts on that noise.

Ranking quality (precision@k, MRR) is necessary but not sufficient. The distinguishing requirement
is **calibrated abstention**: the system has to know, and say, when it doesn't know.

---

## 2. Design

### 2.1 Retrieval backbone

- **PostgreSQL + pgvector** as the single store — dense vectors and full-text search in one
  transactional database, no separate vector service to operate. This is the stack many teams
  already run in production, which is deliberate: it's the boring, defensible choice.
- **Hybrid retrieval**: dense cosine (`<=>`) *and* sparse lexical (Postgres FTS,
  `websearch_to_tsquery` + `ts_rank`), fused with **Reciprocal Rank Fusion** (RRF, k=60). Dense
  catches paraphrase; sparse catches exact identifiers, error codes, and rare tokens that embeddings
  smear.
- **Optional cross-encoder rerank** (ms-marco-MiniLM) over the fused candidates, for the cases where
  the first-stage ranking is weak (see finding 1).
- **Pluggable embedders** behind one `Embedder` protocol: a dependency-free `HashingEmbedder` (so the
  whole test suite runs offline), local `FastEmbed` (bge-small, no API key), and cloud `Voyage`.

### 2.2 The three honesty guards

These are the point of the project. Each one converts a silent failure into an explicit signal.

| Guard | Silent failure it prevents | Mechanism |
|---|---|---|
| **`gap_warning`** | Confident retrieval on an uncovered topic | If the best dense cosine for a query is below a calibrated threshold, the result is flagged "probable corpus gap — treat as noise" instead of returning nearest-noise as if it were an answer. |
| **freshness / staleness** | Serving stale memory as current | Every result reports how old the newest indexed content is; a stale index warns instead of silently serving rot. |
| **anti-re-litigation** | Re-deciding a settled question | The intended call pattern: an agent runs `search()` *before* proposing an idea; a surfaced closed decision (that is **not** itself a `gap_warning`) tells it to back off. Demonstrated end-to-end in [`examples/self_recall_agent.py`](../examples/self_recall_agent.py). |

The `gap_warning` guard is the one with real engineering depth, because a threshold that abstains
correctly is not a constant you can hard-code — which the evaluation is what proved.

### 2.3 Exposed as an MCP server

`recall_mcp` exposes `recall_search` / `recall_index` / `recall_stats` over the Model Context
Protocol (stdio), so any MCP client (e.g. Claude Desktop) can use the memory directly. The
self-recall loop then lives at the agent's own tool-call layer.

---

## 3. Evaluation

A reproducible ablation harness (`recall/eval`, `make eval`) scores every
`embedder × fusion (dense / hybrid / +rerank)` configuration against a labeled query set — 14
answerable and 5 deliberately-unanswerable queries — on a synthetic corpus, using precision@k,
recall@k, MRR, nDCG@10, and a guard-specific **false-confident rate (FCR)**: the fraction of
unanswerable queries the guard *failed* to flag. Numbers below are reproduced by the harness and by
`recall.eval.calibrate.calibrate()`; the full tables are in [`FINDINGS.md`](../results/FINDINGS.md)
and [`RESULTS.md`](../results/RESULTS.md).

### Finding 1 — hybrid + rerank helps *where the embedder isn't already saturated*

On the weak, non-semantic hashing embedder, quality climbs monotonically as the sparse leg and then
the cross-encoder are added:

| fusion | MRR | nDCG@10 |
|---|---|---|
| dense only | 0.68 | 0.76 |
| + sparse (hybrid) | 0.79 | 0.84 |
| + cross-encoder rerank | 1.00 | 1.00 |

On the strong bge-small embedder, dense retrieval already scores a perfect nDCG@10 on this corpus,
so the fusion arms have nothing left to gain. The honest reading — which a real eval must be able to
*show*, not hide — is: **hybrid + rerank buys the most on weaker embedders or harder corpora; on an
easy corpus with a strong embedder it is redundant.**

### Finding 2 — the honest negative: a fixed gap threshold does NOT transfer across embedders

This is the load-bearing result. Measuring the top-cosine distribution for answerable vs.
unanswerable queries, per embedder:

| embedder | answerable cos | unanswerable cos | good threshold | FCR @0.50 | FCR @calibrated |
|---|---|---|---|---|---|
| hashing-64 | 0.30 – 0.68 | 0.35 – 0.45 | — (overlap) | 0.00\* | — |
| bge-small | 0.70 – 0.90 | 0.50 – 0.64 | ~0.70 | **0.80** | **0.00** |
| voyage-3 | 0.53 – 0.70 | 0.09 – 0.32 | ~0.50 | **0.00** | **0.00** |

Three embedders, three completely different cosine regimes. The default 0.50 threshold sits in
Voyage's clean gap (works by luck), sits *below the entire* bge distribution (so the guard almost
never fires — FCR **0.80**), and lands inside hashing's overlap (unseparable at any threshold).
Recalibrating bge-small to ~0.70 makes its guard perfect (FCR 0.00).

**Takeaway:** calibrate the abstention threshold per embedding model against a small labeled set;
do not ship a hard-coded constant, and do not assume a strong embedder's cosines are centered where
a weak one's are. Gap-detection quality is also *bounded by the embedder* — a non-semantic model's
answerable/unanswerable distributions overlap, so no threshold separates them.

### Finding 3 — domain fine-tuning: an honest null result

`finetune/train.py` fine-tunes `all-MiniLM-L6-v2` (OnlineContrastiveLoss on query/gold-chunk
positives and query/wrong-chunk negatives, recipe adapted from a production trainer), then measures
on a **held-out** set of differently-phrased queries:

| model | test MRR | test nDCG@10 |
|---|---|---|
| base | 1.00 | 1.00 |
| + fine-tuned | 1.00 | 1.00 |
| **Δ** | **+0.00** | **+0.00** |

**Zero lift — and that is the correct outcome here.** The 14-document corpus is highly separable;
a modern small embedder already retrieves the right chunk for every held-out query. There is no
headroom. Manufacturing a win would have meant evaluating on the *training* queries (memorization)
or crippling the base model on purpose. The pipeline (held-out split, pre/post measurement, proven
loss) is built and runs end-to-end; on a saturated corpus it correctly reports no gain. To show a
*real* lift you need a corpus the base model struggles on — many mutually-confusable documents or
genuine unseen jargon.

---

## 4. What this demonstrates

For a reviewer, the signal in this repo is less "it retrieves" and more *how the retrieval was
engineered and judged*:

- **RAG beyond the demo**: hybrid dense + sparse retrieval with RRF, cross-encoder reranking, and a
  pluggable embedder abstraction — on a production-shaped Postgres + pgvector stack, not a toy
  in-memory index.
- **Calibrated abstention as a first-class feature.** The `gap_warning` guard, and the calibration
  study behind it, treat "knowing when you don't know" as the actual deliverable — the thing that
  makes retrieval safe to wire into an autonomous agent's decision loop.
- **Evaluation rigor**: an ablation harness with ranking metrics *and* a guard-specific
  false-confident rate; per-embedder threshold calibration; a held-out fine-tuning split.
- **Intellectual honesty.** Two of the three headline findings are negatives — a threshold that
  doesn't transfer, and a fine-tuning run with zero lift — reported as findings rather than buried.
  That is deliberate: in the production system this was extracted from, the expensive mistakes came
  from *forced* positive results, and the discipline here is to measure honestly and say so.
- **Production hygiene**: integration tests run against a real pgvector container (no mock DB), the
  whole diff went through an automated multi-auditor review before commit, and there are no secrets
  or private data in the tree.

## 5. Reproduce

```bash
docker compose up -d --wait
pip install -e ".[fastembed,rerank,eval]"
make eval        # -> results/RESULTS.md + charts
```

The Voyage cloud rows appear when `VOYAGE_API_KEY` is set. Everything else runs key-free.
