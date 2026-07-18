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
and needs to retrieve from that memory before it acts.** Three failure modes dominate here, and
all are failures of *honesty*, not of ranking:

1. **Re-litigation.** The agent re-proposes an idea it already tried and rejected, because nothing
   surfaced the prior decision. Wasted work, and worse, silent drift away from settled conclusions.
2. **Confidently building on a memory that is no longer true.** The semantically-closest match
   keeps winning retrieval even after the decision it records was reversed — superseded or
   expired memory reads exactly like current memory to a vector index.
3. **Confident retrieval on a gap.** The agent asks something the memory genuinely doesn't cover.
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

### 2.2 The honesty guards

These are the point of the project. Each one converts a silent failure into an explicit signal.

| Guard | Silent failure it prevents | Mechanism |
|---|---|---|
| **`gap_warning`** | Confident retrieval on an uncovered topic | If the best dense cosine for a query is below a calibrated threshold, the result is flagged "probable corpus gap — treat as noise" instead of returning nearest-noise as if it were an answer. |
| **trust verdicts + abstention** (v0.2) | Confidently building on a memory that is no longer true | Every hit returns confidence + provenance + validity: a verdict (`ok / superseded / expired / not_yet_valid / low_confidence / invalid_metadata / not_entailed`), a calibrated confidence, and `indexed_at`. A superseded or out-of-window memory loses to its retrieved successor — or, when no valid hit clears the calibrated threshold, to an explicit abstention with a reason. |
| **entailment stage** (v0.3, opt-in) | Confident retrieval on a *near-miss* — a high-similarity memory that does not answer the query, which clears any cosine threshold by construction | An optional QNLI cross-encoder judges the verdict-ok hits ("does this sentence answer this question?") and demotes non-answering ones to `not_entailed`. A decision at the judge's own boundary, not another score to threshold — so nothing to recalibrate per embedder. OFF by default; measured cost and limits in [Finding 5](#finding-5--entailment-abstention-a-judged-decision-stacked-on-the-threshold-not-a-replacement). |
| **`recall lint`** (v0.3) | An author closing a decision without declaring the supersession edge — an orphan memo that looks valid forever | Write-time completeness checks on the supersession graph (dangling/self/cyclic edges, malformed dates, version-sibling and closed-in-prose smells). No DB needed; exit 1 on errors, CI-ready. |
| **freshness / staleness** | Serving stale memory as current | Every result reports how old the newest indexed content is; a stale index warns instead of silently serving rot. |
| **anti-re-litigation** | Re-deciding a settled question | The intended call pattern: an agent runs `search()` *before* proposing an idea; a surfaced closed decision (that is **not** itself a `gap_warning`) tells it to back off. Demonstrated end-to-end in [`examples/self_recall_agent.py`](../examples/self_recall_agent.py). |

The trust layer exists because relevance and trustworthiness are different questions. Traditional
RAG assumes the answer exists and optimizes retrieval quality; an agent's memory has to answer a
prior question — *should I trust any retrieved memory at all?* A false-positive retrieval is worse
than a miss, because the agent confidently builds on a reversed decision. So retrieval here returns
**confidence + provenance + validity, not just relevance**, and a memory that is semantically
similar but superseded or outside its validity window loses to "I don't know". Validity is declared
in the memory itself (frontmatter: `supersedes:`, `valid_from:` / `valid_until:`) and enforced as a
pure post-processing layer over the retriever ([`recall/trust.py`](../recall/trust.py)); the
threshold it abstains at is not a constant you can hard-code — which the evaluation proved.

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
| dense only | 0.63 | 0.72 |
| + sparse (hybrid) | 0.74 | 0.80 |
| + cross-encoder rerank | 1.00 | 1.00 |

On the strong bge-small embedder, dense retrieval already scores nDCG@10 0.97 and the hybrid arm
saturates the corpus at 1.00, so the reranker has nothing left to gain. The honest reading — which a real eval must be able to
*show*, not hide — is: **hybrid + rerank buys the most on weaker embedders or harder corpora; on an
easy corpus with a strong embedder it is redundant.**

### Finding 2 — the honest negative: a fixed gap threshold does NOT transfer across embedders

This is the load-bearing result. Measuring the top-cosine distribution for answerable vs.
unanswerable queries, per embedder:

| embedder | answerable cos | unanswerable cos | good threshold | FCR @0.50 | FCR @calibrated |
|---|---|---|---|---|---|
| hashing-64 | 0.30 – 0.68 | 0.35 – 0.53 | — (overlap) | 0.20\* | — |
| bge-small | 0.70 – 0.90 | 0.51 – 0.64 | ~0.70 | **1.00** | **0.00** |
| voyage-3\† | 0.53 – 0.70 | 0.09 – 0.32 | ~0.50 | **0.00** | **0.00** |

Three embedders, three completely different cosine regimes. The default 0.50 threshold sits in
Voyage's clean gap (works by luck), sits *below the entire* bge distribution (so the guard almost
never fires — FCR **1.00**), and lands inside hashing's overlap (unseparable at any threshold).
Recalibrating bge-small to ~0.70 makes its guard perfect (FCR 0.00). (\* hashing's 0.20 also
wrongly flags answerable queries below 0.50 — no threshold works. \† voyage-3 measured on
the v0.1 corpus.)

**Takeaway:** calibrate the abstention threshold per embedding model against a small labeled set;
do not ship a hard-coded constant, and do not assume a strong embedder's cosines are centered where
a weak one's are. Gap-detection quality is also *bounded by the embedder* — a non-semantic model's
answerable/unanswerable distributions overlap, so no threshold separates them.

### Finding 3 — domain fine-tuning: an honest null result (see also Finding 4)

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

### Finding 4 — validity beats similarity (v0.2 trust layer)

On validity-sensitive queries worded deliberately closer to the *stale* memory, plain search
returns the superseded/expired memory as the answer 83-100% of the time; the trust layer never
presents it as trustworthy (superseded-trust rate 0.00 on both embedders) at zero cost to
ordinary retrieval (identical answerable MRR). Abstention on expired-only queries works on the
calibrated semantic embedder (2/2) and not at all on the weak one (0/2) - the same
embedder-bound limit as Finding 2. The eval also scores the *steelman* timestamp alternative —
"among the confidently-relevant hits, trust the newest", with stale docs re-synced later as any
living corpus does: it trusts the stale memory 83–100% of the time and on bge-small is worse
than plain ranking. Supersession is a relation between two documents; a per-document timestamp
cannot see it. Full table + limits: [FINDINGS.md §4](../results/FINDINGS.md).

### Finding 5 — entailment abstention: a judged decision stacked on the threshold, not a replacement

The calibrated threshold cannot see the **near-miss** class — a high-similarity memory that
does not answer the query (baseline false-confident rate 0.40–1.00 on a held-out 10-query
challenge set). An optional QNLI judge (v0.3, `recall[entail]`, OFF by default) demotes
non-answering hits and cuts near-miss FCR to 0.40–0.60 with the *identical judge on every
embedder — no recalibration*, the transfer property Finding 2 proved a score threshold lacks.
The ablation is the honest half: judge-alone *degrades* far-gap detection (0.00→0.40) — the
threshold and the judge guard different failure classes and must be stacked — and the costs
are measured: ~100× latency, one negation-phrased answerable query wrongly rejected (MRR
1.000→0.929). Abstention quality is bounded by the judge, exactly as gap detection is bounded
by the embedder. Full study:
[docs/ENTAILMENT_SUPERSESSION_STUDY.md](ENTAILMENT_SUPERSESSION_STUDY.md).

## 4. What this demonstrates

For a reviewer, the signal in this repo is less "it retrieves" and more *how the retrieval was
engineered and judged*:

- **RAG beyond the demo**: hybrid dense + sparse retrieval with RRF, cross-encoder reranking, and a
  pluggable embedder abstraction — on a production-shaped Postgres + pgvector stack, not a toy
  in-memory index.
- **Calibrated abstention as a first-class feature.** The `gap_warning` guard, the trust verdicts,
  and the calibration study behind them treat "knowing when you don't know" as the actual
  deliverable — the thing that makes retrieval safe to wire into an autonomous agent's decision
  loop. The v0.2 evaluation makes the failure concrete: on validity-sensitive queries, plain
  vector search returns the superseded memory as the answer 83–100% of the time; the trust layer
  never does (superseded-trust rate 0.00) at zero cost to ordinary retrieval (FINDINGS §4).
- **Evaluation rigor**: an ablation harness with ranking metrics *and* a guard-specific
  false-confident rate; per-embedder threshold calibration; a held-out fine-tuning split.
- **Intellectual honesty.** Two of the four headline findings are negatives — a threshold that
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
