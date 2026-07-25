# Voyage AI models — reference & RE-call recommendations

> Compiled 2026-07-24 from the official docs (the source of truth). Verify against the live pages
> before relying on a number — Voyage revises models and pricing.
> Sources: [embeddings](https://docs.voyageai.com/docs/embeddings) ·
> [contextualized-chunk](https://docs.voyageai.com/docs/contextualized-chunk-embeddings) ·
> [multimodal](https://docs.voyageai.com/docs/multimodal-embeddings) ·
> [reranker](https://docs.voyageai.com/docs/reranker) ·
> [pricing](https://docs.voyageai.com/docs/pricing)

## 0. The headline for us

- **RE-call's `VoyageEmbedder` defaults to `voyage-3`, which is now LEGACY.** The current
  generation is **voyage-4**. Our "best config" is running an outdated model.
- **The whole LOCOMO benchmark on Voyage is effectively FREE.** The free tier is **200M tokens/month**
  on all current models (voyage-4-*, voyage-context-4, voyage-code-3) and rerankers. LOCOMO's 10
  conversations are well under 1M tokens — so embedder cost is not a constraint for this work.
- **The model that targets our actual failure is `voyage-context-4`, not a bigger flat embedder.**
  Our cat1 loss is retrieval recall: a conversation turn embedded *in isolation* is ambiguous, so
  the answer turn isn't retrieved (71% of cat1 failures had no gold in context). Contextualized
  chunk embeddings embed each turn *within its conversation* — precisely that problem.

## 1. Text embedding models (standard `embed` API)

| model | dims (default/options) | context | price /M tok | free tier | note |
|---|---|---|---|---|---|
| **voyage-4-large** | 1024 / 256·512·2048 | 32K | **$0.12** | 200M/mo | "best general-purpose & multilingual retrieval quality" |
| voyage-4 | 1024 / 256·512·2048 | 32K | $0.06 | 200M/mo | general-purpose, cheaper |
| voyage-4-lite | 1024 / 256·512·2048 | 32K | $0.02 | 200M/mo | latency/cost optimized |
| **voyage-4-nano** | 1024 / 256·512·2048 | 32K | — | — | **OPEN-WEIGHT (HuggingFace)** → run local |
| voyage-code-3 | 1024 / 256·512·2048 | 32K | — | 200M/mo | code retrieval |
| **voyage-finance-2** | 1024 | 32K | (older tier) | none | **finance retrieval/RAG** |
| voyage-law-2 | 1024 | 16K | (older tier) | none | legal retrieval/RAG |
| voyage-3-large *(legacy)* | 1024 | 32K | $0.18 | none | superseded by voyage-4-large |
| voyage-3 *(legacy — our default)* | 1024 | 32K | $0.06 | none | **superseded; update this** |

Matryoshka dims (256/512/2048) let you trade storage/latency for quality at the same model — a
smaller dim shrinks the pgvector index. Default 1024 matches our current schema.

## 2. Contextualized chunk embeddings — `voyage-context-4` (the promising one)

- **What it does:** embeds each chunk *within the context of the other chunks from the same
  document*, instead of in isolation. Standard chunking "loses the broader document context that
  helps disambiguate meaning"; this restores it.
- **Specs:** 1024 default (256/512/2048), **120K tokens/document total, 32K/chunk**, **$0.12/M**,
  200M/mo free. (`voyage-context-3` is the superseded predecessor.)
- **Different API:** input is `List[List[str]]` — each inner list is one document's chunks,
  embedded as a group. Or `List[str]` documents with `enable_auto_chunking=True`.
- **Why it maps onto RE-call exactly:** we index each LOCOMO dialogue turn as its own document.
  A turn like *"Yeah, the 14th"* is meaningless alone and won't be retrieved for *"when did X
  happen?"*. Feeding a whole conversation as one document's chunk-list would embed that turn aware
  of the surrounding turns — the single most direct fix for the cat1 retrieval-recall gap.
- **Cost of adoption:** NOT a drop-in model swap. Needs a new embedder that calls the
  contextualized endpoint and groups a conversation's turns as one document. Medium effort, high
  expected payoff.

## 3. Rerankers (cross-encoder, second stage after retrieval)

| model | context | price /M tok | free |
|---|---|---|---|
| **rerank-2.5** | 32K | $0.05 | 200M/mo |
| rerank-2.5-lite | 32K | $0.02 | 200M/mo |
| rerank-2 | 16K | $0.05 | 200M/mo |
| rerank-2-lite | 8K | $0.02 | 200M/mo |

- **What it does:** jointly encodes (query, document) pairs → relevance score 0–1, sorted. More
  accurate than embedding cosine because it reads the pair together. API: `vo.rerank(query,
  documents, model="rerank-2.5", top_k=...)`, up to 1,000 docs, query ≤ 8K tokens.
- **Relevance to RE-call:** RE-call already HAS a reranker stage (`recall.rerank`, cross-encoder).
  Swapping in `rerank-2.5` — retrieve top-N cheaply, rerank to top-k — is a standard, high-yield
  recall/precision boost, and it's the classic remedy when "the answer was retrieved but ranked
  below the k cutoff." Pairs naturally with a stronger first-stage embedder.
- **The docs give no benchmark numbers here** — so any improvement claim must be *measured on our
  data*, not asserted from the page.

## 4. Multimodal — `voyage-multimodal-3.5`

Embeds interleaved text + images (screenshots, PDFs, slides, tables). 1024 dims (256/512/2048),
32K context. **Not relevant to LOCOMO** (pure text) and the docs make no claim it improves
pure-text retrieval. Note for a future image/PDF-memory feature only.

## 5. Recommendation for RE-call, ranked by expected payoff on the cat1 recall gap

1. **`voyage-context-4`** — the architecturally-correct fix (turn-in-conversation embeddings).
   Highest expected impact; needs a new embedder class. **The experiment worth building.**
2. **`voyage-4-large`** — drop-in top general embedder; the cheap screen for "does a strong flat
   embedder move cat1 at all" before investing in (1). Just a model-name change from `voyage-3`.
3. **`rerank-2.5` as a second stage** — orthogonal, stacks with (1)/(2); RE-call already has the
   slot for it.
4. **`voyage-finance-2`** — only for the exactness/finance positioning thread (Q6): if we build a
   finance/dates benchmark to demonstrate "RE-call for domains where exact data matters," this is
   the domain embedder to test there. Not for LOCOMO.
5. **`voyage-4-nano` (open-weight)** — the "save server resources" angle cuts both ways: it's a
   *high-quality* model you can run LOCAL, so it may beat bge-small/large without an API dependency.
   Worth a data point if we care about keeping embedding in-house.

## 6. Concrete next experiments (all effectively free under the 200M tier)

- **A. `voyage-4-large` (drop-in), 4 conv, gpt-4o-mini** — screen cat1 gold-in-context vs bge-large.
- **B. If A helps: `voyage-context-4`** (new embedder, turns grouped per conversation) — the real test.
- **C. Winner of A/B under Sonnet vs Mem0** (Mem0 on its own recommended embedder, for fairness).
- **D. `rerank-2.5`** stacked on the winner — measure the marginal recall/precision gain.

Every number these produce must be **measured on our data** — the docs give specs and prices, not
retrieval-quality guarantees for our corpus. Official docs are truth for *what the models are*, not
for *how well they'll do on LOCOMO*; that we prove ourselves.
