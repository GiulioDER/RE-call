# Running RE-call inside Mem0's own benchmark harness

Every RE-call number published before this document was produced by **RE-call's** harness. Mem0's
published numbers (LOCOMO 92.5, LongMemEval 94.4, BEAM 1M 0.641 avg score / 70.1% pass) were
produced by **theirs**. Those two sets of numbers are not comparable, and no wording makes them
comparable: different answerer prompt, different judge, different retrieval budget, different
scored subset.

This document is the procedure for removing that excuse — running RE-call through
[`mem0ai/memory-benchmarks`](https://github.com/mem0ai/memory-benchmarks) with their prompts,
their judge, their cutoffs and their metric left byte-for-byte untouched.

---

## 1. What is ours and what is theirs

| Piece | Whose | Modified? |
|---|---|---|
| Answerer prompt, judge prompt, judge rules, category filter, cutoffs, metric | theirs | **no** |
| Ingestion chunking (`CHUNK_SIZE`), `user_id` scoping, dataset parsing | theirs | **no** |
| Memory backend behind `add()` / `search()` | ours (`recall_interop.RecallBackend`) | new |
| A `--memory-backend {mem0,recall}` flag in their two runners | ours | **+39 lines each, shown below** |

The entire change to their repo is in
[`recall_interop/memory-benchmarks-backend-swap.patch`](../recall_interop/memory-benchmarks-backend-swap.patch):
five `add_argument` calls and one `if/else` around the constructor, in
`benchmarks/locomo/run.py` and `benchmarks/beam/run.py`. Nothing else in their tree is touched.

```bash
git clone https://github.com/mem0ai/memory-benchmarks.git
cd memory-benchmarks
git apply /path/to/recall/recall_interop/memory-benchmarks-backend-swap.patch
```

## 2. The seam

Their runners talk to the memory system through exactly two async calls
(`benchmarks/locomo/run.py:417`, `benchmarks/beam/run.py:690`):

```python
await mem0.add(messages, user_id, timestamp=<epoch>)   # -> {"results": [...]}
await mem0.search(question, user_id, top_k=200)        # -> [{"memory", "created_at", ...}]
```

`recall_interop.RecallBackend` implements that pair (plus `delete_user` / `close` / the
async-context protocol) on top of `recall.index.Indexer` + `recall.trust.trusted_search`. The four
adapter decisions that change a number — and why each is the faithful choice rather than the
flattering one — are documented at the top of
[`recall_interop/memory_benchmarks.py`](../recall_interop/memory_benchmarks.py). In short:

1. One document per `add()` call, message content **verbatim** — the session date is not injected
   into the text, because Mem0's extractor does not see it either. Both systems get it as
   `timestamp` → `created_at`, which their answerer prints.
2. `created_at` rides in the document filename (RE-call's frontmatter parser keeps only its own
   validity keys, and using `valid_from` would change what the trust layer *decides*). **No RE-call
   library code is modified to make this benchmark work.**
3. `score` is a rank-preserving surrogate, not the cosine. Their harness re-sorts by `score` and
   slices `[:cutoff]`, so `score` *is* the ranking; RE-call ranks by RRF fusion while each hit
   carries its true dense cosine, so handing them the cosine would silently re-rank RE-call by its
   dense leg alone. The real cosine, confidence, trust verdict and RE-call rank are all reported in
   `score_debug`, which their harness writes verbatim into the per-question artifact.
4. The candidate pool is widened to the requested `top_k`.

### The trap in (4), because it is worth a paragraph

`recall.retriever.DEFAULT_CANDIDATE_K` is **20**, so the fused pool holds at most ~40 distinct
chunks. Left at the default, `--top-k 200` returns ~40 memories and RE-call loses every 200-budget
cell for a reason that has nothing to do with retrieval quality. `trusted_search(candidate_k=...)`
exists for exactly this, and the value used is recorded in `RecallBackend.describe()`.

The same class of cap exists one level down: an HNSW scan cannot return more rows than it examined,
so `LIMIT k` with `k > hnsw.ef_search` silently yields `ef_search` rows. `recall/store.py` widens
`ef_search` to `4 * k` on the unfiltered arm as of **0.6.0** — earlier versions, including the
`bench/head-to-head` branch that produced the paired-harness article (0.5.2), do **not**, and would
cap the dense leg at ~40 regardless of `candidate_k`. Their-protocol runs must therefore be done on
0.6.0+; a run on 0.5.2 would report a handicap as a result.

## 3. Reproduce

Prerequisites: local pgvector (`docker compose up -d` in this repo) for the RE-call arm; Mem0's own
`docker compose up -d` (Mem0 server on `:8888` + Qdrant on `:6333`) for the Mem0 arm.

```bash
# RE-call arm — the memory layer costs $0 (local bge-small, no LLM in RE-call's path).
# Only the answerer + judge spend, and both are THEIR models on THEIR prompts.
PYTHONPATH=/path/to/recall RECALL_DSN=postgresql://recall:recall@localhost:5432/recall \
python -m benchmarks.locomo.run \
  --project-name recall-theirs \
  --memory-backend recall \
  --dataset-path /path/to/locomo10.json \
  --top-k 200 --top-k-cutoffs 200 \
  --answerer-model openai/gpt-4o --judge-model openai/gpt-4o
```

```bash
# Mem0 arm — same harness, same models, same cutoff. Their OSS server does the extraction.
python -m benchmarks.locomo.run \
  --project-name mem0-theirs \
  --backend oss \
  --dataset-path /path/to/locomo10.json \
  --top-k 200 --top-k-cutoffs 200 \
  --answerer-model openai/gpt-4o --judge-model openai/gpt-4o
```

`--predict-only` runs ingest + search and stops before the answerer, so the whole plumbing can be
validated for **$0**. Do that first; it is also how the retrieved-memory dumps are produced without
paying for a judge.

### Two labelling constraints that are not optional

- **Their published 92.5 is Mem0 *Platform* (cloud v3) at `top_200`, and their runner's default
  answerer/judge is `gpt-5`.** A run against Mem0 **OSS** with **gpt-4o** is a different cell. It is
  still enormously more comparable than a cross-harness comparison — and it comes with a *matched*
  Mem0 column, which is what makes the RE-call-vs-Mem0 contrast internally valid — but it is not
  the same number as 92.5 and must never be printed as if it were.
- **Routing through OpenRouter pins the model to `openai/gpt-4o`.** Their
  `LLMClient._openai_chat_token_limit_kwargs` selects `max_completion_tokens` vs `max_tokens` from a
  bare `gpt-5`/`o*` prefix on the model string, so an OpenRouter-namespaced `openai/gpt-5` would be
  sent `max_tokens` + `temperature=0` and rejected. Running gpt-5 needs a direct OpenAI key, not a
  patch to their client — patching it would break the "their harness, unmodified" claim.

## 4. Costs, measured not estimated

Their LOCOMO run scores categories 1-4 = **1,540 questions**, and evaluates each at every
`--top-k-cutoffs` value (default four: 10/20/50/200) — four answerer calls **and** four judge calls
per question. Restricting to `--top-k-cutoffs 200` is what makes the run affordable, and it is also
the only cutoff their headline 92.5 is quoted at.

RE-call's own side of the run is free and measurable: local bge-small ingest of one LOCOMO
conversation (419 chunks) takes ~3m40s, and search at `top_k=200` returns in ~120 ms.

Every published cell must carry: harness, answerer model, judge model, retrieval budget, memory
backend + version, embedder, and date. Metered spend is reported per arm from the provider's own
usage figures, never from an estimate.

## 5. Tests

`tests/test_their_harness_backend.py` pins the adapter contract with no API calls: the dict keys
their answerer reads, `created_at` round-tripping the ingest timestamp, `top_k` being honoured past
the default candidate pool, `score` surviving their re-sort, tenant isolation, idempotence across
re-runs, and abstention propagating as an empty list. Two of those were verified by mutation —
reverting the `candidate_k` widening and returning the cosine as `score` each turn a test red.

`recall_interop` is in mypy's `files` (though deliberately not in the wheel's `packages`): it is the
code path behind published numbers.
