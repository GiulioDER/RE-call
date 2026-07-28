# A measured rerank arm for LOCOMO — design

**Date:** 2026-07-27 · **Status:** design approved, not yet implemented
**Predecessor:** `2026-07-26-embedder-gap-predictor-design.md` (the study that made this the obvious
next lever)

## Why

§9a measures LOCOMO at **hit@5 0.671** [0.647, 0.694] and **hit@20 0.855** [0.836, 0.872], n=1 536
answerable, bge-small, hybrid, pool 20, **no rerank**.

For 85.5% of questions the correct memory is already in the retrieved pool and merely ranked below
position 5. Closing that **18.4-point** gap is precisely what a cross-encoder reranker does, and
RE-call already ships one (`CrossEncoderReranker`, `ms-marco-MiniLM-L-6-v2`, pinned revision). It has
never been measured on LOCOMO — every published LOCOMO figure in this repo is `no rerank`.

For scale: 18.4 points is roughly **three times** the entire local-vs-cloud embedder gap measured on
2026-07-27 (+0.059 median across 17 corpora).

The mechanism is viable because `HybridRetriever` reranks the **whole fused pool and truncates
afterwards** (`recall/retriever.py`), which is the only ordering under which reranking can move
hit@5 at all. The same ordering preserves the "top-k is a prefix of top-max(k)" property that §9a's
single-retrieval depth curve depends on, so the curve logic needs no change.

## The change

`recall/eval/locomo.py:264` constructs `HybridRetriever(store, embedder, candidate_k=candidate_k)`
with no reranker. Thread a `reranker` parameter through `run_conversation` → `run` → a `--rerank`
CLI flag. `recall/eval/labelled.py` already exposes exactly this flag; this follows that pattern
rather than inventing one.

## Arms

| arm | reranker | why |
|---|---|---|
| `baseline` | none | re-run, not quoted — both arms from one session on one machine |
| `rerank-shipped` | `ms-marco-MiniLM-L-6-v2` (default) | what users actually get today |
| `rerank-modern` | `bge-reranker-base` | separates two very different failures — see below |

**The second reranker is the point, not a luxury.** §10b measured `rerank_top1` at **0.742 AUC**
against `dense_top1` at **0.753** — that exact cross-encoder scored *below plain dense similarity*
on that task. If the shipped model fails here alone, the result cannot distinguish:

- *"reranking does not help on conversational memory"* → closes the lane, and
- *"a 2019 web-search cross-encoder does not transfer"* → swap the model.

Those have opposite consequences. `CrossEncoderReranker` already accepts a `model` argument, so the
second arm costs one extra pass (~1 hour) and no new code.

## Preregistered before running

- **The ceiling is 0.855**, not 1.0. Any gain is reported against hit@20, because that is what
  "the document was retrieved but mis-ranked" can possibly recover.
- **Rerank may make it worse.** A poorly-transferring cross-encoder reorders a decent ranking into a
  bad one. That outcome is published as measured — it is the §10b prior coming true, and it is
  useful.
- **Latency is measured here, not carried over.** §1's ~1 900 ms/query was a different corpus on a
  different machine. Rerank cost is part of the verdict, not a footnote: at ~2 s/query this is a
  configuration, not a default.
- **Per-category is reported.** cat3 is the floor (0.620 at k=20 against 0.837 for cat1) and is where
  a reranker should help most if it helps anywhere.
- **Success is a defensible measurement, not a win.** Both directions ship.

## Out of scope

Pool-100 arms, any embedder change, LongMemEval. One variable at a time — the pool-100 control in
§9a exists already and mixing it in would confound the reranker's effect with pool depth.

## What gets published

Full hit@k curve for all three arms with CIs, measured rerank latency per query, per-category
breakdown, and — if a rerank arm wins — an explicit statement of what it costs in latency, so the
"$0 and fast" claim stays honest.

Whatever the outcome, §9a gains the arm it has been missing and its `no rerank` caveat stops being
an open question.
