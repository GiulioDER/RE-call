# The Answerability Ladder v4 — the tuned arm

**Date:** 2026-07-31 · **System:** RE-call 0.6.0, `recall-tuned` arm — `BAAI/bge-small-en-v1.5`,
**k=45, candidate_k=250, `cross-encoder/ms-marco-MiniLM-L-6-v2`**
**Manifest:** `results/ladder/manifest_v2.jsonl`, digest `5534c613…` — the same frozen manifest as
v2 and v3, verified at run start via `--expected-digest`
**Responses:** `responses_v4_tuned.jsonl`, 1200 scored, none missing. 9 h 40 m on a 12-core VPS.
**Pre-registration:** [`benchmarks/PREREGISTRATION-ladder-v4.md`](../../benchmarks/PREREGISTRATION-ladder-v4.md)
**Analysis:** `benchmarks/ladder/analyze_v3.py`, **unchanged** — reused verbatim so the analysis
could not be selected after seeing the numbers.

**Prior work** — searched before the arm was designed, not after it ran; the full record is in
[`PREREGISTRATION-ladder-v4.md`](../../benchmarks/PREREGISTRATION-ladder-v4.md) §2.
`docs_search(source_type="memory", query="reranker abstention interaction candidate_k pool width
top cosine ladder tuned arm")` returned four load-bearing hits, **all of which predicted this
null**: [[project-recall-nearmiss-signal-exhaustion-2026-07-29]] §8f/§8g/§8i, plus `FINDINGS.md`
§11–§12 on branch `docs/rerank-abstention-interaction`. That §9 scorecard already records "raise
depth first" as falsified. This arm was run anyway because it measures a different signal
(`top_cosine`, the shipped decision input) on a **free** reranker across all five rungs at n=1200,
where the prior work scored a null-head reader on *voyage* at ring 0 with n=400.

## Verdict: the best free configuration changes nothing on this axis

The comparison is single-variable: same embedder, same manifest, same decision rule as v2. Only
retrieval configuration moved.

```
AUC(answerable vs rung)     v2 default   v4 tuned    delta
  r=0.00                       0.567       0.561     -0.006
  r=0.25                       0.784       0.800     +0.016
  r=0.50                       0.841       0.839     -0.002
  r=0.75                       0.921       0.923     +0.002
  r=1.00                       0.968       0.971     +0.003
```

**Within 0.016 at every rung.** The largest retrieval gain ever measured in this project — hit@5
0.671→0.777, ~2× the best embedder effect ([`FINDINGS.md` §11](../FINDINGS.md)) — buys **nothing**
here. That is a sharper statement than another null on a mediocre config: it is not that we failed
to tune, it is that tuning this well moves this axis not at all.

This is what `FINDINGS.md` §12c predicted, on a different signal. §12c measured a null-head reader
on *voyage*-reranked retrieval at ring 0 (n=400) and found the ceiling rising while the signal
stayed flat. v4 measures `top_cosine` — the shipped system's own decision input — on a **free**
reranker, across all five rungs, n=1200. Same conclusion, wider evidence.

## Scorecard: 3 of 4 predictions PASS, and the one that failed was mine

| | prediction | measured | |
|---|---|---|---|
| **P1** (kill) | \|AUC_r0 − 0.5674\| ≤ 0.03 | 0.561, Δ **0.0064** | ✅ null holds |
| **P2** | below-floor rate ≤ 0.000833 | **0.001667** (2/1200) | ❌ **FAIL** |
| **P3** | Δmean_unans ≥ Δmean_ans − 0.005 | −0.0018 vs −0.0050 | ✅ |
| **P4** | per-question monotone ≥ 0.70 | 0.775 | ✅ |

### P2 failed, and the reasoning behind it was wrong

I predicted the floor would become **more** inert, on this argument: *"a 250-document pool has a
maximum dense cosine at least as high as a 20-document pool's, so scores move up, away from the
floor."*

The premise is true and the conclusion does not follow. `top_cosine` is not the maximum cosine over
the **pool** — it is the maximum over the **45 documents the reranker selected from** that pool.
The cross-encoder reorders by its own relevance score and keeps the top k; nothing obliges it to
keep the highest-cosine document. So a wider pool plus a reranker can *lower* the recorded score,
and it did: 2 responses below the floor instead of 1.

The magnitude is negligible — 2/1200 versus 1/1200, still an inert floor by any reading — but the
direction is opposite to what I registered, and the mechanism I gave was wrong rather than
imprecise. Recorded here rather than smoothed into "P2 essentially held".

## What this does NOT establish

- **Not that reranking is useless.** §11 and §12a measure it winning on retrieval, decisively.
  Retrieval quality and answerability discrimination are separate axes; that separation is the
  result, and it is the claim mem-bench exists to make measurable.
- **Not a statement about `voyage-4-large`.** The paid half of the published best configuration was
  not run and remains untested here.
- **Per-question monotonicity fell**, 0.865 → 0.775. Above P4's bar, but the reranked selection is
  visibly noisier per question than plain dense retrieval, which is consistent with the P2
  mechanism above and worth remembering before anyone reads 0.775 as "unchanged".
- **Same corpus, same distractors, no judge.** A shared-corpus artefact would reproduce here rather
  than be caught.
