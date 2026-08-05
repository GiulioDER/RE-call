# PEPs rerank x pool-width 2x2

Raw reports behind the **2026-08-05** finding that the rerank/pool-width **interaction is negative**:
widening the candidate pool makes the local `ms-marco-MiniLM-L-6-v2` cross-encoder *worse*, not
better. Preregistered 2026-08-04 before any arm ran.

**Prior work searched** (`docs_search(source_type='memory')`, plus the closed-hypotheses index):
the 2026-07-22 `RE-call retrieval levers` closure (rerank +0.043 within noise, pool 20 to 100
+0.000, `hit@50` plateau read as a recall ceiling), `closed-hypothesis-recall-reranker-finetune-2026-07-30`
(the LOCOMO pool-level baseline: MiniLM +0.145, voyage +0.230), and
`project-docs-rag-reranker-voyage-vs-minilm-2026-07-30`. This experiment measures the one cell those
leave open, the **interaction**, which 07-22 never ran. `gap_warning: false` on both queries.

## Provenance

Produced by `python -m recall.eval.labelled` at commit **`de2a712`**, not at this branch tip. The
tip merged `origin/master`, which edited migration `0008` in place; against a pre-existing database
that raises `MigrationChecksumMismatch`, so the arms were run from a worktree pinned at `de2a712`.
Pinning also removed a code-drift confound: **all four arms come from one commit.**

Shared apparatus, so every delta is **paired**: one index (`peps_bge`, 743 sources, 21961 chunks,
isolated pgvector), the same 88 answerable questions from `recall/eval/peps_questions.json`,
local `fastembed` bge-small, `k=5` throughout so only the pool varies.

```
--corpus <peps>/peps --questions recall/eval/peps_questions.json --glob '*.rst' \
--embedder fastembed -k 5 --candidate-k {20,250} --rerank --score-retrieval-on all \
--table peps_bge
```

Each `--rerank` invocation reports both a `hybrid` and a `hybrid+rerank` arm, so two invocations
give the full 2x2.

| file | n | candidate_k | arms |
|---|---|---|---|
| `peps_n88_ck20.json` | 88 | 20 | A1 `hybrid` 0.6250 · A2 `hybrid+rerank` 0.6364 |
| `peps_n88_ck250.json` | 88 | 250 | A3 `hybrid` **0.6477** · A4 `hybrid+rerank` **0.5795** |
| `peps_n44_ck20.json` | 44 | 20 | earlier held-half pair, kept as the replication check |
| `peps_n44_ck250.json` | 44 | 250 | idem |

The `n44` pair predates `--score-retrieval-on` and scored retrieval on the held half only
(`questions[1::2]`, a deterministic stride, so it is still paired across runs). It replicates the
`n88` result in direction on all four predictions.

## Result

| # | prediction | delta (paired) | 95% CI | verdict |
|---|---|---|---|---|
| P1 | A4 − A3 >= +0.08 | −0.0682 | [−0.1591, +0.0227] | FAILS, wrong sign |
| P2 | A4 − A2 >= +0.04 (the interaction) | −0.0568 | [−0.1250, +0.0000] | FAILS, wrong sign |
| P3 | A3 − A1 <= +0.02 | +0.0227 | [−0.0455, +0.0909] | fails by ~1/4 of one question |
| P4 | A2 − A1 <= +0.06 | +0.0114 | [−0.0682, +0.0909] | HOLDS |

For **both P1 and P2 the entire 95% CI sits below the preregistered threshold**. McNemar is 0.24 and
0.18, so this is not a demonstration that reranking *hurts*; it is a demonstration at 95% that it
**does not deliver the preregistered benefit**.

Mechanism, and it is a dose-response the wrong way: rerank broke/fixed **6/7 at pool 20** (churn,
p=1.00) but **12/6 at pool 250**. A wider pool does not give the cross-encoder more to select from,
it gives it more rope.

## Reading these files

- Per-arm `misses` lists are **untruncated** and carry question ids, so per-question hit vectors are
  recoverable after the fact. That is what makes the paired bootstrap and McNemar possible without
  re-running anything.
- The apparatus invariant is on the **sample sizes**, not the rates: `retrieval_scored_on` 88,
  `false_abstain.n` 44, `abstention_accuracy.n` 11. ⚠️ `false_abstain.rate` legitimately moves with
  `candidate_k` (0.0227 at ck20, 0.0909 at ck250) because `research_search` shares the pool
  (EVAL-002). Asserting the *rate* would false-void a valid arm.
- ⚠️ **Latency figures in these files are not usable.** Concurrent test sweeps ran on the same box
  for part of every run. `hit@5` is unaffected, retrieval being deterministic on a fixed index.

## Three corpora, one model

The 07-30 work found this same English-only MiniLM collapses hit@1 from 0.590 to 0.190 on a
*bilingual* corpus, and proposed that language probably also explained the 07-22 null. PEPs rules
that out as the whole story: English corpus, English model, still negative.

| corpus | language | MiniLM effect |
|---|---|---|
| LOCOMO (conversational) | English | **+0.145** hit@5 |
| PEPs (technical prose) | English | **−0.068** at pool 250, +0.011 at pool 20 |
| sentiment-agent memos | bilingual | **−0.400** hit@1 |

A local cross-encoder's value is corpus-dependent **even holding language fixed**. Switching to a
multilingual cross-encoder would fix the bilingual collapse but would not have rescued PEPs.

## Scope

Says nothing about `voyage:rerank-2.5`, a different and stronger reranker. Does not prove reranking
fails on the private memory corpus; it removes the "wide pool is what rerank needs" explanation,
which leaves LOCOMO as the outlier rather than that corpus.
