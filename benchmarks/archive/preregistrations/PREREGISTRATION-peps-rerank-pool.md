# PREREG — does POOL WIDTH explain the 07-22 rerank null?

Written 2026-08-04 before any arm was run. Branch `bench/peps-rerank-pool` off `53fe2a4`.

## The contradiction being resolved

| corpus | rerank effect | source |
|---|---|---|
| private memory, 46 held-out Qs | **+0.043 hit@5, within noise**, 57x latency; pool 20→100 **+0.000** | 2026-07-22 |
| LOCOMO, n=200, pool-level | **+0.145 hit@5** CI [0.095, 0.200] | 2026-07-30 `rerank_pool_arms.json` |

The 07-22 levers were measured **one at a time**: rerank at the default pool, and pool widening
without rerank. **The interaction was never measured.** The best-config memo's central claim is that
the two are not independent (a reranker whose pool equals its output reorders a prompt instead of
selecting what goes into it).

**Hypothesis H:** the 07-22 null measured the WEAK form. Rerank's value comes from selecting out of
a wide pool, so it scales with pool width, and this is a property of the FORM rather than of the
corpus.

**Rival hypothesis R:** the 07-22 null is a property of that CORPUS (`hit@50` plateaued ~0.50, a
recall ceiling), and pool width will not rescue it anywhere outside LOCOMO.

## Apparatus

`recall/eval/labelled.py` (the shipped runner) on the **PEPs** arm: 733 `.rst` from `python/peps`,
110 file-level labelled questions (`recall/eval/peps_questions.json`, the repo's stated public
reproducible arm). Local `fastembed` bge-small, local `ms-marco-MiniLM-L-6-v2`, isolated pgvector on
port **5434** (5432 and 55432 belong to other worktrees; do not touch them). `k=5` throughout, so
only the POOL varies. One index, reused across both runs via `--table peps_bge`.

Four arms from two invocations (each `--rerank` run reports both `hybrid` and `hybrid+rerank`):

| arm | candidate_k | rerank |
|---|---|---|
| A1 | 20 | no |
| A2 | 20 | yes  ← the 07-22 form |
| A3 | 250 | no |
| A4 | 250 | yes ← proposed config B |

## Predictions

| # | Prediction | Discriminates |
|---|---|---|
| P1 | A4 − A3 >= **+0.08** hit@5 | rerank helps at a wide pool on a non-LOCOMO corpus |
| P2 | **A4 − A2 >= +0.04** | THE INTERACTION. The whole question. |
| P3 | A3 − A1 <= **+0.02** | pool width alone does little (reproduces 07-22's `+0.000`) |
| P4 | A2 − A1 <= **+0.06** | the weak form is weak here too (reproduces 07-22's `+0.043`) |

**H predicts P1-P4 all hold.** **R predicts P1 and P2 FAIL** (rerank stays flat regardless of pool).
If P4 fails HIGH (rerank already big at pool 20), pool width is NOT the explanation and the 07-22
null is corpus-specific in a way this experiment cannot attribute.

## Invariants

- **I1** both runs read the SAME index (same `--table`, reused by content hash). A re-embed between
  arms would confound pool width with index drift.
- **I2** the `hybrid+rerank` arm must differ from `hybrid` on at least one question. Identical
  output means the reranker did not run, or was inert, and the arm is not a measurement.
- **I3** all 110 questions load and every answerable one carries `relevant_files` (the runner
  asserts this itself and exits non-zero otherwise).
- **I4** `candidate_k` is echoed in the report and must equal what was requested.

## What this does NOT settle

The private memory corpus. PEPs is a third corpus, so a positive result supports the FORM
explanation but does not prove rerank helps on the user's own memos. That arm needs the 46 held-out
private questions, which are not on this machine.
