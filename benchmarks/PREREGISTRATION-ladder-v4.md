# Pre-registration — Answerability Ladder v4: the tuned retrieval arm

**Date:** 2026-07-30 · **Status:** written and committed BEFORE the arm runs
**Manifest:** `results/ladder/manifest_v2.jsonl`, digest
`5534c61356acaa7b62ac5a79dbec7383674fc052984d10c1d0cc89e26a532bd5` — the SAME frozen manifest as
v2 and v3, verified at run start via `--expected-digest`
**Analysis:** `benchmarks/ladder/analyze_v3.py`, unchanged and already committed. Reusing the v3
analysis verbatim is deliberate: an analysis written or edited for this arm would be a free
parameter, and this arm exists partly to test a claim I would prefer to be wrong about.

## 1. What this arm changes, and what it holds fixed

RE-call's published best configuration is `--embedder voyage:voyage-4-large --k 45
--candidate-k 250 --reranker local` ([[reference-recall-best-configuration-2026-07-28]]). This arm
runs the **free half** of it:

| | headline arm (v2) | this arm (v4) |
|---|---|---|
| embedder | `BAAI/bge-small-en-v1.5` | **unchanged** |
| `k` | 5 (shipped default) | **45** |
| `candidate_k` | 20 (shipped default) | **250** |
| reranker | none | **`cross-encoder/ms-marco-MiniLM-L-6-v2`** (local, pinned revision) |

`voyage-4-large` is deliberately NOT run: it is a paid API, and paid work on this project is closed
by standing decision. Holding the embedder fixed at v2's is not a compromise — it is what makes
this a **paired, single-variable comparison** against an arm already on the board. Any difference
measured here is retrieval configuration, not embedding.

The arm is **separately labelled** (`recall-tuned`), per `SUITE-DESIGN.md` rule 4. The headline arm
stays at shipped defaults and does not quietly become the tuned one.

## 2. Prior work, and why this is not a re-run of it

**Prior work searched** — `docs_search(source_type="memory", query="reranker abstention interaction
candidate_k pool width top cosine ladder tuned arm")`, plus a direct read of
`results/FINDINGS.md` §11–§12 on branch `docs/rerank-abstention-interaction` (`b5db888`). Four hits
are load-bearing and **all four argue this arm will produce a null**; they are cited in place below
rather than listed. Chief among them: [[project-recall-nearmiss-signal-exhaustion-2026-07-29]]
§8f/§8g/§8i, and its §9 scorecard which records **"raise depth first" as already FALSIFIED**.

This section exists because skipping it on 2026-07-28 cost ~4 h re-measuring six abstention signals
that had been measured on 07-24 with the same conclusion.

**`results/FINDINGS.md` §12 already measured reranking against abstention on these very ladder
arms, and the result was negative.** §12c: a span-grounded null head re-measured on reranked
retrieval leaves the best AUC **unchanged at 0.7396**, while the fraction of the achievable ceiling
*falls* from 65.7 % to 49.6 %. The mechanism §12c gives is general and is the reason this arm is
predicted to fail:

> a reranker optimises relevance, and the near-miss class is defined as maximally relevant and
> answer-free, so better relevance ranking packs more convincing distractors into the window.
> Relevance is not answerability.

§12b separately showed pool-level reranking lifts the retrieval *ceiling* above 0.90 — the ceiling
moves, the signal does not follow it.

**What the prior work did not measure, and this arm does.** Four differences, none cosmetic:

| | §8i / §12c | this arm |
|---|---|---|
| reranker | `voyage:rerank-2.5` (**paid**) | `ms-marco-MiniLM-L-6-v2` (**local, free**) |
| signal scored | null-head reader (`deberta-v3-base-squad2`) over the window | **`top_cosine`** — the shipped system's own score |
| rungs | ring 0 only | **all five** (0.00 / 0.25 / 0.50 / 0.75 / 1.00) |
| n | 200 + 200 | **1200** + 200 answerable |

The signal difference is the substantive one. `CrossEncoderReranker.rerank` **reorders only — every
hit keeps its dense cosine** — so under this config `top_cosine` becomes *"the maximum dense cosine
among the 45 documents the reranker selected out of a 250 pool"*. That is neither the dense top-1
nor the reader's judgement, and no prior arm has recorded it. It is also the number that decides
the shipped abstention, which is what mem-bench's axis publishes.

So: the same mechanism under test, on a **free** reranker, against the system's **own** decision
rather than a research probe, at 6× the n, across the whole ladder rather than its first rung.

## 3. Predictions

Registered before any v4 data exists. **P1 is the kill condition.**

**P1 (KILL).** Discrimination at the near rung does not materially improve:
`|AUC_v4(answerable vs r=0.00) − 0.5674| ≤ 0.03`.
*If AUC rises by more than 0.05, §12c's mechanism claim does not transfer to this signal and that
is a real, publishable discovery — the arm would then be the interesting one, not the null.*

**P2.** The shipped 0.50 floor becomes **no less inert**: `below_floor_rate_v4 ≤ 0.000833` (v2's
1/1200). Rationale: a 250-document pool has a maximum dense cosine at least as high as a
20-document pool's, so scores move up, away from the floor. A guard already firing once in 1200
cannot be fixed by raising the distribution further from it.

**P3 (the mechanism test).** Mean `top_cosine` rises on **both** arms, and the rise on the
unanswerable arm is **at least as large** as on the answerable arm:
`Δmean_unanswerable ≥ Δmean_answerable − 0.005`.
*This is what "relevance is not answerability" predicts at the level of the score itself: a wider
pool and a relevance-optimised selection help the near-miss class as much as the real one.*

**P4.** The graded axis survives the config change: per-question monotone (non-increasing) across
rungs ≥ 0.70, as in v2 (0.865) and v3 (0.865). *If the gradient is an artefact of the narrow
default pool, it should degrade here.*

## 4. What a PASS of P1 would and would not mean

A PASS of P1 is a **null**: the best free configuration of RE-call does not read the answerability
axis better than its shipped defaults do. That is worth publishing precisely because the
configuration is otherwise the best one measured — the largest retrieval gain recorded in this
project (`FINDINGS.md` §11, hit@5 0.671→0.777) buying **nothing** on this axis is a sharper
statement than another negative result on a mediocre config.

It would NOT mean reranking is useless: §11 and §12a measure it winning on retrieval, decisively.
It would mean retrieval quality and answerability discrimination are separate axes, which is the
claim mem-bench exists to make measurable.

## 5. The deflating reading, registered in advance

If P1, P2, P3 and P4 all pass, the honest summary is: *nothing about this arm is surprising, and it
was predicted from a mechanism already published in §12c.* Registering that now means a clean null
cannot later be dressed up as an insight. The result's value is that the prediction was made before
the measurement and the measurement is on the public board either way.
