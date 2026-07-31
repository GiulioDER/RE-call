# Pre-registration — are mem-bench axes 2 and 3 sensitive to retrieval configuration?

**Date:** 2026-07-31 · **Status:** written and committed BEFORE either arm runs
**Manifests:** mem-bench `manifests/isolation/manifest_v1.jsonl` (digest `b3615583…`) and
`manifests/temporal/manifest_v1.jsonl` (digest `7fa9cab4…`) — the same frozen manifests the shipped
default-config submissions cite, so these are single-variable comparisons against rows already on
the board.

## 1. This is a question about the AXES, not about the score

RE-call's default-config results are already published: isolation `leak_rate 0.0000` /
`own_tenant_recall 0.8417`, temporal `covering 0.2417` / `contradiction 0.7500`. Nothing here is an
attempt to improve them, and the default arm remains the headline either way — a tuned variant is a
separately labelled arm and never the headline
([[reference-recall-best-configuration-2026-07-28]] and the ladder adapter's own docstring).

The question worth answering is a property of the instrument:

> **Does a figure on these axes move when retrieval configuration changes?**

Axis 1's central weakness is that it barely moves between *systems* — four configurations within
0.016 at the near rung. If axis 2's leak rate turns out to be **invariant to configuration while
still separating filtering strategies 0.00 vs 1.00**, that is the opposite and much better
property: the axis measures architecture rather than tuning. Asserting that is cheap; measuring it
is a two-minute run on a 400-document corpus.

**Arm:** `voyage-4-large` + `voyage:rerank-2.5` + `k=45`, `candidate_k=250` — RE-call's published
best configuration, the same one ladder v5 is running on axis 1. Paid, and trivially so at this
size: 400 and 240 documents, 120 queries each.

## 2. Prior work — searched, and it changed a prediction

`docs_search(source_type="memory", query="benchmark axis sensitivity to retrieval configuration
invariant architecture versus tuning")`, run before this file was committed. **`gap_warning: true`**
(top-3 cosine 0.548 / 0.541 / 0.538): nobody has asked this question here, which is unsurprising for
axes one and two days old. Two hits are load-bearing anyway:

- **[[project-recall-v050-auth-and-retrieval-2026-07-22]] — "corpus vocabulary dominates,
  rerank/pool/chunking all ~0."** Measured on 110 hand-labelled questions per corpus. This
  **weakens P2 as originally drafted**: the isolation corpus is 400 short template-generated
  documents, so the reranker and the wider pool should contribute close to nothing. What should
  still move is the **embedder** — [[reference-recall-best-configuration-2026-07-28]] records
  `voyage-4-large` at **+0.282 hit@5** on the private memory corpus (0.348 → 0.630) and 16 of 17
  held-out BEIR corpora. P2 is therefore a prediction about the embedder specifically, and if
  `own_tenant_recall` does not move, the honest reading is that this corpus is too easy to
  discriminate retrieval quality at all — not that the axis is invariant.
- **[[project-recall-finance-market-nogo-2026-07-25]] — "NO public benchmark exists for
  point-in-time / as-of correctness in document retrieval."** Only Look-Ahead-Bench, which measures
  *parametric* look-ahead bias with retrieval pinned off. Recorded here because it is the strongest
  external justification mem-bench's axis 3 has: **the axis fills a documented instrument gap**, and
  that is worth stating on the board rather than discovering later.

The two facts already established about these axes, both measured rather than argued:

- **Axis 2's leak floor is architectural.** `PgVectorStore` binds one tenant per store, sets a
  per-connection GUC read by a Postgres RLS policy, and carries `tenant_id` in the primary key.
  Retrieval configuration operates *inside* the candidate set RLS has already restricted.
- **Axis 3's failure is a missing mechanism, not a weak one.** RE-call returns the value and its
  revision together on 75 % of revised instances. Nothing in `trusted_search` prefers the assertion
  whose interval covers the reference time; `now` gates the trust verdict through frontmatter
  windows this corpus does not carry.

## 3. Predictions

**P1 (axis 2, and it is a claim about RLS).** `leak_rate` stays **exactly 0.0000** on every tier.
*If a single leak appears, RLS is not doing what the source says it does, and that finding is worth
more than the arm* — it would mean a foreign document became a candidate, which the architecture
forbids.

**P2 (axis 2) — revised after the prior-work search.** `own_tenant_recall` **rises above
0.8417**, and if it does, the cause is the **embedder**, not the reranker or the pool: prior work
measured rerank/pool/chunking at ~0 on a corpus of this shape, and `voyage-4-large` at +0.282 hit@5.
*If it does not move at all, the honest reading is that a 400-document template corpus is too easy
to discriminate retrieval quality — a limitation of our corpus, not evidence that the axis is
invariant.* That distinction is registered now because it is exactly the one that would be
convenient to blur afterwards.

*P1 and P2 together are the axis-property claim: the security figure is invariant, the utility
figure is not. If BOTH move, the axis is measuring tuning. If NEITHER moves, the arm was inert and
proves nothing.*

**P3 (axis 3 — KILL CONDITION on my own published write-up).**
`covering_selection_rate` does **not** rise above **0.35** (from 0.2417).
*The submission at mem-bench#25 states the failure is architectural. If the best available retrieval
configuration lifts it materially, that claim is wrong: the failure would be retrieval-limited, the
write-up needs correcting, and the correction matters more than the number.*

**P4 (axis 3, deliberately counterintuitive).** `contradiction_rate` **rises above 0.7500**.
*Better retrieval finds BOTH the value and its revision more reliably, and serving both is what this
figure counts. A system getting better at retrieval should get worse at this — if that holds, it is
the cleanest demonstration available that retrieval quality and temporal correctness are separate
axes.*

## 4. What a clean result looks like

P1 ✅ P2 ✅ P3 ✅ P4 ✅ would say: **axis 2's security figure and axis 3's correctness figure are
both invariant to retrieval configuration, while the utility figures move.** That is exactly what a
benchmark measuring architecture rather than tuning should look like, and it is the strongest
available answer to "did you just not tune it?".

## 5. The deflating reading, registered in advance

If every prediction holds, this arm changes no headline number and teaches nothing about RE-call.
Its entire value is a property of mem-bench, demonstrated rather than asserted. Registering that
now means a null cannot later be narrated as an achievement.
