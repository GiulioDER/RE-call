# Pre-registration — the Answerability Ladder, v3: embedder generality

**Date:** 2026-07-29 · Written **before** the v3 arm runs. The git history of this file is the
evidence. Predecessors: [`PREREGISTRATION-ladder.md`](PREREGISTRATION-ladder.md),
[`PREREGISTRATION-ladder-v2.md`](PREREGISTRATION-ladder-v2.md); results in
[`results/ladder/H1_VERDICT.md`](../results/ladder/H1_VERDICT.md) and
[`H1_VERDICT_v2.md`](../results/ladder/H1_VERDICT_v2.md). Neither is edited.

**Prior work searched** — `docs_search(source_type="memory")` on "voyage calibrated abstention
threshold arm ladder rerank best configuration". Load-bearing hits, and both **changed this
design rather than decorating it**:

- [[project-recall-beam-bestconfig-blocked-2026-07-28]]: on `voyage-4-large` the shipped 0.50 floor
  **starves 14/60 (23.3 %)**; calibrated to 0.300 it starves 0/54. So 0.50 is not uniformly inert —
  it is inert on some embedders and over-eager on others.
- [[project-recall-threshold-embedder-fragile-2026-07-28]]: 0.50 sits at the 0th percentile of five
  of six embedder distributions, and is **inert on `bge-small` AND `bge-large`**.

**What that ruled out.** The obvious v3 was "run RE-call at its best config (voyage + rerank)". Two
reasons it is not this file. First, it needs an API credential this machine does not hold, and
credentials are not something to route through a chat. Second, and more important, it would mostly
re-answer a **deployment** question — where should the constant sit — when the **capability**
question is already settled by data in hand: if RE-call had no near-miss ability, the rungs would
be indistinguishable at *every* threshold, and the v2 sweep shows they separate monotonically
(0.000 / 0.055 / 0.105 / 0.265 / 0.430 at t = 0.60, false-abstain 0.000).

## 0. The claim this file tests

v2 found that top-1 cosine declines monotonically with excision distance (−0.0397 / −0.0539 /
−0.0837 / −0.1100, all CIs excluding zero, non-increasing per question 173/200). **That was measured
on exactly one embedder.** `H1_VERDICT_v2.md` §5 says so in its own Limits section: *"the gradient
is a property of that pairing until another system is run through the same manifest."*

> **Is the graded answerability axis a property of retrieval, or an artefact of `bge-small`?**

This is the generality check the v2 verdict named and did not run. It costs **$0** and needs no
API key.

## 1. Design

**The manifest is reused UNCHANGED** — `results/ladder/manifest_v2.jsonl`, digest
`5534c61356acaa7b62ac5a79dbec7383674fc052984d10c1d0cc89e26a532bd5`, 1 200 instances, 200 questions
× 6 rungs. Same frozen excision sets, same questions, same distractors. Only the **system** changes.
That is the benchmark's design working as intended: a manifest is system-agnostic, so two arms are
**paired at the instance level** and directly comparable.

| | v2 arm (published) | v3 arm (this file) |
|---|---|---|
| embedder | `BAAI/bge-small-en-v1.5` (384-dim) | **`thenlper/gte-base` (768-dim)** |
| family | BGE | **GTE — a different family, not a bigger BGE** |
| everything else | shipped defaults | shipped defaults |

`gte-base` is chosen because a *different architecture family* is a stronger generality test than a
larger model of the same family, and because it runs locally at no cost. It is **not** claimed to be
better than `bge-small`; this arm is not a quality comparison and must not be reported as one.

## 2. Fixed parameters

- **Manifest:** `results/ladder/manifest_v2.jsonl`, unchanged, digest as above.
- **Embedder:** `thenlper/gte-base`, shipped defaults otherwise (no reranker, no calibration,
  default `k`/`candidate_k`) — matching the v2 arm knob for knob.
- **Table/tenant:** a dedicated pair (dimensions differ: 768 vs 384), so neither arm's rows can
  reach the other.
- **λ ∈ {1, 3, 10}**, unchanged. Choosing λ after seeing results is forbidden by this file.

## 3. Predictions

**Cosine scales are not comparable across embedders**, so a prediction that v3 reproduces v2's
*absolute* −0.1100 would be a category error. The claims below are about **sign, monotonicity and
ordering**, which are scale-free.

- **P1 — the kill condition.** The paired change in top-1 cosine from `r=0.00` to `r=1.00` is
  **negative with a bootstrap 95 % CI excluding zero**. *If this fails, the v2 gradient was an
  artefact of one embedder and the benchmark's central claim is embedder-specific* — a major
  negative result that would be published as prominently as the original finding.
- **P2 — monotonicity.** The four rung means are non-increasing in distance, as in v2.
- **P3 — per-question consistency.** Non-increasing across all five rungs for **≥ 0.70** of
  questions (v2: 0.865). Set below v2's value deliberately: predicting the same number would be
  predicting noise.
- **P4 — and I expect to be right for an uncomfortable reason.** The shipped 0.50 floor stays
  **inert** on `gte-base`: correct-abstain ≤ 0.02 at every rung, including `r=1.00`. Predicted from
  [[project-recall-threshold-embedder-fragile-2026-07-28]] rather than from hope. If it holds, it is
  a *second independent embedder* on which the shipped constant cannot express the signal the
  system demonstrably has — which is a stronger statement about the constant than v2 alone could
  make.

## 4. What this arm does NOT establish

- **Not a quality comparison.** Nothing here licenses "gte-base is better/worse than bge-small".
  Retrieval quality is not measured; only the *shape* of the distance response is.
- **Two embedders is not "embedder-independent".** A reproduction on one more family raises
  confidence; it does not generalise to unseen models, and specifically says nothing about API
  embedders (`voyage`, `text-embedding-3-*`) where the 0.50 constant behaves differently.
- **Still no judge**, so answer *correctness* remains unmeasured and every accuracy an upper bound.
- **Same corpus, same distractors** — LOCOMO only. A shared-corpus artefact would reproduce here
  rather than being caught.

## 5. Known to cut against us

- If P1 holds, the most likely reading is mundane: *any* dense retriever's top-1 similarity falls
  as you delete the relevant documents. That would make the axis real but **unsurprising**, and the
  interesting content stays where v2 put it — in the gap between a graded signal and a binary
  decision, not in the gradient's existence.
- `gte-base` is 768-dim against `bge-small`'s 384, so it is not a clean single-variable change:
  family, size and dimensionality all move together. The arm cannot attribute a difference to any
  one of them.
