# The Answerability Ladder — design

**Date:** 2026-07-29 · **Status:** design approved, not yet implemented
**Kind:** a **public benchmark others run on their own systems** — not another arm of our own
evidence suite.
**Companions:** [`benchmarks/SUITE-DESIGN.md`](../../../benchmarks/SUITE-DESIGN.md) (the seven-track
internal suite; this is a different artifact and does not replace it) ·
[`benchmarks/EXPERIMENT-CONVENTION.md`](../../../benchmarks/EXPERIMENT-CONVENTION.md) ·
[`docs/RESEARCH_PROTOCOL.md`](../../RESEARCH_PROTOCOL.md)

Prior work: `docs_search "BEAM benchmark evaluation results answerability abstention"`,
`source_type=memory` — found `project-recall-beam-benchmark-2026-07-28`,
`project-recall-abstention-bounded-domain-2026-07-24`,
`reference-locomo-judge-audit-accepts-wrong-answers-2026-07-28`,
`project-recall-threshold-embedder-fragile-2026-07-28`. All four are load-bearing here and are
cited inline. `benchmarks/SUITE-DESIGN.md` and `benchmarks/PREREGISTRATION-currency.md` already
existed on master and were read before this was written; nothing in this file re-proposes them.

---

## 0. Why this exists, and the trap it must avoid

`benchmarks/SUITE-DESIGN.md` already argues that BEAM's aggregate cannot price an abstention
claim: abstention is 10 % of its questions while false-abstention risk applies to the other 90 %,
so every abstention policy tested came out net **negative** on its average (§9i). That argument is
sound and is not restated here.

This file makes a **different and stronger** claim, and it is the reason a new benchmark is
justified rather than a new track:

> Answerability is not a property of a system. It is a function of **how far the question sits
> from what the corpus contains** — and every published abstention number silently fixes that
> distance at one arbitrary value, then reports it as universal.

The evidence that this hidden parameter exists is already measured, on two corpora, by signals
sharing no mathematics:

| regime | dense cosine (unanswerable vs answerable) | lexical coverage | separation |
|---|---|---|---|
| BEAM (adversarial) | **0.676 vs 0.641** — inverted | **0.741 vs 0.717** — inverted | signal points the wrong way |
| ordinary held-out corpus | separates in the right direction | — | AUC **0.78** |

Same system, same metric, opposite conclusions. Two measures with no shared mathematics invert
*identically*, which rules out an artefact of either one. The difference is entirely in **how the
unanswerable questions were built** — BEAM's ask for details never stated about topics discussed
at length; an ordinary held-out probe asks about topics simply absent.

**The trap.** `benchmarks/PREREGISTRATION-currency.md` names it exactly, and its wording binds
this file too:

> Measuring a real capability we happen to be good at is legitimate — it is what BEAM does for
> extraction quality. Measuring an artefact of our own implementation is rigging.

This benchmark is built on an axis where **we already know we collapse** (section 6). That is the
defence, and it is the only one that survives contact with a hostile reader.

## 1. The claim, stated so it can fail

**H1.** Abstention performance varies systematically and monotonically with excision distance.
*Fails if:* the curve is flat. Then the axis is a fiction and the benchmark is worthless — this is
the primary kill condition and it is checked before any comparative result is reported.

**H2.** The axis reconstructs the published literature: BEAM's regime lands at the near end, the
far-gap probes at the far end.
*Fails if:* the known numbers do not land where predicted. Then the axis does not explain the
disagreement it was built to explain, and the unifying claim goes.

**H3.** The curve's *shape* is a property of answerability, not of the ring-construction function.
*Fails if:* rebuilding rings with a different neighbour function changes the shape (section 5.5).

Predictions for all three are committed in the pre-registration **before the builder runs**, per
`SUITE-DESIGN.md` rule 2 and the standing rule to predict before measuring.

## 2. Architecture — four units, four boundaries

| Unit | Does | Depends on | Released |
|---|---|---|---|
| `builder` | source corpora → frozen manifest | source corpora, standalone BM25 | script (run-once by us) |
| `manifest` | instance ids, labels, excision doc-id lists, hashes | nothing | **yes — this IS the benchmark** |
| `adapter` | the `MemorySystem` protocol third parties implement | nothing | yes, + 2 reference impls |
| `scorer` | manifest + responses → curve, 2×2 per ring, λ-pricing | nothing | yes |

The **manifest is the product**; everything else is replaceable plumbing. A third party who
distrusts our builder can read the manifest, verify the hashes, and never run our code.

**Redistribution is solved by shipping the manifest, not the corpora** — question ids, frozen
excision doc-id lists, and corpus hashes, plus a builder that fetches sources itself. This is what
BEIR does and it sidesteps every licence question. No corpus text is redistributed.

### 2.1 The adapter boundary

```python
class MemorySystem(Protocol):
    def ingest(self, docs: Iterable[Document]) -> None: ...
    def query(self, question: str) -> Response: ...
```

`Response` carries: `answer: str | None` (`None` **is** the abstention), `cited_ids: list[str]`,
and measured `tokens`. A third party implements two methods. Reference adapters ship for RE-call
and Mem0 so the interface is proven against two genuinely different architectures before release —
an interface with one implementation is a class, not a protocol.

## 3. The excision ladder

For a question `q` with gold evidence set `G`:

- **d = 0** — excise exactly `G`. The topic remains wholly intact; that one fact is absent.
  *This is BEAM's regime, constructed mechanically.*
- **d = 1…n** — excise `G` ∪ a widening ring of `G`'s BM25 neighbours.
- **d = max** — excise the whole topic cluster (session / thread boundary where the corpus
  declares one, otherwise BM25 to saturation). *This is the far-gap regime.*

**Ring widths, the number of rings, and the saturation rule are fixed in the pre-registration, not
here.** They are free parameters, and a free parameter chosen after seeing a curve is how a curve
gets the shape its author wanted. This file commits to fixing them before the builder runs; it
deliberately does not fix them now, because the corpora have not been inspected yet and choosing
while looking at a half-built corpus is still post-hoc.

**The x-axis is a count** of excised documents and tokens, not a similarity score. BM25 only
*orders* what to remove; the resulting doc-id lists are then **frozen into the manifest**, so every
lab excises identically regardless of what embedder it runs. This is what makes the axis
non-circular: a system under test never computes its own distances.

**Pairing.** Every unanswerable instance is paired to its own answerable original — same question,
same corpus, same generator, differing only in what was excised. This is the design that defended
the Mem0 head-to-head (identical generator, identical judge, identical questions), and it
differences out shared annotation error, which matters directly given section 8.1.

### 3.1 Build item: BM25 must be decoupled from the store

`recall/eval/bm25.py` is the right implementation — dependency-free on purpose, a 40-line
Robertson variant written out in its own docstring specifically so a baseline cannot silently
change between releases of a third-party package. But it imports `recall.store.PgVectorStore` and
scores over indexed chunks. The builder needs BM25 over **raw documents with no database**. This is
real work, not a detail, and it must not fork the formula: one scoring function, two callers.

## 4. Scoring

**Abstention is scored mechanically.** The label is ground truth *by construction* — no judge
decides whether the system should have abstained. This arm therefore costs **$0 in API spend**,
which is not an aesthetic preference: OpenRouter credits are currently exhausted (the BEAM
best-config arm died at 5/60 on a `402`).

- **The full 2×2 per ring** — correct-abstain, false-abstain, correct-answer, false-answer. Never
  collapsed to a single accuracy.
- **λ-pricing** over λ = cost(false answer) / cost(false abstention), λ ∈ {1, 3, 10}, **fixed in
  the pre-registration**. λ = 1 reproduces BEAM's implicit weighting. Choosing λ after seeing
  results is forbidden by this file.
- **No headline scalar.** Deliberate, and it has a cost: a curve is harder to adopt than a number,
  and adoption is the point of a public benchmark. It is accepted because a derived scalar is
  exactly how BEAM's aggregate came to hide the abstention pricing problem — the number gets cited
  and the diagnostic does not.

### 4.1 Answer quality is a separate, later layer — and it is nugget-based

Scoring whether an *answered* question was answered correctly needs a judge, costs money, and is
**out of v1**. When it lands it is **nugget-based** rather than a single verdict, because the
scorer of a benchmark built to fix measurement cannot itself be a `gpt-4o-mini` judge that accepts
**62.81 %** of intentionally wrong, topically adjacent answers (Penfield Labs audit; scripts
public). That finding has **not** yet been reproduced on our own dumps, and doing so is a
prerequisite for this layer, not an optional extra.

## 5. Invariants asserted on the artefact

Exit code 0 is not a measurement. The standing rule is to verify the **artefact**, not the
process — predicting an outcome cannot notice a corrupted apparatus. Each run asserts:

1. **Excised documents are absent from the ingested slice**, counted post-ingest, per system. A
   system that caches across rings would otherwise pass every level silently — and would look like
   a strong result rather than a broken harness.
2. **d = 0 instances retain surviving topical neighbours.** If a question's topic is only one
   document deep, d = 0 is secretly d = max and the instance is dropped, not scored.
3. **Answerable originals are actually answered by ≥ 1 system.** A question no system answers is
   broken, not hard, and cannot anchor a pair.
4. **Manifest hash matches at eval time**, and the builder is deterministic: two builds from the
   same source hashes produce byte-identical manifests. Tested, not assumed.
5. **Ring-robustness arm (H3).** Rebuild rings with a different neighbour function
   (random-within-topic). If the curve shape changes, BM25 is reported as a confound and the
   curve does **not** ship as an answerability result.

## 6. How we lose it

Published in the same document as any win, per `SUITE-DESIGN.md`'s writing rule.

- **RE-call false-abstains at 0.481 while retrieval hit@5 is 0.970** — it refuses half the
  questions it had just answered correctly (LongMemEval, per-question).
- On BEAM's abstention category we score **0.467 against Mem0's 0.533**, false-abstaining at
  **9.6 % against their 4.1 %**.
- **Six candidate signals have already been measured and all failed** (AUC): dense cosine 0.753
  (what ships), cross-encoder rerank 0.742, RRF fusion 0.739, QNLI entailment 0.648, margin 0.579,
  ratio 0.545. The shipped threshold is at its ROC ceiling; retuning it is a closed question and
  this benchmark is not an invitation to reopen it.

**We expect to lose the near end of our own benchmark.** The benchmark's value is that it
*localises* where every system fails — not that we win it.

The one finding that runs the other way is Mem0's, and it is why the axis is worth measuring at
all: scoring **their** published BEAM answers with our judge, they fabricate an answer on **46 %
of unanswerable questions** (32 of 70; abstained → mean 0.974, answered → 0.016). Their published
aggregate is honest — our independent re-judge reproduced their BEAM 1M cell to **0.0005** — so
this is a real behaviour at a real number, not a scoring dispute.

## 7. v1 scope

| | |
|---|---|
| **Anchor corpora** | LOCOMO (`evidence` turn ids) + LongMemEval (`answer_session_ids`) — memory-shaped, and the corpora Mem0 and Zep publish on, so results are directly comparable |
| **Generality arm** | 3 BEIR corpora via `recall/eval/beir.py` + qrels — shows the curve is not an artefact of chat data. Selected only from those a third party can fetch without registration (see 8.3); the three are named in the pre-registration |
| **Measured** | abstention curve only |
| **Adapters** | RE-call, Mem0 |
| **Out of scope** | answer-quality judging (section 4.1), currency, attribution, tenant isolation, multi-hop reasoning, 10M-token scale |

Sequencing: pre-registration → builder + manifest (with section 5 invariants and H1 flat-curve check)
→ RE-call arm ($0) → Mem0 arm (costs money) → write-up.

**H1 gates everything downstream.** If the curve is flat on the RE-call arm alone, the benchmark
is dead and no money is spent on the Mem0 arm.

## 8. Honest risks

**8.1 LOCOMO's evidence labels are less exposed than its answers, not clean.** The Penfield audit
found 99 score-corrupting errors in 1 540 questions (6.4 %) in the **answer key**, and
`recall/eval/locomo.py:50-51` notes retrieval scoring reads `evidence`, not `answer` — but also
that "the two are annotated by the same pass, so treat the evidence labels as carrying" the same
risk. Using evidence for excision is *safer*, not *safe*. Mitigated by pairing (section 3) plus a sampled
manual check that gold evidence genuinely contains the answer; **not** by assertion.

**8.2 The Mem0 arm costs money.** Their ingest pipeline is LLM-based. The RE-call arm is free;
the head-to-head is not, and credits are currently exhausted.

**8.3 BEIR licences vary per dataset.** Manifest-only release sidesteps redistribution, but the
builder must still fetch, and some datasets gate that. Datasets that cannot be fetched by a third
party without registration are excluded from v1 rather than shipped as a footnote.

**8.4 A curve is harder to adopt than a number** (section 4). Accepted deliberately.

**8.5 Excision is not the same as absence.** A fact removed from a corpus may still be inferable
from what remains — especially at d = 0, where the whole topic survives. This is a *feature* at
the near end (it is exactly BEAM's regime) but it means "unanswerable" strictly denotes "not
stated", not "not derivable". The write-up must say so; a system that reasons its way to a correct
answer at d = 0 is penalised by this benchmark and that is a known limitation, not a defect to
hide.

**8.6 This is not a replacement for BEAM.** BEAM measures extraction quality, multi-hop reasoning
and scale, and measures them well. Our BEAM loss (0.594 against Mem0's 0.650) is published
alongside these results rather than instead of them, per `SUITE-DESIGN.md` track G.

## 9. What this benchmark does NOT measure

Extraction quality, summarisation, multi-hop reasoning, currency/supersession, attribution,
tenant isolation, cost, latency, and anything at 10M-token scale. Several of those are tracks in
`benchmarks/SUITE-DESIGN.md` and remain there.

## Status note, 2026-07-29 (post-publication audit) — v1 scope was narrowed after this spec

Section 7 above lists LongMemEval, a 3-corpus BEIR generality arm, and a Mem0 adapter as v1 scope.
None shipped in v1. The narrowing was deliberate and is explained, with a reason for each deferred
item, in the companion implementation plan's
[`## Out of scope for this plan (plan 2)`](../plans/2026-07-29-answerability-ladder-v1.md#out-of-scope-for-this-plan-plan-2)
section — this spec did not reference it, which is what this note fixes.
