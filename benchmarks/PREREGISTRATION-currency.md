# Pre-registration — the currency benchmark (RE-call vs Mem0)

Written **before** the corpus is built or any arm is run. As with
[PREREGISTRATION.md](PREREGISTRATION.md), the git history of this file is the evidence.

## 0. Why a new benchmark, and the trap it must avoid

BEAM cannot price this library's central claim, and we can now say exactly why rather than
complain about it: abstention is 10 % of its questions while false-abstention risk applies to the
other 90 %, so every abstention policy we tested came out net **negative** on its aggregate
(§9i). A system that declines to invent cannot win that average by construction.

That is a reason to measure the claim elsewhere. It is **not** a licence to build a benchmark we
win. The line this file commits to:

> Measuring a real capability we happen to be good at is legitimate — it is what BEAM does for
> extraction quality. Measuring an artefact of our own implementation is rigging.

Concretely: every question is scored on the **outcome** (is the served fact the current one?),
never on the mechanism (did the system read a `supersedes:` edge?). Mem0 cannot use our
annotations and must not be scored for failing to.

## 1. The claim, stated so it can fail

> A memory layer should serve the **current** fact, know when it has **no** fact, and do both
> without an LLM in the retrieval path.

Three measurable sub-claims, each falsifiable:

- **C1 Currency.** When a fact has been superseded, the system serves the current version and not
  the stale one.
- **C2 Abstention.** When the corpus never contained the answer, the system says so rather than
  constructing one.
- **C3 Cost.** Total tokens — ingest **plus** query — as a function of query volume.

## 2. Corpus: RFCs, because we did not make them

The corpus is the public IETF RFC series. It is chosen for one property no synthetic corpus has:
**real, declared, machine-readable supersession**. Every RFC carries `Obsoletes:` and `Updates:`
headers, written by their authors years before this benchmark existed, for their own reasons.

This matters because the alternative — generating a corpus with supersession we designed — would
let us shape the exact distribution our library handles best. We cannot shape the RFC series.

- **Ground truth for C1** is the obsoletes chain: for a given topic at a given date, the current
  document is a fact of the record, not a judgement call.
- **Ground truth for C2** comes from topics absent from the corpus slice, held out mechanically.
- **Licence**: RFCs are freely redistributable (BCP 78 / RFC 5378). The exact slice, its hashes,
  and the build script ship with the results.

### Two conditions, both published

| condition | corpus | what it tests |
|---|---|---|
| **ANNOTATED** | RFCs with their `Obsoletes:`/`Updates:` headers intact | can a system exploit declared structure when it exists |
| **RAW** | the same documents, headers stripped, ingested as an undated stream | can a system infer currency without declared structure |

**We expect to win ANNOTATED and to lose or tie RAW.** Publishing only the first would be the
rigging this file exists to prevent. The **difference between the two conditions is the actual
finding** — it prices what declared structure is worth, which is a question a user choosing
between these tools genuinely has and which neither vendor has answered.

## 3. Metrics, fixed now

**C1** — accuracy on "what is the current specification for X?", plus the **stale-service rate**:
how often the system serves a superseded document as if current. The second is reported
separately because it is the failure a user actually pays for, and an accuracy number hides it.

**C2** — the full 2×2: correct-abstain, false-abstain, correct-answer, false-answer. Reported as
a matrix, never as a single accuracy.

Because a false answer and a false abstention are not equally bad, and no single ratio is right
for everyone, results are reported as a curve over **λ = cost(false answer) / cost(false
abstention)** for λ ∈ {1, 3, 10}. λ = 1 reproduces BEAM's implicit weighting; λ = 10 reflects a
domain where inventing is expensive. **Fixing λ after seeing the results is forbidden by this
file.**

**C3** — ingest tokens and query tokens measured by the harness meter (not vendor claims), then
the **crossover query volume** at which each system's total cost overtakes the other. This is
arithmetic once both sides are measured, and nobody has published it.

## 4. Both arms, one judge, same input

- Both systems ingest byte-identical corpus slices.
- One judge instance scores both arms, as in the BEAM harness — this removes judge drift from the
  comparison, and it is the design that let us reproduce Mem0's published BEAM number to 0.0005.
- Retrieval budgets are matched on **tokens**, not on memory count. Matching on count is the error
  that put our BEAM arm at 4.5× Mem0's context and made its cost look intrinsic
  (`benchmarks/beam/rank_probe.py`: ~30,000 tokens per question against Mem0's ~6,700).
- Mem0 runs at its shipped defaults; RE-call runs at its shipped defaults. Any tuned variant is a
  separately labelled arm, never the headline.

## 5. Things we already know cut against us

- **On RAW we are likely to lose.** BEAM measured `knowledge_update` at 0.583 for us against
  0.775 for Mem0. Their extraction resolves updates at ingest; our supersession is
  annotation-driven and inert on unannotated text. We expect this to reproduce.
- **If ANNOTATED shows little gain, the thesis is weak**, and this file commits to reporting that
  rather than reframing it.
- **RFCs are not conversation.** A win here does not transfer to chat-shaped memory, and must not
  be described as if it did.
- **`Obsoletes:` is coarse** — a whole document supersedes a whole document, where a memory layer
  usually deals with a fact superseding a fact. This favours us by making the relation easier
  than the general case, and the write-up must say so.
- **Our abstention is currently worse than Mem0's**: 0.467 against 0.536 on BEAM's abstention
  category, and we false-abstain at 9.3 % against their 4.1 %. C2 is not a lap of honour; on
  today's evidence we may lose it.
  <!-- Mem0's cell is 0.536, not the 0.533 previously written here and in SUITE-DESIGN.md: it is
  (38 x 0.974 + 32 x 0.016) / 70 from FINDINGS.md §9h's own n=70 table. Both documents are now
  stated to three places from that arithmetic rather than from each other. -->

  **Not yet reconciled:** our own 0.467 is asserted in both planning documents and is not derivable
  from any committed artifact — unlike Mem0's cell, which is. It needs an artifact citation or a
  re-measurement before it appears in anything published.
- **The k=45 result that motivates our retrieval budget is not significant** (p=0.754, n=60). It
  is used here as a configuration choice, not as a claim.

## 6. What this benchmark does NOT measure

Extraction quality, summarisation, multi-hop reasoning, conversational memory, and anything at
10M-token scale. BEAM measures several of those and measures them well; this is not a replacement
for it, and our BEAM loss is published alongside these results rather than instead of them.
