# The RE-call evaluation suite — design

Companion to [PREREGISTRATION-currency.md](PREREGISTRATION-currency.md), which pre-registers
Track A. This file specifies the whole suite and the rules that apply to every track.

## What "incontestable" actually means, and what it cannot mean

It cannot mean "a benchmark nobody disputes". Anyone can dispute anything, and a suite built to
be undisputed is a suite built to flatter its author — the exact failure this project spent a day
diagnosing in someone else's numbers.

What is achievable is narrower and stronger: **a third party can re-run every number without us,
and every choice that could move a result was fixed before the result was visible.** That is the
property that let us reproduce Mem0's published BEAM cell to 0.0005 — their harness is public, so
their claim was checkable. Ours must be checkable the same way.

Five rules, binding on every track:

1. **Third-party corpora only.** No corpus we generated. If we shape the data we shape the result.
2. **Pre-registration before the corpus is built**, not before the run. Choices made while looking
   at a half-built corpus are still post-hoc.
3. **One command to reproduce**, from a clean checkout, with pinned dependencies and corpus hashes.
4. **Both systems at shipped defaults.** Any tuned variant is a separately labelled arm and never
   the headline. (Our BEAM arm at k=200 vs k=45 is the cautionary case: matching the competitor on
   memory COUNT rather than TOKENS quietly put us at 4.5x their context.)
5. **Per-question artifacts published**, not summary statistics. A reader must be able to find the
   questions we lost and read our answers.

And one rule about writing it up: **every track publishes its losses in the same document as its
wins.** The BEAM result — we lose, p_holm = 0.026 — ships alongside this suite, not instead of it.

## Why not just use BEAM

BEAM is well built and we keep it in the suite as an external cross-check. But its aggregate
cannot price this library's central claim, and we can now show why rather than assert it:
abstention is 10 % of its questions while false-abstention risk applies to the other 90 %, so
every abstention policy we tested came out net negative on its average (§9i). A system that
declines to invent cannot win that mean by construction. That is a reason to measure the claim
elsewhere — not a reason to dismiss BEAM, and not a licence to build a benchmark we win.

## Tracks

Each track states its claim, its metric, and — non-negotiably — how we lose it.

### A. Currency (pre-registered)

**Claim.** When a fact is superseded, serve the current version, not the stale one.
**Corpus.** IETF RFCs; `Obsoletes:`/`Updates:` headers are real declared supersession written
years before this benchmark existed.
**Metric.** Accuracy on "what is the current specification for X?", plus **stale-service rate**
reported separately — the failure a user actually pays for.
**Conditions.** ANNOTATED (headers intact) and RAW (stripped), both published.
**How we lose.** On RAW we expect to lose: BEAM measured our `knowledge_update` at 0.583 against
Mem0's 0.775, and their extraction resolves updates at ingest where our supersession is inert on
unannotated text. If ANNOTATED shows little gain, the thesis is weak and the write-up says so.

### B. Point-in-time validity

**Claim.** Answer "what was true as of date T?", not merely "what is true now".
**Corpus.** The same RFC series, queried at historical dates the obsoletes chain makes unambiguous.
**Metric.** Accuracy at T, split by whether the correct answer is the current document or a
retired one. The second half is the interesting one: a system biased to recency fails it, which
is precisely the failure mode we rejected in §9j.
**How we lose.** We have `valid_until` but have never tested point-in-time retrieval end to end.
If RE-call cannot beat a plain recency ordering here, the temporal machinery earns nothing.

### C. Abstention, priced properly

**Claim.** Say "I don't know" instead of inventing.
**Corpus.** Questions whose answers are mechanically held out of the ingested slice.
**Metric.** The full 2×2, reported over a **pre-fixed** cost ratio λ = cost(false answer) /
cost(false abstention), λ ∈ {1, 3, 10}. λ=1 reproduces BEAM's implicit weighting. Fixing λ after
seeing results is forbidden.
**How we lose.** Our abstention is currently WORSE than Mem0's — **(ours: citation pending)**
against Mem0's 0.536 on BEAM's category, with a 9.3 % false-abstain rate against their 4.1 %, and
the paired test on 300 questions
puts false-abstain against us at p_holm = 0.026. On today's evidence we lose this track at λ=1 and
need λ≥3 to come out ahead. That must be stated, not discovered by a reader.

> This sentence previously read "0.533 vs 0.533" — two identical numbers under the word WORSE, so
> it disproved itself. Mem0's cell is **0.536**, derived from FINDINGS §9h's own n=70 table:
> (38 × 0.974 + 32 × 0.016) / 70. Our own cell is quoted as 0.467 <!--@ citation-pending: no committed artifact retains this cell; re-derive or retract --> in `PREREGISTRATION-currency.md`
> but is **not derivable from any committed artifact**, so it is left as citation-pending here
> rather than propagated into a second planning document.

### D. Attribution

**Claim.** The cited source actually supports the answer.
**Metric.** For every answered question, does the chunk the system cites contain the supporting
fact? Scored mechanically where the gold value is a literal, by judge where it is not, with both
reported.
**Why it belongs.** This is the one claim where "no LLM in the retrieval path" is an advantage
rather than a cost: we return the source text, not a paraphrase of it, so attribution is
checkable by construction. It is also the claim an auditor or a regulated user cares about most.
**How we lose.** If our cited chunk supports the answer no more often than Mem0's extracted fact
does, the provenance argument is decorative.

### E. Tenant isolation

**Claim.** One tenant's data never reaches another's query.
**Metric.** Pass/fail under adversarial cross-tenant queries, including queries crafted from
tenant A's content and issued as tenant B. Any leak is a failure; there is no partial credit.
**Why it belongs.** It is a security property, binary, and unusually easy to verify — which makes
it the least disputable number in the suite.
**How we lose.** A single leak fails the track outright, ours or theirs.

### F. Cost and latency

**Claim.** Lower total cost of ownership and lower latency.
**Metric.** Ingest tokens, query tokens, ingest seconds, query latency (median/p90/p95 with CIs),
and the **crossover query volume** at which each system's cumulative cost overtakes the other.
**Measured, not quoted** — the harness meter, never vendor figures.
**How we lose.** The token advantage is configuration-dependent: at k=200 we were 4.5x Mem0's
context; only at k=45 do we sit below it, at equal accuracy. If the crossover lands at a query
volume no real deployment reaches, the cost story is theoretical.

### G. External cross-check (BEAM, as-is)

We keep running Mem0's own harness unmodified and publish the result whichever way it falls. It
is the track we currently lose, and keeping it is what makes the other six believable.

## Sequencing

A and F first: A is the thesis, F is already half-measured (latency exists; token accounting needs
the resumed-rows defect fixed — the meter under-counted by ~a third when rows came from `--resume`).
Then D, then C, then B. E can land any time; it is independent and cheap.

## Known defects to fix before any track ships

- **Usage meter under-counts resumed rows.** A `--resume`d question carries no usage into the new
  process, so `usage` understates true cost. Must carry usage forward or exclude those rows from
  the denominator.
- **The date-format fix is a null result** (p=0.727) and stays in on correctness grounds only. No
  track may claim a benefit from it.
- **Retrieval budgets must be matched on tokens, not memory count**, in every track.
