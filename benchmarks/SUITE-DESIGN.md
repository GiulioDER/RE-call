# The RE-call evaluation suite — design

Companion to [PREREGISTRATION-currency.md](archive/preregistrations/PREREGISTRATION-currency.md), which pre-registers
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
wins.** The BEAM result — we lose the aggregate, 0.594 against 0.650 — ships alongside this suite,
not instead of it.

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
**How we lose.** On BEAM's abstention category Mem0 scores **0.536**; our own cell is
**citation-pending**, so the single comparison this track exists to make cannot yet be stated in
either direction. What can be stated is the aggregate, and we lose it: **0.594 against 0.650** over
300 questions. On the matched full-run measurement we false-abstain at **3.3 % against their
4.1 %** — not worse — and the paired McNemar with Holm across three families finds **none
significant against us** (accuracy 0.39, false-abstain 1.00, abstention 1.00).

So the honest position going in is that this track is **undetermined**, not lost. It must not be
written up as a loss we bravely disclosed, and it must not be written up as a win: the abstention
cell that would settle it does not exist yet. What is structurally true in advance is that λ=1
reproduces BEAM's own weighting, under which a system that declines to invent cannot win the mean
by construction (10 % of questions reward abstention, 90 % punish false abstention) — so a λ=1
deficit would be a property of the weighting as much as of the system, and this track may not
present it as the latter.

**What counts as losing it**, fixed now rather than after the numbers land: our abstention cell
below Mem0's 0.536 on a held-out corpus at λ=1, or a false-abstain rate above theirs at any λ in
{1, 3, 10}. Both are measurable the moment our own cell has an artifact behind it. Neither is
measured today, and "undetermined" is not a result this track may ship — it is the reason the
track is not yet runnable.

> **Corrected 2026-07-29 — a mismatched pair and a stale test.** This passage previously asserted
> "a 9.3 % false-abstain rate against their 4.1 %, and the paired test on 300 questions puts
> false-abstain against us at p_holm = 0.026". Both halves were wrong, and both in the direction
> that flattered the disclosure.
>
> **9.3 % and 4.1 % are not the same measurement.** 9.3 % is the shipped policy's false-abstain
> rate in §9i's entailment sweep (30 unanswerable / 270 answerable, conversations 0-14); 4.1 % is
> Mem0's rate in the full 300-question head-to-head, where our comparable figure is **3.3 %**. The
> matched pair runs the *other way* than the sentence claimed.
>
> **p_holm = 0.026 was the pre-calibration-fix state.** After the fix, no family is significant
> against us. Quoting it as the current false-abstain result stated a defeat that the run does not
> show.
>
> An earlier pass corrected 9.6 % → 9.3 % in this same sentence (CHANGELOG, *"published a loss as
> a tie"*). That made the digit internally right and left the comparison mismatched — **the defect
> was never the digit.** A number can be verified against its own source and still be the wrong
> number to put on that side of the word "against".
>
> Unchanged from the earlier correction: Mem0's cell is **0.536**, derived from FINDINGS §9h's own
> n=70 table as (38 × 0.974 + 32 × 0.016) / 70. Our own cell is quoted as 0.467 <!--@ citation-pending: no committed artifact retains this cell; re-derive or retract --> in
> `archive/preregistrations/PREREGISTRATION-currency.md` but is **not derivable from any committed artifact**, so it stays
> citation-pending here rather than propagating into a second planning document.

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
- **The BEAM arm violates this suite's own Rule 5.** Every BEAM figure quoted in this document —
  the head-to-head cells (0.594/0.650, 66.7/71.0 %, 3.3/4.1 %) and the three Holm-adjusted
  p-values — is reported in `results/RESULTS.md` on branch `bench/beam-1m` and regenerates from
  the commands in FINDINGS §9e, but `benchmarks/results/` is gitignored, so the per-question dumps
  behind those cells are committed **nowhere**. That is reproduce-command tier, not
  retained-artifact tier, and Rule 5 requires per-question artifacts. Either the dumps ship or the
  rule is amended in the open — the one thing the suite cannot do is hold other people's
  benchmarks to a rule its own cross-check track fails. (Reproducing them is a paid run, ~$45.)
