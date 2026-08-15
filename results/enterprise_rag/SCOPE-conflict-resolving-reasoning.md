# Scope: what would it take for reasoning to win the rows it is supposed to win?

Written 2026-08-15. **A scope, not a result and not a pre-registration.** Nothing here has been
measured. The target is the 11 rows that
[`FINDING-where-the-deficit-actually-is.md`](FINDING-where-the-deficit-actually-is.md) identifies
as clean answer-control failures: gold documents fully retrieved, answer still wrong, in the two
categories a reasoning layer is supposed to take.

`probe_reasoning_reach` already established that the SHIPPED mechanism cannot help: it detects
contradictions by matching a nine-word status vocabulary between claims sharing a subject, and its
response when it fires is `_fail_closed("ambiguous_evidence")`, an abstention. So this is new work,
not wiring. What follows is what the 11 rows actually need.

## The rows are not one problem. They are three.

I read all 11 questions and their `answer_facts`. The categories the benchmark assigns are not the
mechanisms the rows require.

### A. Supersession-ordered synthesis, 4 rows

`qst_0418`, `qst_0419`, `qst_0420`, `qst_0425`. Every one is an **old value against a new value**,
and the required answer asserts the new one while explicitly labelling the old:

| row | the conflict |
|---|---|
| `qst_0418` | v2 tier thresholds against "the previous thresholds" |
| `qst_0419` | the updated start endpoint against a legacy alias kept behind a flag |
| `qst_0420` | an updated GiB-based egress model against the token-based one it replaced |
| `qst_0425` | v1 `integrity` against the embedded `signature` field it no longer uses |

🔑 **This is RE-call's own thesis, pointed at answering instead of retrieval.** Supersession is
what this library exists to reason about.

⚠️ **And the obvious move is wrong.** The trust layer DEMOTES superseded content, which would hide
the old value. The gold answers require it mentioned, labelled: `qst_0419`'s facts say the answer
*may* mention the legacy alias and *must not* present it as current. So the requirement is ordered
synthesis, "new asserted, old retained and marked", not filtering. A layer that simply dropped the
superseded chunk would fail these rows in the other direction.

### B. Multi-part coverage within one document, 6 rows

All six `intra_document_reasoning` rows have **one gold document** and a compound question:
"who attended AND which two checkpoints with dates", "what term discount AND what onboarding
credit", "which items pushed to v1 VERSUS kept in MVP". Each carries 2 to 7 `answer_facts` that
must all appear.

**These are not reasoning failures and no conflict logic touches them.** They are coverage
failures on a single retrieved document. This is the one place the withdrawn "answers are too
short" thesis might survive, because here the requirement is a conjunction of facts under
CORRECTNESS rather than under the completeness rubric that turned out to be judge-dependent.

### C. Attribution and authority, 1 row

`qst_0413`: two people disagree in a ticket, one is the authority, and the other's request was
never confirmed. The answer must attribute each position and must not claim an approval that is
not recorded. Neither A nor B addresses it. **One row. I would not build for it.**

## Step 0, and it is the whole risk

**Before building anything: does the supersession signal exist in the retrieved chunks?** This is
the same question `probe_reasoning_reach` just answered for the contradiction detector, and
answering it there cost nothing and saved a paid run.

The four A rows need a recency or version signal the layer can order on. Candidates, in the order
they cost nothing to check:

1. **In-text version markers.** "v2", "v1", "updated", "legacy", "no longer", "previously",
   "replaces". `qst_0419` and `qst_0425` visibly carry them in the question itself.
2. **Document metadata.** These are `.txt` files with no frontmatter, so `valid_from` is absent and
   `indexed_at` is ingest time, identical for every chunk. **Assume this signal does not exist**
   until measured; it is why the shipped detector's window overlap check is vacuous here.
3. **Chunk order within a document**, where the newer statement follows the older.

⛔ **If only signal 1 exists, then "reasoning" here is text pattern recognition over retrieved
chunks, and it belongs in the answer layer rather than in the graph or trust layers.** That is a
real architectural finding and it should be established before a line of code is written, because
it decides which module the work lands in.

## The measurement instrument, and why it needs no judge

`answer_facts` are explicit and mostly carry hard anchors: `POST /v1/capacity/migrations/start`,
`+$0.085`, `sha256-only`, `Score ≥85`, `20%`, `150 ms`, `72 hours`. So **fact-token hit rate** is
computable mechanically:

- hand-extract anchor tokens per fact, roughly 40 across 11 rows, committed as a fixture;
- score an answer by which anchors it contains;
- negative facts ("must not invent `/v1/capacity/migration/start`") are absence checks, which are
  the cheapest and sharpest of all.

⚠️ **What this instrument cannot do**, stated before it produces a number:

- It measures **token presence, not correctness.** A fact stated in different words scores as a
  miss, so the absolute rate is a LOWER BOUND and only the DELTA between arms is meaningful.
- It cannot score "must not present the legacy endpoint as primary", which is about framing rather
  than presence. That fact needs a human read, and at n=11 a human read is affordable.
- With 11 rows, no interval will separate anything. **This is a mechanism probe, not a measurement
  of effect size**, and it must be reported as one.

## Cost

| item | cost |
|---|---|
| step 0, does the signal exist | **zero.** Offline, over chunks already retrievable |
| the fact-anchor fixture | no API spend, roughly 40 hand-extracted anchors |
| the A/B run | 11 rows × 2 arms × 1 call on a cheap model. **Pennies** |
| judge | **none.** Mechanical scoring plus a human read of 11 answers |

The expensive thing is not the run, it is the retrieval substrate: these rows need the
`voyage-4-large` index, which lives on VPS2. Re-retrieving 11 questions there is cheap; the
alternative is to freeze the 11 retrieved evidence bundles ONCE into a fixture and iterate offline
against it, which I would prefer, because it also makes the probe reproducible by anyone with the
fixture and removes the index from the loop entirely.

## What I would build, if step 0 says the signal is there

Smallest thing that could work, in the answer layer, not the graph:

1. A **recency-ordering pass** over the evidence items, keyed on in-text version markers, that
   annotates each item as `current` or `superseded-by-a-sibling` with the marker that decided it.
2. Those annotations travel **inside the delimited data region** of the user message, as library
   authored field names on library authored values. No corpus byte reaches the instruction channel
   and `SYSTEM_PROMPT` does not move, which keeps the frozen evidence contract intact.
3. The answer is asked to assert the current value and label the superseded one. That instruction
   is a **selection among library authored literals**, per the contract the parity pre-registration
   already sets out.

**Mechanism B, the six coverage rows, is a separate change and should not be bundled.** Its fix is
about answering a compound question completely, and mixing it into the same diff would confound
two effects, which is the mistake the parity work exists to prevent.

## What I would not do

- **Not touch the trust layer.** It demotes, and these rows need retention with a label.
- **Not widen the nine-word vocabulary.** It would make the shipped detector fire more, and firing
  more means abstaining more, which cannot win a correctness row.
- **Not build for `qst_0413`.** One row, a third mechanism.
- **Not run the benchmark.** Eleven rows, mechanically scored, no judge. If the mechanism works
  here it earns a real run; if it does not, nothing was spent finding out.

## Before any of it runs

Per `docs/RESEARCH_PROTOCOL.md` and the standing instruction, this needs its own pre-registration
committed BEFORE the first measurement: the predicted fact-hit delta on the four A rows, the
prediction that the six B rows do NOT move (they are a different mechanism, and if they move, the
probe is measuring something other than what it thinks), and the invariant that no arm may reach
the instruction channel with a corpus byte.
