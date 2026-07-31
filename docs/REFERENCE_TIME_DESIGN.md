# Reference time for as-of retrieval: design, and why the obvious version must not ship

Prior work searched: `docs_search(source_type="memory")` on temporal/validity/newest-wins. Two
memos are load-bearing and both changed this design rather than decorating it:
`project-recall-entailment-supersession-phase0-done-2026-07-18` (supersession already shipped) and
`project-recall-finance-market-nogo-2026-07-25`, which already recorded the conclusion this
document re-derives from question data: **"Zep/Graphiti already ships bi-temporal point-in-time.
RE-call has validity time only."**

Status: **design rejected in its obvious form.** No code proposed. The measurement below is the
deliverable.

## The question

`benchmarks/check_temporal_inert.py` showed the temporal layer could not fire on benchmark data.
`recall/eval/locomo.py` now emits `valid_from`, and `benchmarks/check_temporal_live.py` shows the
layer is live end to end: 419/419 chunks carry a window, and an early as-of yields
`not_yet_valid`.

That leaves the question this document exists to answer: **what reference time should a question
be asked with?** Getting it wrong produces confident wrong demotions, which is worse than the
inert state we started from.

## The obvious design

Derive the reference time from a date named in the question, and let the trust layer demote any
turn whose `valid_from` postdates it. LLM-free: a regex and a date parser.

## Why it must not ship

### 1. Most temporal questions carry no date at all

Measured on `benchmarks/audit_data/locomo_errors.json`, the 156-error LOCOMO audit set:

| | count |
|---|---|
| audit errors, total | 156 |
| `TEMPORAL_ERROR` | 26 |
| ...of those, carrying an explicit date in the **question** | **9 of 26** |
| all errors carrying a date in the question | 26 of 156 |

So the mechanism has no input for roughly two thirds of the errors it is meant to fix. "How long
did it take for Jon to open his studio?" names no time at all. That alone caps the upside; it does
not by itself make the design harmful.

### 2. `valid_from` is when a turn was SAID, and questions anchor on when the event HAPPENED

This is the one that kills it, and the corpus states it outright.

> **"What setback did Melanie face in October 2023?"**
> Session 17 is dated **October 13, 2023**. D17:8 is Melanie saying: *"recently I had a setback.
> **Last month** I got hurt and had to take a break from pottery."*

The event is **September**. The turn is **October**. The question anchors on the event.

A filter keyed on `valid_from` compares the question's anchor against the *utterance* date. Ask
that same corpus "what setback did Melanie face in September 2023" and the only evidence that
exists, said in October, is demoted `not_yet_valid`. **The design deletes retrospective testimony,
and retrospective testimony is frequently the only evidence there is.**

People say "last month", "back in June", "when I was a kid". Every one of those is an event time
that precedes its utterance time, and a conversational corpus is made of them.

### 3. The question's own anchor is sometimes wrong

Three of the four dated questions inspected anchor on a year the corpus does not contain:

| question | anchor | gold evidence actually from |
|---|---|---|
| "...workout class ... in December 2023?" | Dec 2023 | session_1, **17 December 2022** |
| "...donate to a homeless shelter in December 2023?" | Dec 2023 | session_2, **22 December 2022** |
| "...big moment with Samantha in October 2023?" | Oct 2023 | session_29, **31 October 2022** |

These happen to fail safe, because a later anchor demotes nothing. But it means the anchor cannot
be trusted as ground truth about when anything occurred, so a design that tightened the window
around it, rather than only bounding it above, would fail unsafely.

## What this says about the boundary

A reader asked whether cheap field extraction generalises past dates, or quietly becomes the
distillation step we were avoiding. The measurement answers it, and the boundary is sharper than
"field by field":

> **As-of retrieval keyed on utterance time is sound exactly where utterance time equals event
> time. In conversation, it usually does not.**

Getting the rest needs **event time** extracted from the turn's own prose, "last month" resolved
against the session date. That is bi-temporal, it is what Zep and Graphiti ship, and it is an
extraction step over natural language, which is the write-time distillation this architecture
exists to avoid. The `$0` claim and event-time extraction are in genuine tension, and this is the
first place in the project where that tension is measured rather than asserted.

## What is worth doing instead

Narrower, and it survives every objection above, because it does not touch the event/utterance
distinction at all.

**Supersession of explicitly revised state.** §9l's first failure row is *"14 days, using the
updated deadline of 15 Apr"* against a gold of 7 days. That is not an event-time problem. Two
turns assert a value for the same field, one revises the other, and the question concerns the
state before the revision. Supersession is already shipped
(`resolve_successor`, `superseded_by`, `recall lint`), and the missing piece is only the edge
between the two turns.

Pre-registered, before any implementation:

- **P1** In a corpus of conversation turns, fewer than 15% of turns assert a value for a field
  that a later turn revises. If it is lower than that, the mechanism cannot move an aggregate
  score no matter how well it works, and this stops here.
- **P2** Restricted to questions whose gold evidence is a superseded-then-revised field, selecting
  by supersession edge beats newest-wins. Newest-wins is already falsified (§9j, §9l), so the
  comparison is against the *un*filtered baseline as well.
- **P3** On all other questions the change is inert, within noise. A supersession edge that moves
  unrelated questions is a bug, not a win.

P1 is a counting exercise over the corpus and needs no retrieval run. It should be measured before
anything is built, because a null there ends the line cheaply.

## What does not change

`valid_from` on ingested turns stays. It is correct on its own terms (a turn was said on its
session date), it is what makes the trust layer able to fire at all, and
`benchmarks/check_temporal_live.py` pins that it does. Nothing in this document argues for
reverting it. What is rejected is *using it as a question's reference time*, which is a different
claim from *recording it*.

## Correction owed to §9l and to Part 4

Unchanged by this document, and still outstanding. §9l says "no retrieval-side change we can
afford replicates it", which is stronger than what was measured. The accurate statement is:
recency is falsified, the temporal layer could not fire, and the successor most people would reach
for is now measured to be unsound on this corpus for a reason that has nothing to do with
affordability.
