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

---

# P1 result, and whether bi-temporal is worth building

Both measured after the design above was written, against its pre-registered gate.

## P1: revised state is effectively absent from LOCOMO

`benchmarks/check_p1_supersession_density.py`, all 10 conversations:

| | count | rate |
|---|---|---|
| turns | 5,882 | |
| carrying a revision marker ("actually", "changed", "postponed", ...) | 88 | 1.5% |
| ...and also carrying a value (date, number, weekday) | **3** | **0.1%** |

Pre-registered floor was 15%. The measured rate is **0.1%**, and the proxy was designed to
over-count, so the true rate is lower still. Inspecting all three survivors confirms it: *"a little
girl around 8"*, *"it changed my view on helping"*, *"actually taken last Friday"*. **None revises
an earlier assertion.** The real count is zero.

**Verdict: P1 fails by two orders of magnitude. Supersession-based selection cannot move a LOCOMO
aggregate**, because LOCOMO speakers essentially never revise a previously asserted field value.
The line stops here, as pre-registered, and it cost one counting script rather than an
implementation.

### Scope, stated at the width of the data

This measures **LOCOMO**. §9l's failure table is **BEAM**, whose `temporal_reasoning` split is
constructed to test exactly this and is therefore likely far denser in revisions. The BEAM corpus
is not cached locally, so **the null above does not transfer to BEAM and is not claimed to.**
Settling it is the same script pointed at BEAM once that corpus is available.

### What the BEAM cases themselves say about the ceiling — added with issue #167

Density is not the only thing that bounds a supersession selector, and it turned out not to be the
binding one. Enumerating §9l's seven questions into `results/beam_9l_temporal.json` and
classifying the five that were answered, by the mechanism a selector would need:

| mechanism | n |
|---|---|
| instance disambiguation — several similar events, the question names one | 2 |
| **supersession of a revised value** | **1** |
| field VALUE vs the time it was ASSERTED | 1 |
| event time vs utterance time | 1 |

So **supersession reaches one of the five even at perfect accuracy**, on the very split whose
density this section proposes to measure. That does not settle the density question, and it does
not transfer the LOCOMO null; it reprices the measurement. Running `check_p1_supersession_density`
against BEAM is still cheap once the corpus is there, but a *dense* result would no longer license
building the selector on its own — the reachable gain is capped by the case mix, not by the rate.

Two of the four mechanisms are not orderings in time at all, which is the more useful half: the
value/assertion row is two readings of the *same turn*, and the event/utterance row is the exact
confusion this document's own analysis identifies. Neither is reachable by any rule that sorts
instances by date, and the second is the one this document argues cannot be fixed without
event-time extraction.

Narrow by construction: a hand reading of five answers from the run artifact, with the corpus
never consulted. Five items is a list, not a rate.

## Can bi-temporal be implemented in RE-call?

Yes, and the plumbing is the cheap part.

| piece | effort | why |
|---|---|---|
| storage | **none** | chunk metadata is JSONB and the Indexer splices `**meta`; new keys need no migration |
| `Validity` dataclass | trivial | add `event_from` / `event_until` beside `valid_from` / `valid_until` |
| frontmatter | trivial | add the keys to `VALIDITY_KEYS`, reuse `_parse_date` |
| `_verdict` | small | it already takes one reference time; it needs a second, plus which axis a query filters on |
| `trusted_search` | small | `now=` already exists; add `as_of_event=` |

Call it 150 to 250 lines. Nothing about the architecture resists it.

**The expensive part is populating event time**, and that is measurable:

| turns (of 5,882) | | rate |
|---|---|---|
| carrying a relative-time expression resolvable by arithmetic against the session date | 334 | **5.7%** |
| carrying one that is not ("recently", "when I was", "back in") | 264 | 4.5% |
| share of all relative-time turns that arithmetic could resolve | | **55.9%** |

`last week/month/year` (171), `yesterday` (66), `next week/month` (42) are all pure arithmetic
against a date the corpus already gives us. So **event time is extractable for free on about 6% of
turns**, with no model, and the largest unresolvable bucket is `recently/lately` (206), which
carries no arithmetic content for anyone, model or not.

Notably, the design's own killer example is in the **resolvable** half: *"recently I had a setback.
Last month I got hurt"* on a session dated 13 October 2023. `recently` is noise; `last month`
resolves to September by subtraction. Arithmetic bi-temporal gets that case right.

## Does it have real benefits?

Two different answers, and conflating them is how this gets oversold.

**For benchmark scores: probably not, and nothing here argues it would.**

Bi-temporal does not add recall. Today RE-call applies no as-of filter at all, so a retrospective
turn is retrieved on similarity regardless. Event time's role is to stop as-of filtering from
*destroying* those turns. It is a **prerequisite for a feature, not a feature**, and the feature it
unlocks currently has no demonstrated upside: P1 kills supersession at 0.1%, and only 9 of 26
`TEMPORAL_ERROR` questions name a date to filter on in the first place. Building it to move a
LOCOMO number would be building the enabling half of a mechanism whose other half is already
measured to be inert.

**For the library's actual users: yes, and it is the stronger case.**

"What was the plan as of last Tuesday" is a real agent-memory question, and today RE-call cannot
answer it. That is a capability, not a score, and it is what Zep and Graphiti ship. It should be
justified as a product decision with its own success criterion, not smuggled in as a benchmark
optimisation.

### The honest recommendation

Do not build bi-temporal to fix `temporal_reasoning`. The measurements say the path from event time
to that score runs through mechanisms already measured to be absent from the corpus.

Build it if, and only if, as-of retrieval is wanted as a **capability** for agent memory, in which
case the arithmetic-only version covering ~6% of turns is a sound, genuinely `$0` first cut, and
its limits are already quantified above: it will resolve just over half of the relative-time turns
and none of the `recently` bucket.

The `$0` claim survives this either way, which is worth stating because it was in doubt: the free
arithmetic subset is real, and what it cannot reach is mostly what nothing can reach cheaply.

---

# BUILT: bi-temporal as-of retrieval (`known_as_of`)

Built as a **user capability**, on the explicit instruction that user value is the project's
objective and benchmarks are only an instrument for improving it. The success criterion is that an
agent can honestly replay what it knew at a past instant. It is *not* a `temporal_reasoning` fix,
and the analysis above stands: it will not move that score. Those are different claims and only
one of them is being made.

## What turned out to be true

`project-recall-finance-market-nogo-2026-07-25` recorded that "RE-call has validity time only".
That was right about querying and wrong about storage: **`indexed_at` has been a real, indexed
column all along**, populated on every write and reaching every hit as `ScoredChunk.indexed_at`.
Both temporal axes were already stored. Only one could be asked about.

## The two axes

| axis | parameter | column / key | verdict it produces | the question it answers |
|---|---|---|---|---|
| valid time | `now` | `valid_from` / `valid_until` | `expired`, `not_yet_valid` | *when was this true?* |
| transaction time | `known_as_of` | `indexed_at` | `not_yet_known` | *when did we know it?* |

They compose. `trusted_search(..., now=june, known_as_of=tuesday)` asks what we believed on Tuesday
about the state of the world in June.

```python
res = trusted_search(store, emb, "When and where is the launch?", known_as_of=cutoff)
```

Verified end to end against a real store: the same query returns both memos today, and as-of an
earlier instant returns the original as `ok` while marking the later revision `not_yet_known`.

## Deliberate choices

- **Opt-in.** Passing nothing leaves every existing caller byte-identical.
- **Inclusive boundary.** A memory written *at* the instant existed at that instant.
- **A hit with no `indexed_at` stays visible.** Defaulting an unknown write time to "after the
  as-of" would silently empty result sets for any store predating the column.
- **Checked before supersession.** A memory that did not exist yet cannot meaningfully be reported
  as superseded, and its successor is a document the caller cannot see.
- **`not_yet_known` reads differently from `not_yet_valid` in `abstain_reason`.** One means the
  memory had not been written; the other means it had been, and did not apply. Only the first
  exonerates a past decision, so a caller replaying one must be able to tell them apart.

## Known limit, stated rather than discovered later — mechanism built, NOT yet merge-ready

**As first shipped**, `known_as_of` filtered hits by write time and did **not** rewind
supersession: edges carried no timestamp, so an edge added after the as-of instant still applied,
and a memory current at that moment could read as `superseded` by a document the caller could not
see. Point-in-time replay was honest about *which memories existed* and approximate about *which
were current*.

**Closed 2026-08-01.** The prompt came from a reader on the Part 4 thread, arguing that utterance
time is the axis to **order** on rather than to filter on. That reframing makes the missing
timestamp derivable rather than absent: an edge `A -> B` becomes assertable when B is written, and
B's `indexed_at` has been an indexed column since the beginning. So the corpus format did not need
to record anything new.

`PgVectorStore.supersession_all()` returns the edges and their dates from the single scan that
already builds the edge map, `resolve_supersession_candidates` is the pure rule behind it (keyed per **claim**,
`(superseding file, superseded basename)`, because one file can carry several `supersedes` values),
and `resolve_successor` filters **per step**, so a chain `a -> b -> c` whose second edge postdates
the instant resolves to `b`. Replay is now
honest about which memories existed *and* about which were current.

### `first_indexed_at`: the column that made the rewind sound

The first version dated everything from `indexed_at`, which records the LAST write. A bug audit
found what that costs: `replace_sources` re-inserts with `now()` and `Indexer.index_path` skips
files whose content hash is unchanged, so fixing a typo in a superseding memo moved **its** date
forward while its predecessor kept the old one. A replay of an earlier instant then dropped a
long-standing edge and served the superseded memory as `ok`, where the pre-change code correctly
said `superseded`. A wrong answer replacing a right one, in the layer built to prevent that.

The root cause predated edge dating entirely: `known_as_of` on **hits** had it too, so a
re-indexed memory read `not_yet_known` for every instant before its edit. The store claimed it had
never held a document it had held for months.

`first_indexed_at` fixes both. There are THREE writers and each must clamp with `LEAST(..., now())`, or a
stored date in the future (clock skew, a restore) makes the row permanently `not_yet_known`.
`_upsert_in` preserves it on conflict; `replace_sources` captures and restores it across its
delete, which `ON CONFLICT` alone cannot cover because the row is gone before the insert runs;
and `touch_files` carries it across in the same UPDATE that moves `indexed_at`, because for a
migrated row that column is the only evidence of age there is. The third arrived later and
shipped without the clamp the first two had, which is the argument for naming all three here.
Existing tables migrate with two DDL statements and no DML: `ADD COLUMN` with **no default**,
then `SET DEFAULT` separately. Both halves are load-bearing and each was got wrong once.

`DEFAULT` is what stamps existing rows, not `NOT NULL`. Postgres evaluates a non-volatile default
at ALTER time and applies it to every existing row through stored metadata, so `ADD COLUMN ...
DEFAULT now()` claims the whole corpus was first written at the upgrade instant. Setting the
default as a separate statement affects only future inserts and rewrites nothing, leaving existing
rows NULL — the honest answer for a row whose first write was never recorded.

And it must be DDL, never DML. The first version backfilled from `indexed_at` and then set NOT
NULL, which bricks a shared multi-tenant table permanently: the backfill is ordinary DML and the
FORCE'd RLS policy narrows it to one tenant, while `SET NOT NULL` validates with a heap scan that
is not narrowed, so it trips over another tenant's NULLs and `ensure_schema` raises for everyone,
unrepairably. Measured against a real Postgres as a `NOSUPERUSER NOBYPASSRLS` role: on a two-row two-tenant
table the role sees 1 row, the backfill touches 1, and `SET NOT NULL` still fails with
`NotNullViolation` because its validating scan sees both.

A superuser bypasses RLS, and CI connects as one, so it is tempting to conclude CI cannot check
this. It can, and does: `tests/test_tenancy.py`'s `unprivileged_dsn` fixture makes a throwaway
`NOSUPERUSER NOBYPASSRLS` role, so the policy applies to the role under test regardless of who
connects. That fixture's own docstring explains it was written because an earlier version could
only run as a superuser, "the single configuration in which RLS does not apply".

`indexed_at` stays, and still means the last write. That is the right answer for staleness, which
asks how fresh a corpus is. The two axes were being served by one column.

Two narrower residues, both genuine and both fail-closed:

- An edge whose superseding file has no recorded date applies unconditionally, the inverse of the
  rule for hits, where an unknown write time leaves the hit visible. Both refuse to present
  something as healthier than it is. Note the schema makes `indexed_at` NOT NULL, so this branch
  is defence in depth against caller input rather than a state the live table can reach.
- `unresolved` is not rewound. An `ambiguous_supersession` claim written after the as-of instant
  still forces an abstention at that instant. Fail closed, so it costs recall rather than
  correctness.

**Fan-in was a second blocker and is now fixed.** Keeping one superseder per target resolved
ties by scan order, which is time-independent and so answered a different question: where two
documents superseded the same target, a replay saw only the surviving winner's edge and could
drop a claim that was live at the instant. Renaming the two files flipped the answer. The resolver
now returns every claim per target and `resolve_successor` picks the one live at the instant;
`supersession()`'s single winner is untouched for callers who never replay.

Note what none of this touches: the mechanism is transaction time on both sides, so it needs no
event-time extraction and no objection from the first half of this document applies to it.

## Not built

Event-time extraction from prose ("last month" resolved against a session date). The measurements
above size it: ~6% of turns carry an arithmetic-resolvable relative-time expression, 55.9% of all
relative-time turns. It remains the honest boundary of the `$0` claim, and nothing here depends on
it, because `indexed_at` is recorded by the store rather than extracted from language.

## The reference instant from the conversation in scope: sound, inert, and kept as an assertion

The last open thread from the Part 4 reader. Having conceded that utterance time is the axis to
order on rather than to filter on, the reader placed the reference instant at the last utterance of
the conversation in scope. That version survives every objection in the first half of this document,
because it never compares a question's anchor against a turn's utterance date at all.

It is also inert, and the corpus says so outright. Of **5,882 turns** across the ten conversations,
**zero** fall after their own conversation's end. That is by construction rather than by luck: a
turn cannot postdate the transcript it belongs to. On in-scope questions the bound demotes nothing,
ever. It is exactly as safe as claimed, and it does exactly nothing.

**So the slot stays out of the ranker.** `recall/retriever.py` composes a dense score and an
optional reranker, and no clock reaches it. A term pinned at zero would still occupy a slot and
still invite someone to re-weight it later, on the strength of a mechanism whose measured
contribution is zero.

**And the check stays in**, because flat and absent differ in observability rather than in ranking.
Inert is a property of closed transcripts, not of corpora generally. Ingest live transcripts and the
scope has no last utterance. Merge haystacks, as §10 did when LongMemEval's per-question
haystacks became one 19,195-session corpus and hit@5 fell from 0.970 to 0.366, and there is one
boundary per source rather than one for the whole. Either change makes the bound start demoting real evidence, and it would do so quietly,
because nothing in the retrieval path announces that a no-op has become load-bearing.

`benchmarks/check_utterance_postdating.py` is that announcement, and it is the fourth member of the
`check_temporal_*` family. Five things in it are load-bearing, and each one replaced a version that
read as a check and was not one:

- **A derived instant reports VACUOUS, not PASS.** Take the instant as the maximum over a
  conversation's own turns and `said > max(said)` is false by arithmetic, on any data whatsoever.
  The first version of this script did exactly that and reported a clean zero, which is the
  guard-that-cannot-fire pattern this repo already keeps a script for. The instant must be
  DECLARED, via `--snapshot`, which is what a live scope records and a closed transcript does not.
- **The control is what makes the zero mean anything.** Widening the scope to the whole corpus,
  which is what merging does, demotes **16,681 turn-instant pairs, 28.4% of all pairs**. So the
  predicate fires; it is the closed-transcript shape, not the predicate, that makes the bound do
  nothing here. Run with `--snapshot 2023-06-01` and the invariant breaks as designed: 3,344 turns
  postdate their scope's declared instant, and the script exits non-zero.
- **That control is taken over per-conversation instants, never over the declared snapshot**, and
  a pre-push bug audit is why the sentence exists. A snapshot collapses every scope onto one
  instant, which makes the merged count non-empty exactly when P2 is, so the control agreed with
  the thing it was meant to check independently. The consequence was the mirror of the defect this
  script exists to prevent: in the one mode the design mandates, a run where the invariant
  genuinely HELD reported `CONTROL FAILED` and exited 1, leaving the passing verdict unreachable.
  A guard that cannot pass is as uninformative as one that cannot fail.

- **P2 is tested before the control**, and a second audit is why. A single transcript has nothing
  to merge with, so its control is empty by construction; with the control tested first, a live
  transcript whose turns postdate its declared instant printed a non-zero P2 count, named the
  demoted turn, and then concluded `CONTROL FAILED ... the predicate never fires`. The one corpus
  shape the script exists to announce could not reach the announcement. A non-empty P2 is itself
  proof that the predicate fired, which is all the control was ever there to establish.

- **A control that is empty BY CONSTRUCTION is not a control that failed**, and a third audit is
  why. Widening the scope to "the whole corpus" is a no-op when the corpus holds one transcript, so
  the merged count is empty there whatever the data says. Reported as a failure, it told a single
  live transcript that genuinely held that "the predicate never fires", one invocation after the
  same corpus had demonstrated it firing, and returned exit 1 for both a real break and a real
  hold. `CONTROL FAILED` is now reserved for two or more transcripts, where an empty control is a
  fact about the data.

`tests/test_utterance_postdating.py` pins the predicate, `main()`'s verdicts, its exit codes and
its printed counts, for a CI that cannot run the script against the corpus because `locomo10.json`
is gitignored. The first version covered the predicate only, which is exactly how the
unreachable-pass defect survived it. **Thirty-four rules were ablated in turn and the suite
confirmed red for all thirty-four**: the two predicate rules, the declared instant, the control's
source, five verdict rules, six P1 reports, four `dia_id` rules including testing type before
emptiness, four exit-code rules (a usage error, a malformed corpus and `--help` must none of them
share code 1 with a real finding), the real `sys.argv` path, seven printed measurements, and three
share-denominator variants, because the published 28.4% needs its composition pinned and not only
its magnitude.

**The pre-audit suite had 7 test functions and never called `main()` once**; it now has 33. So
every rule at the `main()` level, which is all three verdict rules, all four exit-code rules, the
four printed measurements and the `sys.argv` path, exists only because of the audits. That is
checkable from the file rather than from this sentence, which is the point: an earlier draft of
this paragraph claimed "twelve of those were added after the audits", a number that corresponds to
nothing countable and that I published without measuring, in the same document that argues for
verifying the label that names what you measured.

Four audit rounds each found defects in the previous round's fixes, twenty-five in total. Rounds
three and four found them mostly in the **tests**, and three of those share one shape: **a count
asserted only where its true value is zero is satisfied by a counter hard-coded to zero.** It
appeared on the P1 line, then on the P2 line after the P1 one was fixed, and each time the test
read as a check on a published figure. Alongside them, a square 2x2 fixture that could not tell
`turns x instants` from either factor squared, and an ablation that mutated a falsy-value ordering
rule by deleting the type check entirely, so it pinned "a type check exists" while the ordering it
named stayed green.

That last one is the sharpest, and it is why the count above was re-measured rather than re-quoted:
**an ablation can be mislabelled exactly the way a test can.** A red result is evidence only about
the rule the mutation actually removed, which is not always the rule in its label. An earlier
version of this paragraph said twenty-nine rules were red for all twenty-nine; the real figure was
twenty-eight of twenty-nine, and the discrepancy was a mutation that did not remove what it named.
