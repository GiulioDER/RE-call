# Pre-registration: E-2, a revised ordering gate, measured only on questions E-1 never read

**Date:** 2026-09-25   **Status:** predicted, not yet measured

Branch `claude/mm1-mm3`. Baseline: served C9 at `3eb447c4`. Local directional experiment over public
data; not an AML hosted evaluation.

## Why

E-1's Stage 0 (`docs/preregistrations/2026-09-25-aml-c9-gated-chronological-order.md`, result and
correction) found its frozen phrase list reaches 39 of BEAM 100K's 40 event-ordering questions
through one phrase, `the order (in which|that|of)`, which is BEAM's template; misses 27 of 36
ScriptMem ordering questions, which say "what is the correct order" and "from nearest to farthest";
misses BEAM's "can you list in order how", because `list .* in` needs a word between; and fires
falsely five times on BEAM, every time on `timeline` used as a noun. That record said a revised
list would be a new pre-registration and could only be tested on data not used there. This is it.

The revised list is written from those observations, so every question E-1's census read (BEAM
100K, ScriptMem, LoCoMo, LongMemEval-S) is development data for E-2 and is not used to judge it.

## The revised gate, fixed now

`asks_for_order_v2(question)` in `recall_aml/order_gate.py`, matched as E-1's is (case-insensitive,
anywhere in the question, `.` does not cross a line break). It is E-1's list with three changes:

**Removed:** `timeline`, replaced by `timeline of`.

**Changed:** `list .* in (the )?order` becomes `list (.* )?in (the )?order`, so "list in order"
matches.

**Added:** `correct order`, `right order`, `nearest to farthest`, `farthest to nearest`,
`newest to oldest`, `latest to earliest`, `last to first`.

The full list, for the record:

`in what order`, `in which order`, `what order`, `chronological`, `chronologically`, `sequence of`,
`the order (in which|that|of)`, `order (did|do|were|was)`, `timeline of`, `which came first`,
`which happened first`, `what happened (first|next|last|before|after)`,
`list (.* )?in (the )?order`, `rank .* by (date|time)`, `earliest to latest`, `oldest to newest`,
`first to last`, `correct order`, `right order`, `nearest to farthest`, `farthest to nearest`,
`newest to oldest`, `latest to earliest`, `last to first`.

Nothing else. **The action is E-1's unchanged**: when the gate fires, the returned items are
re-sorted by `created_at`, earliest first, whatever direction the question asks for; only the gate
differs, so any difference between E-1 and E-2 is attributable to which questions they reach.
Reverse-direction phrasings are gated because the question is still about order; the reader, not
the renderer, handles direction.

## Held-out questions

I have not read the question text of any of these. X-1's Stage A mapped the four unlabelled sources,
and its report and committed outputs carry counts and mappings, not question text. Each question is
taken as its source file holds it, before any suffix an adapter appends (X-1 appends one to
PersonaMem-v2 queries).

| Set | Source | Order label |
|---|---|---|
| BEAM 500K, all probing questions | public BEAM release, `data/500K-00000-of-00001.parquet`, dataset revision `3205395e` | `event_ordering` |
| BEAM 1M, all probing questions | same release, `data/1M-00000-of-00001.parquet` | `event_ordering` |
| MemLens 32K, all questions | X-1's pinned file | none |
| MobileMem-Omni, English filtered questions | X-1's pinned file | none |
| PersonaMem-v2, all queries | X-1's pinned file | none |
| CLBench, the final user turn of every task | X-1's pinned file | none |

**BEAM 500K and 1M are held out in items, not in authorship.** They come from the same generator as
BEAM 100K and very likely share its templates, so recall there says whether the gate still catches
BEAM, not whether it generalises. The unlabelled sources are the generalisation test.

**Judging a fire on an unlabelled source.** Every question either gate fires on is written out
verbatim and judged by one rule, fixed now: *it asks the reader to produce, choose or check an order
of two or more events or items in time or sequence.* A question that only mentions a time, a
duration, or a single "first" or "last" is not an ordering question. I judge each fire with the
gate's identity hidden (fires of both gates pooled, deduplicated, shuffled with
`random.Random(20260925)`), and the judgements are committed before the gates are unblinded. Misses
on unlabelled sources cannot be counted and are not claimed.

## Stage 0, census (free, no model)

Both gates, E-1's frozen list and E-2's, run on every held-out question.

| Measure | E-1 predicted | E-2 predicted |
|---|---|---|
| BEAM 500K + 1M event_ordering, share caught | 0.90 to 1.00 | at least E-1's, and at most +0.05 above it |
| BEAM 500K + 1M other types, false-fire rate | 0.005 to 0.03 | 0.000 to 0.015, below E-1's |
| Unlabelled sources, share of all questions fired, each source | 0.00 to 0.03 | 0.00 to 0.03 |
| Unlabelled sources, share of fires judged ordering, pooled | 0.50 to 0.90 | 0.60 to 0.95, at least E-1's |

The BEAM prediction for E-2's gain is small on purpose: if 500K and 1M share 100K's template,
E-1 already catches them, and the additions target wordings BEAM did not use.

**Gate for Stage 1 (E-2 joins E-1's answer run):** on held-out BEAM, E-2's recall at least E-1's and
its false-fire rate at most E-1's; and on the unlabelled sources pooled, E-2's number of fires
judged *not* ordering at most E-1's. If any fails, E-2 stops at Stage 0 and E-1 keeps its list.

## Stage 1, answers (only if Stage 0 passes)

E-2 adds one arm, **E2**, to E-1's Stage 1 run rather than a run of its own: the same stored
retrieval, reader, prompts, three answers per question per arm, interleaved, and spend cap. The run
answers the union of the questions either gate fires on; on a question only one gate fires on, the
other gate's arm equals H by construction and is not re-asked. E-1's record, arms, predictions and
decision rule are untouched by this.

| Contrast | Set | Predicted |
|---|---|---|
| E2 − H | BEAM event_ordering, questions E-2 fires on | +0.05, band 0.00 to +0.10 (E-1's prediction) |
| E2 − E1 | BEAM, questions where the gates disagree | within ±0.02 of 0; the gates should disagree on few BEAM questions |
| E2 − H | LongMemEval-S gated ordering questions | +0.02, band −0.05 to +0.08; direction only |

## Decision rule

Recommend E-2's list over E-1's only if Stage 0 passes, and in Stage 1 E2 − H on BEAM event
ordering is at least E1 − H minus the H′ − H noise floor. E-2 then inherits E-1's remaining
conditions (X-1 held-out non-inferiority, never deployed during an AML job, serving decision the
user's). If E-1 itself is not recommended, neither is E-2: the revision changes reach, not the
mechanism.

## Apparatus checks, fixed now

1. `asks_for_order_v2` unit tests: one per added or changed pattern, `timeline` as a noun not
   firing, the near-misses E-1 already rejects still rejected, each seen red against a deliberate
   mutation before green.
2. E-1's `asks_for_order` and its tests unchanged; E-1's census result reproduces exactly at the
   E-2 commit (same counts on the same files).
3. The blinded judging file is committed before the unblinded counts are computed.

## Spend

Stage 0: none; the BEAM 500K and 1M parquets are fetched once to VPS3, where the census runs in a
throwaway environment, never on VPS2. Stage 1: inside E-1's cap (USD 5), since E2 only adds the
questions E-2 reaches and E-1 does not.

## What I already know

- Every question and count in E-1's Stage 0 result and its correction.
- X-1 Stage A's source counts and mappings, which gave no question text for the held-out sources.
- BEAM 500K and 1M exist in the public release at revision `3205395e`; I have read their file names,
  nothing inside them.

## Confounds I can name now

- **Same generator.** BEAM held-out recall is expected to be high for both gates and says little
  about generalisation; the unlabelled sources carry that question.
- **My judging is not independent of my list.** I wrote the list and I judge the fires; blinding to
  which gate fired removes the obvious bias but not a shared notion of what counts as order. The
  rule is written above so a second reader can re-judge the committed file.
- **Few fires expected on unlabelled sources.** Most of those questions are about preferences,
  images and documents; a pooled count in single digits cannot rank the gates, only flag one that
  fires on the wrong kind of question.

<!-- FROZEN PREREGISTRATION ENDS HERE. APPEND RESULTS BELOW. -->

## Results

No measurement had run when this record was committed.

### Stage 0 result, 2026-09-26

`scripts/aml_e2_census.py` at `57781a68` on VPS3, in a throwaway uv environment. BEAM 500K and 1M
fetched from the public release at `3205395e` (sha256 prefixes `41b5acbbb55a586b` and
`af05921c97935503`); the unlabelled sources from X-1's pinned files. **Apparatus check 2 held:**
E-1's census re-run at this commit is identical to the committed one, key for key. Check 1: the
E-2 tests (`fa7a3dc1`) pass 46 of 46 and each of the three recorded mutations failed its named test.
Check 3: the blinded judging file's index (`b553bf17`, file sha256 `a4be58b765091387...`) and key
hash were committed before any judgement, the judgements (`b7ce8aae`) before the key was read, and
the key's hash was unchanged at unblinding. Outputs: `results/aml-e2/census.json`,
`judge-index.json`, `judgements.json`, `unblinded.json`. The judging file itself is not committed:
it is 380k characters of third-party dataset text, in a public repository.

**Labelled held-out sets (BEAM), counts are questions:**

| Set | E-1 caught | E-2 caught | E-1 false fires | E-2 false fires |
|---|---:|---:|---:|---:|
| BEAM 500K (70 event_ordering, 630 other) | 56 (0.800) | 57 (0.814) | 3 (0.005) | 1 (0.002) |
| BEAM 1M (70 event_ordering, 630 other) | **9 (0.129)** | **9 (0.129)** | 5 (0.008) | 3 (0.005) |

**Unlabelled held-out sets, share of questions fired:**

| Source | n | E-1 | E-2 |
|---|---:|---:|---:|
| MemLens 32K | 789 | 25 (0.032) | 25 (0.032) |
| MobileMem-Omni EN | 1,171 | 22 (0.019) | 22 (0.019) |
| PersonaMem-v2 | 5,000 | 8 (0.002) | 3 (0.001) |
| CLBench (final user turn) | 1,899 | **427 (0.225)** | **151 (0.080)** |

**Blind judgements** (487 pooled fires, one rule, gate identity hidden; 53 judged ordering: MemLens
25/25, MobileMem 18/22, PersonaMem 1/8, CLBench 9/432):

| | E-1 | E-2 |
|---|---:|---:|
| fires | 482 | 201 |
| judged ordering | 53 | 48 |
| judged not ordering | 429 | 153 |
| share judged ordering | 0.110 | 0.239 |
| same, excluding CLBench | 44/55 (0.80) | 44/50 (0.88) |

**Against the predictions:**

- E-1 BEAM held-out recall, predicted 0.90 to 1.00: **falsified**, 0.800 on 500K and 0.129 on 1M.
  E-2's, predicted at least E-1's and at most +0.05 above: **held** (+0.014 and 0).
- E-1 BEAM false-fire rate, predicted 0.005 to 0.03: 1M 0.008 held; 500K 0.0048, just under the
  band. E-2's, predicted 0.000 to 0.015 and below E-1's: **held** on both.
- Share of questions fired per unlabelled source, predicted 0.00 to 0.03 for both gates: held for
  MobileMem and PersonaMem; **falsified** for MemLens (0.032, both) and CLBench (0.225 and 0.080).
- Share of fires judged ordering, predicted 0.50 to 0.90 (E-1) and 0.60 to 0.95 (E-2): **both
  falsified** (0.110 and 0.239), entirely through CLBench; outside it both are inside their bands.

**The gate passes.** On held-out BEAM, E-2's recall is at least E-1's and its false-fire rate at
most E-1's; on the unlabelled sources, E-2's fires judged not ordering (153) are at most E-1's
(429). So E-2 joins E-1's Stage 1 answer run as arm E2, as registered.

**What the census says beyond the gate:**

1. **Both lists fit BEAM's generator, not ordering questions.** BEAM 1M asks "How did my
   discussions about X progress in order?" and "the sequence in which I brought up…"; neither list
   has `in order?` or `sequence in which`, so both catch 9 of 70. Recall on BEAM measures the
   template of the split, 0.975 on 100K, 0.80 on 500K, 0.13 on 1M. Which split AML's BEAM questions
   come from is not known here, so the gate's reach on AML's BEAM is not known either.
2. **CLBench is where a gate would do harm.** Its final user turn often carries a pasted document,
   and the phrases occur in the document: E-1 fires on 22.5% of CLBench questions and 98% of those
   fires ask for no order. Chronological order hurt information extraction by 0.167 on BEAM when
   applied ungated; firing on one CLBench question in twelve (E-2) or five (E-1) would apply exactly
   that. Neither gate should serve unless it reads only the question's ask, which is a new design
   and would need its own record.
3. **E-2's additions did not transfer.** The seven phrasings written from ScriptMem ("correct
   order", "nearest to farthest"...) fired 5 times across 8,859 held-out questions and none was an
   ordering question. E-2's gain is entirely from dropping bare `timeline`, which removed 276 false
   fires and lost 3 genuine "timeline" asks (plus 2 genuine asks E-1 had caught through a document
   word).
4. **Where genuine ordering questions live, both gates work.** MemLens's "sort these facts in
   chronological order" and MobileMem's "which happened first, A or B?" are caught by both, and
   together they are 43 of the 53 genuine orderings found.

Borderline judgements (plans, procedures, rankings by score or alphabet, audit records with a
timeline field) were all judged not ordering; they are listed with their ids in
`results/aml-e2/judgements.json`, so a second reader can re-judge the committed file.

### Stage 1 result, 2026-09-26 (E2 arm of E-1's Stage 1)

From the same run (`results/aml-e2/stage1-beam-score.json`): **E2 − H on the 40 event_ordering
questions E-2 fires on is −0.010 [−0.023, +0.002] on alignment** (predicted +0.05, band 0.00 to
+0.10: falsified) and −0.005 [−0.064, +0.056] on the rubric mean. **E2 − E1 where the gates disagree
(6 questions) is −0.145 [−0.393, +0.003]**, predicted within ±0.02: E-2's list does worse than E-1's
exactly where they differ, largely on the non-ordering questions only E-1 reorders. The decision rule
(E2 − H at least E1 − H minus the noise floor) is met only because both are near zero, and E-2
inherits E-1's result: **not recommended**. Neither list serves.
