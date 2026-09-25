# Pre-registration: T-2, a date-restricted retrieval leg for questions that name a time

**Date:** 2026-09-25   **Status:** predicted, not yet measured

Branch `claude/mm1-mm3`. Baseline: served C9 at `3eb447c4`. Local directional experiment over
public data; not an AML hosted evaluation.

## Why

Temporal questions are the one Textual type measured as limited by retrieval, not by the reader:
on BEAM 100K, C9's returned 100 items cover the rubric of temporal_reasoning questions at 0.53
(lenient judge) to 0.62 (quote-verified), against 0.71 to 0.95 for the other types
(`docs/preregistrations/2026-09-24-c9-beam-ability-probe.md`). Cycle 1's C1 median is 16.4 with
leaders at 58 to 65. T-1 (resolving relative dates inside returned items) works on what is already
retrieved; T-2 is its retrieval-side partner: when the question itself names a time, also retrieve
from the memories stored at that time.

Two facts fix the design:

1. **Search is not told when the question is asked.** `SearchRequest` carries `query`, `user_id`,
   `top_k` and `options` only (`recall_aml/models.py`). Absolute expressions ("in March 2023",
   "on 7 May") resolve on their own; relative ones ("last Saturday", "three weeks ago") need an
   anchor. T-2 anchors them to the tenant's latest stored memory time, since in AML's flow a Search
   follows the Adds it asks about. How far that proxy sits from the true question date is measured
   in Stage 0, not assumed.
2. **LoCoMo has no retrieval headroom** (turn-level evidence is in the returned 100 for 99.87% of
   questions, `docs/preregistrations/2026-09-24-aml-c9-locomo-loss-diagnosis.md`), so T-2 cannot be
   judged there. LongMemEval marks its evidence sessions (`answer_session_ids`, `has_answer`), so
   evidence recall can be measured with no reader and no judge, and it has temporal questions of
   its own.

## The mechanism, fixed now

New module `recall_aml/temporal_query.py`, default-off variant flag `date_leg`, experiment override
`RECALL_AML_DATE_LEG`:

1. **`query_time_range(query, anchor)`** returns a UTC date range, or nothing. Absolute: a full date
   in any of `2023-05-07`, `7 May 2023`, `May 7, 2023` (the day); a month with a year (the month); a
   month without a year (that month in the anchor's year, or the previous year if that month has
   not yet begun at the anchor); a bare year (the year). Relative, against the anchor day:
   `yesterday`, `last night` (the day); `last` / `this past` weekday (that day); `last week`,
   `last weekend`, `last month`, `last year` (the span); `N days / weeks / months / years ago`
   (the day, week, month or year it points at, widened by ±1 of its own unit). The earliest
   match in the query wins; a query with none gets no leg.
2. **Anchor:** the tenant's latest `event_time` among stored raw items.
3. **The leg:** when a range exists, a second dense query over the primary index (`voyage-code-4`),
   restricted to raw items whose `event_time` falls in the range, top 20, fused into the main
   ranking by the same deterministic RRF as the multimodal visual leg (`fuse_hits`). No range, no
   leg: the Search is byte-identical to today.

No part of 1 to 3 is tuned on any measured set.

## Stage 0: census (free, no model, no service)

Share of questions for which `query_time_range` returns a range, on public question text:

| Set | n | Predicted share with a range |
|---|---|---|
| LoCoMo, all | 1,535 | 0.05 to 0.20 |
| LoCoMo, category 2 (temporal) | 320 | 0.10 to 0.30 |
| BEAM 100K, all probing questions | 400 | 0.10 to 0.30 |
| BEAM 100K, temporal_reasoning | 40 | 0.20 to 0.60 |
| LongMemEval-S, all | 500 | 0.10 to 0.35 |
| LongMemEval-S, temporal-reasoning | 133 | 0.30 to 0.70 |

Also measured, not predicted as a gate: on LongMemEval-S, the gap in days between each question's
latest haystack session and its `question_date` (how good the anchor proxy is), and, on questions
with a relative expression, the share whose range computed from the proxy anchor contains the range
computed from the true `question_date`.

**Gate:** Stage 1 runs only if LongMemEval-S temporal-reasoning has a share of at least 0.15. If the
anchor proxy's range matches the true-date range on fewer than half of the relative-expression
questions, relative expressions are dropped from Stage 1 and only absolute ones are tested, stated
in an amendment before Stage 1 starts.

## Stage 1: retrieval (VPS3; no reader, no judge)

- **Data:** LongMemEval-S (`xiaowu0162/longmemeval-cleaned` at
  `98d7416c24c778c2fee6e6f3006e7a073259d48f`), the temporal-reasoning questions for which Stage 0
  found a range, each question its own tenant holding its whole haystack, one Add per session with
  the session's date on every message.
- **Arms:** B (served C9) and T2 (C9 with `date_leg`), as two processes over one ingest, as in the
  MM-1 experiment; the query-embedding cache is shared, so both arms search with the same vector.
- **Metric:** evidence-session recall: the share of a question's `answer_session_ids` present in the
  returned top 10, and in the top 100, by the returned items' `session_id`. Paired by question,
  bootstrap 10,000, seed 20260925.
- **Control:** on questions with no range the two arms must be byte-identical (apparatus check).

| Contrast | Predicted |
|---|---|
| T2 − B, evidence recall@10, questions with a range | **+0.05**, band 0.00 to +0.12 |
| T2 − B, evidence recall@100, questions with a range | +0.02, band 0.00 to +0.06 |
| T2 − B, any metric, questions without a range | exactly 0 (check) |

## Stage 2: answers (only if Stage 1 passes)

Only if T2 − B on recall@10 is at least +0.03 with a CI lower bound above 0. The same questions are
then answered and judged with LongMemEval's temporal prompts from the pinned AML checkout when
present there (otherwise LongMemEval's own), by `deepseek/deepseek-v4.1-flash` pinned to one
provider, reasoning off, bounded output; arms B, B2 (noise floor) and T2 interleaved per question.

| Contrast | Predicted |
|---|---|
| T2 − B, answer accuracy, questions with a range | +3 points, band −2 to +8 |
| B2 − B | within ±3 points |

## What would falsify this

- Stage 0: LongMemEval temporal share below 0.15: T-2 barely fires where it is aimed; stop.
- Stage 1: T2 − B recall@10 at or below 0 on questions with a range, or any difference at all on
  questions without one (the leg leaks).
- Stage 2: T2 − B accuracy at or below 0.

## Decision rule

Recommend T-2 if Stage 1 recall@10 gain is at least +0.03 with CI above 0 and Stage 2 accuracy gain
is at least +2 points with |B2 − B| smaller, and a later check shows LoCoMo overall and Coding MRR
non-inferior (at least −0.5 points and −0.01). Any change to C9 needs an explicit user decision and
never during an AML job.

## Apparatus checks, fixed now

1. `query_time_range` unit tests, one per pattern, each seen red against a deliberate mutation
   before green; a query with no expression returns nothing.
2. The service test: with the flag on and no range in the query, Search output is byte-identical to
   the flag off (red-proved by forcing the leg on).
3. Stage 1 logs, per Search, whether the leg ran and the range used; the leg must run on exactly the
   questions Stage 0 marked.
4. Every question's evidence sessions are present in its tenant (checked after ingest).

## Spend

Stage 0 and Stage 1's metrics: none. Stage 1's ingest runs C9's Add-time compile on DeepSeek V4.1
Flash (as in the MM-1 experiment, about USD 0.001 per Add) for about 50 sessions per question;
capped at USD 5, with the USD 40 balance floor. Stage 2: about USD 2 to 4. Nothing runs alongside
another OpenRouter job.

## What I already know

- T-1 fires on 99.9% of LoCoMo questions' top 20 (`docs/preregistrations/2026-09-25-aml-c9-relative-dates-and-conflict-adjacency.md`);
  those phrases are in memories, not in questions, so that says nothing about T-2's firing rate.
- The multimodal census found 4.4% of public questions name their medium; questions often name
  content rather than form (`docs/preregistrations/2026-09-25-aml-c9-multimodal-scope-and-dates.md`).
  The same may hold for time, which is why Stage 0 comes first.

## Confounds I can name now

- **The anchor proxy.** A tenant's latest memory is not the question's date; LongMemEval questions
  can be asked long after the haystack ends. Stage 0 measures the gap.
- **`event_time` is when a session was stored, not when the event happened.** A memory about last
  year's trip stored today is outside a "last year" range. T-2 retrieves by storage time only.
- **LongMemEval is one of AML's Textual sources**; nothing here is tuned on it.

<!-- FROZEN PREREGISTRATION ENDS HERE. APPEND RESULTS BELOW. -->

## Results

No measurement had run when this record was committed.
