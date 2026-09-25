# Pre-registration: is the evidence really in C9's context, when the judge must quote it?

**Date:** 2026-09-25   **Status:** predicted, not yet measured

## The question

The C9 BEAM probe (`2026-09-24-c9-beam-ability-probe.md`) scored coverage with Qwen3-14B judging a
context of about 20K tokens: 0.95 for summarization, 0.92 for multi-session, 0.91 for event
ordering, 0.53 for temporal and 0.84 for information extraction. Answer scores were far lower. The
top-N follow-up (`2026-09-24-c9-beam-topk-reader.md`) was null, and it named judge leniency as the
first reading to rule out: "the evidence is present" may simply be untrue.

This pass re-measures coverage for the same 200 questions of those five types, from the same stored
retrieval with all 100 items in returned order, without trusting the judge:

- `openai/gpt-4.1-mini` must return, per rubric criterion, a VERBATIM quote of at most 300
  characters from the context, or an empty quote;
- a criterion counts only if its quote occurs in the context after normalising whitespace and case
  (`verified_quotes` in `benchmarks/beam/aml_c9_probe.py`);
- a non-empty quote that fails the check is counted as invented.

It is offline and sends nothing to C9. Model spend is capped at $2.

## Predictions

| type | lenient coverage (run 1) | quote-verified coverage |
| --- | --- | --- |
| summarization | 0.950 | 0.45 to 0.80 (point 0.65) |
| multi_session_reasoning | 0.923 | 0.50 to 0.85 (point 0.70) |
| event_ordering | 0.911 | 0.45 to 0.80 (point 0.65) |
| temporal_reasoning | 0.529 | 0.25 to 0.55 (point 0.40) |
| information_extraction | 0.838 | 0.60 to 0.85 (point 0.75) |

Invented quotes: 2% to 15% of the non-empty quotes.

## Decision rule, fixed now

Answer scores from run 1: summarization 0.304, multi-session 0.550, event ordering 0.225.

- **Reader miss confirmed:** quote-verified coverage minus the answer score is 0.20 or more for
  summarization AND event ordering. The next lever is the CONTENT the reader gets: dates, speakers,
  or Add-time synthesis such as a summary record per session.
- **The lenient judge misled:** quote-verified coverage is at most the answer score plus 0.10 for
  both. The next lever is RETRIEVAL coverage: diversifying across sessions, or pulling in neighbours.
- Anything between is reported as mixed, per type.

## What this cannot show

A single quote of 300 characters per criterion undercounts evidence that is spread over several
passages. So this measure is strict in the opposite direction from the lenient judge, and the truth
lies between the two. The quoting model may also quote verbatim text that does not actually support
the criterion. The verbatim check rules out invention, not irrelevance.

## Result

(appended after the run; nothing above is edited)

### Run 1, 2026-09-25 05:03 to 05:09 UTC, VPS2, offline

The code is `benchmarks/beam/aml_c9_probe.py` at `9860e342`. **The run stopped at the
pre-registered $2.00 cap after 178 of 200 questions** (34 to 36 per type): gpt-4.1-mini cost more
per question than my estimate. The remaining 22 were not run. The lenient and answer columns below
are recomputed on the same scored subset, so every row compares like with like.

| type | scored | quote-verified | lenient (run 1) | answer (run 1) | verified minus answer | invented quotes |
| --- | --- | --- | --- | --- | --- | --- |
| summarization | 34 | **0.764** | 0.941 | 0.329 | **+0.435** | 30 of 157 (19.1%) |
| multi_session_reasoning | 35 | **0.805** | 0.914 | 0.557 | **+0.247** | 18 of 85 (21.2%) |
| event_ordering | 35 | **0.714** | 0.899 | 0.233 | **+0.481** | 31 of 164 (18.9%) |
| temporal_reasoning | 34 | **0.618** | 0.567 | 0.297 | +0.321 | 16 of 55 (29.1%) |
| information_extraction | 36 | **0.745** | 0.819 | 0.812 | -0.067 | 15 of 77 (19.5%) |

**Predictions:** the quote-verified coverage held for summarization, multi-session, event ordering
and information extraction. It was falsified high for temporal (0.618 against 0.25 to 0.55).
Invented quotes were falsified high (19% to 29% against 2% to 15%). Those count as 0, so the
verified figures are conservative.

**Decision rule: "reader miss confirmed".** Verified coverage minus the answer score is +0.435 for
summarization and +0.481 for event ordering, both well past 0.20. Even when every point must be
backed by a verbatim quote, 71% to 80% of the evidence those answers need sits in the 100 items C9
returns, and the reader scores 0.23 to 0.33. The lenient judge overstated coverage by 0.11 to 0.19,
but not enough to change the reading.

Information extraction is a useful calibration row. Its verified coverage (0.745) is BELOW its
answer score (0.812), so the strict measure undercounts real evidence, as the pre-registration
expected. The truth lies between the strict and lenient columns.

**Next lever, per the rule: the content the reader gets, not retrieval.** Candidates are dates and
speakers in the windows (another session measured +18 temporal points on LoCoMo), and Add-time
synthesis such as a summary record per session for summarization.
