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
