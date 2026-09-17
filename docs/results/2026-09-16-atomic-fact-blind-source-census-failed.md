# Atomic fact blind source census stopped on completion ceiling

Measured 2026-09-16 under
[`2026-09-16-atomic-fact-blind-source-census.md`](../preregistrations/2026-09-16-atomic-fact-blind-source-census.md).

## Verdict

`STOP_BLIND_SOURCE_CENSUS`.

The first OpenRouter response from `deepseek/deepseek-v4-flash` ended with
`finish_reason=length` at the frozen 120 completion token ceiling. The repository client raised
`CompletionTruncated` before returning content to the census builder. The protocol required a hard
stop on any provider exception, so the run was not retried.

No private pool or public census artifact was written. There are zero usable completions, zero
accepted questions, and zero retrieval calls. No dense or atomic ranking has been observed for any
candidate.

This failure is a harness ceiling error, not retrieval evidence and not a question quality result.
The smallest valid correction is a separately preregistered run with the same source manifest,
model, prompts, validation, and gates, changing only the completion ceiling.

The machine-readable stop receipt is
[`2026-09-16-atomic-fact-blind-source-census-failed.json`](2026-09-16-atomic-fact-blind-source-census-failed.json).
