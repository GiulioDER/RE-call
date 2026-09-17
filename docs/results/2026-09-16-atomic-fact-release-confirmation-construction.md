# Atomic fact release confirmation construction passed

Measured 2026-09-16 under
[`2026-09-16-atomic-fact-release-confirmation-construction.md`](../preregistrations/2026-09-16-atomic-fact-release-confirmation-construction.md).

## Verdict

`READY_TO_PREREGISTER_ATOMIC_RELEASE_RETRIEVAL`.

The frozen writer accepted 96 valid questions from 96 distinct sources after 119 candidate calls.
It remained below the 160 attempt ceiling and preserved the preregistered source order. No query
was regenerated, no rejected output was retried, and no retrieval result was inspected.

## Gate results

| Measure | Result | Frozen gate | Verdict |
| --- | ---: | ---: | --- |
| Accepted rows | 96 | exactly 96 | pass |
| Distinct sources | 96 | exactly 96 | pass |
| Attempted sources | 119 | at most 160 | pass |
| Production roots represented | 3 | at least 3 | pass |
| Largest root share | 69.8% | at most 80% | pass |
| Source and answer integrity errors | 0 | 0 | pass |
| Question validation errors | 0 | 0 | pass |
| Provider calls accounted | 119 of 119 | all | pass |
| Restricted private ACL | true | true | pass |

The accepted rows contain 67 `sentiment-agent`, 26 `recall`, and three `agent-memory-bench`
sources. Their constructions are 47 extractive fallback, 45 extractive heading, and four
extractive field views.

The 23 rejected outputs were 11 five-token answer overlaps, seven invalid JSON responses, and five
forbidden structural words. Every rejection advanced to the next candidate with no retry.

## Provider receipt

| Field | Value |
| --- | ---: |
| Model | `deepseek/deepseek-v4-flash` |
| Calls | 119 |
| Prompt tokens | 20,131 |
| Completion tokens | 28,145 |
| Total tokens | 48,276 |
| Provider latency | 556,265 ms |
| Reported cost | $0.007979 |

The private pool SHA256 is
`414b441fd95ccc0de7a6ede329515941c09f8aef8e9fc8e20ccbdfc1d7514f0f`. It remains outside the
repository under `C:\Users\gde00\.codex\evals\atomic-fact-release-confirmation-2026-09-16`.
The committed artifact contains only aggregate counts, hashes, and provider accounting.

The machine-readable aggregate is
[`2026-09-16-atomic-fact-release-confirmation-construction.json`](2026-09-16-atomic-fact-release-confirmation-construction.json).
