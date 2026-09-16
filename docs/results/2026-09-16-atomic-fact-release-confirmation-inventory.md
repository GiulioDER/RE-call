# Atomic fact release confirmation inventory passed

Measured 2026-09-16 under
[`2026-09-16-atomic-fact-release-confirmation-inventory.md`](../preregistrations/2026-09-16-atomic-fact-release-confirmation-inventory.md).

## Verdict

`READY_TO_BUILD_ATOMIC_RELEASE_CONFIRMATION`.

The production snapshot contains 1,070 eligible atomic-development-disjoint sources, more than
six times the frozen minimum of 160. This removes the earlier 45 source ceiling and supports the
planned 96 source prospective confirmation without weakening the development isolation boundary.

The earlier holdout excluded 1,482 sources because it treated every historical retrieval negative
as prior exposure. The corrected boundary excludes 335 distinct sources that actually supplied
facts to atomic parser, representation, pilot, current-query, or blind-rescue development. It
still checks candidate fact uniqueness against the entire 1,551 source snapshot, including those
excluded sources.

## Gate results

| Measure | Result | Frozen gate | Verdict |
| --- | ---: | ---: | --- |
| Snapshot sources | 1,551 | diagnostic | measured |
| Sources with atomic views | 1,404 | diagnostic | measured |
| Development-excluded sources | 335 | diagnostic | measured |
| Eligible candidate sources | 1,070 | at least 160 | pass |
| Production roots represented | 4 | at least 3 | pass |
| Largest root share | 62.3% | at most 80% | pass |
| Old blind candidates reconstructed | 45 | exactly 45 with frozen hash | pass |
| Accepted census sources in old population | all | all | pass |
| Source and view integrity errors | 0 | 0 | pass |
| Restricted private ACL | true | true | pass |

Candidate distribution is 667 `sentiment-agent`, 344 `recall`, 58 `agent-memory-bench`, and one
`ai-boost-cad`. Construction distribution is 554 extractive fallback, 443 extractive heading, and
73 extractive field views.

## Receipts

| Artifact | SHA256 |
| --- | --- |
| Source snapshot | `12885c6ea1cde2f81758d3cc247b77e3127ba10484f1c257289f9d3946b7722d` |
| Candidate manifest | `fdd00ba7d356b4b07d376c5b2d87b146c2f58c45bf35ea55b17d85e21fcf8e62` |
| Private inventory file | `a5df11069bce5eb4041e87ca5ebf3e3b1d8b35fb3b1814b61cd1725e46eba768` |
| Reconstructed old candidate manifest | `edfd11a46216b4b10b3b063c65623456bfd75876c37667f43e57505c616d1b60` |

The private inventory remains outside the repository under
`C:\Users\gde00\.codex\evals\atomic-fact-release-confirmation-2026-09-16`. It contains source
identifiers and row-level hashes. The committed JSON contains aggregates and hashes only.

The inventory made no model calls, embeddings, retrieval requests, or production-serving changes.
The next step is the separately preregistered 96 row question construction.

The machine-readable aggregate is
[`2026-09-16-atomic-fact-release-confirmation-inventory.json`](2026-09-16-atomic-fact-release-confirmation-inventory.json).
