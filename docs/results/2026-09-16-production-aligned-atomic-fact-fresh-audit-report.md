# Production aligned atomic fact fresh audit stopped at pool construction

Measured 2026-09-16 under
[2026-09-16-production-aligned-atomic-fact-fresh-audit.md](../preregistrations/2026-09-16-production-aligned-atomic-fact-fresh-audit.md).

## Verdict

`INSUFFICIENT_FRESH_POOL`.

The frozen exclusions left 22 eligible sources with unique normalized questions, against the
required 250. The harness stopped before writing a pool, calculating construction aggregates, or
running any embedding. The protocol explicitly forbids reducing the target or weakening exclusions,
so this run cannot be converted into a pass.

## Interpretation

This is a data availability stop, not a quality result. The exclusions intentionally remove every
source exposed in three prior pools and the live source admission trace. Almost the complete local
`recall` and `sentiment-agent` eligible population has therefore already appeared in development or
evaluation artifacts.

The 22 remaining sources are still usable as a fully enumerated exploratory population because no
retrieval or embedding outcome has been observed for them. Any such experiment must be separately
preregistered, must state that its sample size was known before its prediction, and cannot authorize
serving or support a precise effect estimate.

The clean confirmatory path is prospective. New memory sources must be sealed before their queries
or retrieval outcomes are inspected, then accumulated until the required sample exists.

## Integrity

The preregistration commit is `504aecb6` and the harness commit is `8f0e2bf3`. The command failed
with `INSUFFICIENT_FRESH_POOL: need 250, found 22`. No private pool or public aggregate beyond this
failure receipt was produced.
