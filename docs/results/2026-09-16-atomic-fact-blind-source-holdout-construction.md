# Atomic fact blind source holdout stopped at inventory

Measured 2026-09-16 under
[`2026-09-16-atomic-fact-blind-source-holdout-construction.md`](../preregistrations/2026-09-16-atomic-fact-blind-source-holdout-construction.md).

## Verdict

`STOP_BLIND_HOLDOUT_CONSTRUCTION`.

The fully source-disjoint holdout population is smaller than predicted. The six frozen exclusion
inputs name 1,482 distinct sources. After index, date, and prior-gold exclusions, 86 source files
remain. Only 45 of those contain a first eligible atomic fact whose normalized answer span is
unique to one source. The preregistered target required 80 distinct sources, so it is
mathematically impossible on this production snapshot.

The preflight made zero OpenRouter calls and spent zero model tokens. It did not generate a query,
run retrieval, inspect a candidate ranking, or expose a gold result. This means the 45 source
population remains unconsumed for a separately preregistered exhaustive census.

## Gate result

| Measure | Result | Frozen gate | Verdict |
| --- | ---: | ---: | --- |
| Distinct excluded sources | 1,482 | diagnostic | measured |
| Parsed unexcluded sources | 86 | diagnostic | measured |
| Eligible unique-fact sources | 45 | at least 80 | fail |
| Accepted rows | 0 | exactly 80 | fail |
| Model calls | 0 | zero before impossible target stop | pass |
| Input hash mismatches | 0 | zero | pass |

The local private receipt reports mode `0666` because Windows does not expose the requested POSIX
`0600` semantics through `chmod`; that literal gate also failed. The artifact contains no query or
gold row. Any later Windows construction must preregister and verify a user-only NTFS access list
instead of claiming a POSIX mode.

## Next decision

Do not weaken the exclusions or relabel this stopped protocol. The useful next experiment is a new
exhaustive construction over all 45 still untouched eligible sources. It should call the frozen
writer once per source, retain every question that passes the frozen validation, and treat the
result as exploratory because its effective sample size cannot support a production promotion
decision. Retrieval remains unmeasured, so this is still a clean point to preregister that census
and its later comparison.

The machine-readable aggregate is
[`2026-09-16-atomic-fact-blind-source-holdout-construction.json`](2026-09-16-atomic-fact-blind-source-holdout-construction.json).
