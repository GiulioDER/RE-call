# Atomic fact Context 4 retrieval pilot passed

Measured 2026-09-16 under
[`2026-09-16-atomic-fact-exhaustive-22-pilot.md`](../preregistrations/2026-09-16-atomic-fact-exhaustive-22-pilot.md),
with the pre-outcome implementation clarifications for
[source scope](../preregistrations/2026-09-16-atomic-fact-context4-implementation-clarification.md),
[source sanitation](../preregistrations/2026-09-16-atomic-fact-context4-sanitization-clarification.md),
and [manifest scope](../preregistrations/2026-09-16-atomic-fact-context4-manifest-scope-clarification.md).

## Verdict

`PROMISING_ATOMIC_FACT_PILOT`, with `atomic` selected. Both atomic and equal RRF cleared every
frozen decision gate. This is the first positive retrieval result in the current research sequence.

It authorizes prospective gold accumulation and an isolated product shadow. It does not authorize
serving, tuning on these 22 consumed rows, or a production claim.

## Frozen result

| Arm | Exact @1 | Exact @3 | Exact @5 | Exact @10 | Exact @20 | Gold @1 | Gold @3 | Gold @5 | Gold @10 | Gold @20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 4 | 12 | 14 | 16 | 18 | 12 | 16 | 17 | 19 | 19 |
| Atomic | 16 | 20 | 21 | 22 | 22 | 20 | 21 | 21 | 22 | 22 |
| Equal RRF | 15 | 17 | 19 | 21 | 22 | 19 | 19 | 20 | 21 | 22 |

Relative to dense, atomic gained 12 exact and 8 gold rank one successes with zero losses. It also
gained four exact and three gold successes at rank 20. Seventeen rank one selections changed;
changed selection precision was 0.706 for exact span and 0.882 for gold source.

Equal RRF gained 11 exact and 7 gold rank one successes with zero losses. It matched atomic at rank
20 but trailed it at every lower frozen cutoff, so the preregistered tie rule selected atomic.

## Post hoc family diagnostic

This diagnostic was calculated only after the frozen verdict and did not affect arm selection.

| Family | Rows | Dense exact @1 | Atomic exact @1 | Dense gold @1 | Atomic gold @1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fallback | 9 | 1 | 5 | 4 | 9 |
| Field | 2 | 0 | 1 | 0 | 1 |
| Heading | 11 | 3 | 10 | 8 | 10 |

Atomic reached every exact span and gold source by rank 20 in all three families. The benefit is not
confined to headings, although the two-row field family is too small to interpret separately.

## Corpus and integrity

The auxiliary corpus contains 6,304 views across 1,410 sources, mapped to 11,352 pinned ordinary
chunks. Fifteen aggregate or index sources were excluded by the frozen basename rule, seven
manifest objects were outside the seven frozen roots, and 151 included sources produced no eligible
view. Every included manifest object matched its pinned bytes and every view matched its unchanged
parent before embedding.

Voyage Context 4 produced 1,024-dimensional vectors under the shared embedding lock, 8 GB memory
limit, zero swap, 250 percent CPU quota, four threads, and lowered priority. The run took 112.120
seconds. Production routing and the active generation were not changed.

The private vector artifact SHA256 is
`c45534c79ce10ea64076a21fe85eb211f87344958dc1329be7a4ee3debdfb27c`. The private row artifact
SHA256 is `a3f88b03b9a12cde3dcde039873f9238f86540e3dd63b94318b8325204e3eec4`.
Both remain outside the repository with mode 0600.

## Limits

The pilot has only 22 rows and covers the complete available population after extensive source
exclusions. The questions were generated from the same title, heading, and field structure that the
atomic rendering exposes, so structural coupling may overstate performance on natural user
queries. The frozen baseline is a retained production generation rather than the current active
generation. Latency, online candidate union behavior, and answer quality were not measured.

## Next action

Build an isolated atomic shadow beside the current production generation, without serving it. Seal
new prospective queries before retrieval, using query wording that does not expose the rendering
template and sources not present in this pilot. Compare dense and atomic on exact span, gold source,
latency, and zero-view fallback behavior. Do not tune rendering or fusion on these 22 rows.

The machine-readable aggregate is
[`2026-09-16-atomic-fact-context4-pilot.json`](2026-09-16-atomic-fact-context4-pilot.json).
