# Production structural edge canary rerun

Date: 2026-09-10

Status: passed.

## Repairs

The planner metrics access was repaired to read `plan.trace.expansion_steps`, and the production
generation store now uses generation-scoped `chunk_id`, `source_uri`, and `generation_id` queries.
The production serving checkout then received the service-only calibrated graph reranking repair
from `ab68875`, removing the obsolete seed-relative cosine rejection while retaining ordinary trust
evaluation and the structural safety checks.

Production serving HEAD at measurement time:

```text
94363549cbdacb20a94a3b4268bd8d216bef3022
```

The local repair was `8203c0c8`; the production serving commits were `563fb73` and `9436354`.

## Calibration

The locked three-file fixture was ingested into the isolated tenant
`structural-edge-canary-20260910` as generation `gen_01e5d39a320848c2a72787424f32afd9`.
The preregistered 20 answerable and 20 unanswerable labels certified calibration
`cal_d1e47ca54b3d46b99b5a0186ce005b62` with separability `1.0 [1.0, 1.0]`, threshold `0.447`, and
checksum `8701c78e946ed7f1e209b10665c0f643f139ca8543932e2305d13d1fb84306f2`.
The calibration was published and the generation was promoted only in the canary tenant.

## Paired result

The original query and budgets were unchanged: `Who owns the booking?`, `k=1`,
`mode=proposal_assisted`, `max_steps=4`, `max_graph_nodes=4`, and `max_evidence_tokens=512`.

| Arm | Evidence | Graph diagnostics |
|---|---|---|
| `off` | `a.md` | graph not requested; 0 relations inspected; 0 candidates |
| `one_hop` | `a.md`, then `b.md` | ready; 1 relation inspected; 1 candidate discovered; 1 accepted; 1 new trusted item; 0 rejected |

Both arms returned the same generation. The baseline evidence item was byte-identical. The
`one_hop` arm reported `references` seed activations `1`, candidate admissions `1`, and new trusted
evidence `1`. The added item was exactly `b.md`, with verdict `ok` and text:

```text
# Canary B

The attached itinerary contains the booking details.
```

Both responses had `trust_state=trusted` and `outcome=abstained` with `refusal_reason=no_answer_provider`.
That answer-provider refusal is orthogonal to the measured evidence expansion; the structural
effect criterion passed.

## Cleanup

`recall_forget` removed 3 chunks, 3 sources, and 3 staged upload files, with no sources not found.
A follow-up inventory returned an empty entry list. The existing `memory` tenant was not touched.

## Verification notes

The local focused suite passed 99 tests, Ruff, and compilation. The remote graph suite passed 22 of
23 tests; the only failure was an older assertion expecting the pre-reranking `semantic_graph_precision_v1`
policy fingerprint. Remote compilation and the related-evidence regression passed.
