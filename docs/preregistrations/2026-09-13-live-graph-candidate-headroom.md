# Live graph candidate headroom audit

Preregistered on 2026-09-13 before implementing or running the audit.

## Question

The linked tail live experiment failed against its same path removed topology control. Before
authoring typed relations, determine whether the frozen real memory queries contain enough
retrieval headroom for graph selection to improve gold evidence.

The audit asks three separate questions:

1. Is missing gold evidence present in raw retrieval ranks 9 through 20?
2. When it is present, does current topology connect it to a protected rank 1 through 8 seed?
3. When topology connects candidates, does the selector promote gold or non gold evidence?

## Frozen population and identity

1. Query set: `docs/preregistrations/2026-08-17-memory-queries.json`.
2. Query set SHA256: `63d290a61189758a88b47bfed981ae1331d87fc004b752d632c1f5b58f5aa192`.
3. Population: all 50 queries.
4. Tenant: `memory`.
5. Embedder: `voyage:voyage-4`.
6. Generation: pin one certified generation and record generation, pipeline, corpus, calibration,
   source commit, and artifact SHA256.
7. Retrieval profile: `fast`.
8. Raw retrieval pool: 20.
9. Protected prefix: 8.
10. Final context cap: 10.
11. Linked candidate cap: 2.
12. Tail replacement margin: 0.05.
13. One recorded pass. Latency is diagnostic only in this audit.

## Arms

The audit uses two arms with the identical graph retrieval and context assembly path:

1. `linked_tail_true`: linked tail with true topology.
2. `linked_tail_removed`: linked tail with relations removed.

The literal graph off arm is excluded because it uses a different raw retrieval depth and cannot
isolate topology.

## Audit data contract

An explicit benchmark only environment gate may expose raw retrieval and linked candidate identity
inside the existing performance diagnostics. The gate must require both
`RECALL_BENCHMARK_PIN=1` and `RECALL_BENCHMARK_GRAPH_AUDIT=1`. Ordinary serving responses must not
contain the audit payload.

For each raw hit, record only chunk identifier, canonical source, ordinal, raw rank, and raw cosine.
For each linked candidate, record the same identity plus whether it entered the final pre trust
context. Do not record rejected candidate text. Existing source authorization and generation
binding remain mandatory.

The analysis normalizes each frozen gold identifier to `recall/<source>:<ordinal>`. It reports:

1. queries missing any gold evidence from direct ranks 1 through 10;
2. queries with missing gold available in raw ranks 9 through 20;
3. queries with a graph connected gold candidate in ranks 9 through 20;
4. queries where true topology promotes a gold candidate;
5. queries where true topology promotes a non gold candidate;
6. paired gold rescues and regressions against removed topology;
7. relation type counts for connected and selected candidates.

## Predictions and decision rules

Prediction 1: at least 5 of the 50 queries have missing gold evidence available in raw ranks 9
through 20. Fewer than 5 means linked tail has too little real headroom, so the next work moves to
base retrieval or chunk selection rather than graph relation authoring.

Prediction 2: current topology connects a gold tail candidate for fewer than half of the headroom
queries. If true, relation coverage is the primary graph bottleneck.

Prediction 3: generic `references` produce at least as many distinct non gold final promotions as
gold final promotions. If true, relation precision is also a bottleneck and unrestricted reference
traversal must not be promoted.

Typed relation authoring proceeds only if at least 5 queries have raw tail gold headroom and at
least 3 remain unconnected by current topology. The pilot must be preregistered separately and must
use relation annotations authored without consulting the gold labels for the evaluated query.

## Required red proof

Add a behavior test for the audit gate. Its invariant is that setting the audit flag alone never
exposes raw candidate identity. The test must fail at the intended assertion when the
`RECALL_BENCHMARK_PIN` requirement is deliberately removed, then pass after restoration.

Add a second test proving that the explicit two flag benchmark gate records ranks 1 through 20 and
marks linked candidates selected into the pre trust context. Its red proof may deliberately disable
the audit payload assignment and must fail on the missing diagnostic value.

## Measurement command

The exact command is implemented with the runner. It must pin the isolated VPS2 source checkout,
certified generation, and source commit, and it must write a new immutable JSON artifact. The audit
must not run while another embedding process is active.
