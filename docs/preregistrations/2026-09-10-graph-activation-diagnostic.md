# Pre registration: graph activation diagnostic

Date: 2026-09-10

Status: locked before live measurement

## Purpose

The production quality run showed that the combined selective gate rarely starts traversal.
This diagnostic asks whether directional semantic graph traversal can improve evidence quality when
the early selective refusal is removed. It is a mechanism diagnostic, not a production promotion
test.

## Frozen inputs

1. Query file: `docs/preregistrations/2026-09-10-graph-activation-diagnostic-queries.json`.
2. Query count: 5 content based relation probes.
3. Query file SHA256: `f4812cb0681234f5f569aac6b77a1dd297d7c0386a6eb424c50d38edc75595ac`.
4. Each expected id is a chunk from the referenced object document in the active graph snapshot.
5. Tenant: `memory`.
6. Embedder: `voyage:voyage-4`.
7. Retrieval profile: `fast`, `k=5`.
8. Generation: `gen_b02a44a99917424ba3bd8011280e3712`.
9. Reasoning budget: `max_steps=12`, `max_graph_nodes=32`, `max_evidence_tokens=2048`.
10. Graph policy variant: `directional`. This retains directional authored traversal and removes
    the selective, hub, corroboration, and cosine admission filters for diagnosis only.

## Arms and protocol

1. `off`: no graph expansion.
2. `one_hop`: graph expansion with the frozen `directional` diagnostic variant.
3. Start a fresh server process per arm.
4. Run one unscored warmup pass, then five recorded passes in frozen order.
5. Preserve every response, graph diagnostic, refusal, trust state, evidence list, and latency.

## Diagnostic gate

The mechanism is considered quality positive only if both conditions hold:

1. At least two of five probes produce an admitted graph candidate in the one hop arm.
2. One hop improves mean object document hit@5 by at least 0.10 absolute versus off, without
   changing trust state or introducing a refusal on a query that off serves.

Failure of this diagnostic does not prove that the production combined policy cannot help. Passing
it does not authorize changing the production policy. It identifies whether the current quality
null is caused by no activation or by ineffective candidate ranking and trust admission.

## Analysis

An evidence hit occurs when a trusted top five item has a chunk id in the frozen expected list.
Report query level hit@5, MRR, trusted evidence changes, candidate discovery, candidate admission,
new trusted evidence, graph gate reason, trust state, refusal reason, and paired latency. Keep all
five passes and all refusals in the denominator. Report the per query table before the conclusion.

Because the answer provider is disabled, answer correctness and unsupported claim rate are not
estimable.

## Reproduction command

```powershell
python scripts/run_live_graph_performance_attribution.py --query-set docs/preregistrations/2026-09-10-graph-activation-diagnostic-queries.json --output C:\Users\gde00\AppData\Local\Temp\recall-live-graph-activation-diagnostic-20260910.json --generation-id gen_b02a44a99917424ba3bd8011280e3712 --passes 5 --warmup-passes 1 --variant directional --control none
```

This preregistration is not edited after the first live request. Results belong in a separate
report.
