# Structural edge production default rollout

Applied 2026-09-11 after the full LoCoMo external judge evaluation.

The public reasoning default `graph_expansion=auto` now selects bounded global `one_hop` expansion
for every nonempty query. The explicit `off` and `one_hop` overrides remain available. The serving
path protects eight direct retrieval items and fills at most the remaining two context slots with
structurally related candidates, preserving the ten item evidence cap and the ordinary trust
evaluation boundary.

The decision was based on the fixed OpenRouter judge evaluation in
`docs/results/2026-09-11-structural-edge-external-judge.md`:

| arm | all row score | paired delta |
|---|---:|---:|
| baseline | 39.13% | reference |
| global 8 direct + 2 structural | 41.41% | +2.2786 points |
| category selective 8 direct + 2 structural | 41.60% | +2.4740 points |

The global arm passed the preregistered two point answer gate. I selected it as the production
default because it applies the winning budget consistently across query shapes. Category selective
remains represented by the earlier evaluation artifact but is not the default route.

## Implementation

* `recall/query_class.py` uses `one_hop` for nonempty automatic queries and versions the activation
  policy as `graph-activation-v2-global-one-hop`.
* `recall_mcp/server.py`, `recall/cli_commands/reasoning_cmd.py`, `docs/API.md`,
  `docs/REASONING_GRAPH.md`, and `docs/REASONING_OPERATIONS.md` describe the new default.
* Explicit `graph_expansion=off` still disables expansion for callers that need the baseline.

## Verification

The consumer test was first run red against the old category selective implementation: three
nonselected query cases failed the intended graph policy assertion. After the route change:

```text
python -m pytest tests/test_query_class.py tests/test_reasoning_api.py -q
57 passed
```
