# Graph readiness optimization smoke, 2026-09-10

This is a post deployment smoke check for commit `1725ba468b4ea1a7a236b35eee609efce9712936`.
It is not the preregistered performance remeasurement.

## Change

`GenerationStore.graph_readiness()` now reads only the immutable semantic graph marker from
`recall_generations.validation_summary`. It no longer loads entity, mention, relation, or evidence
rows for every readiness check. The projection loader still compares the loaded graph fingerprint,
graph identity, tenant, and generation with the marker before caching it.

## Deployment parity

1. Serving checkout: `1725ba468b4ea1a7a236b35eee609efce9712936`
2. Imported `recall_mcp/service.py` SHA256:
   `38bf79d0daf1d2e621d600763d1b77dc9a2bfcd7e8cfb1cb0f828d5961f403d0`
3. Schema: current `0024`, required `0024`
4. MCP handshake: 22 tools
5. Pinned generation: `gen_b02a44a99917424ba3bd8011280e3712`

## Live smoke

Command:

```powershell
.venv/Scripts/python.exe scripts/run_live_tty_graph_precision.py `
  --limit 17 --variant combined --control none --timeout 240 `
  --output $env:TEMP/recall-live-graph-readiness-optimization-smoke-20260910.json
```

The frozen query prefix produced 34 successful MCP calls. All 17 one hop calls reported
`graph_readiness=ready`. The compact readiness span was 5.899 ms median and 23.573 ms maximum in
this smoke sample. These are smoke observations, not a p95 gate.

The selective gate refused 14 calls with `graph_gate_not_met`. The positive query `should I propose
paid API runs for recall` discovered 8 candidates and admitted 2 `references` candidates. Its policy
fingerprint was `bd95d38fc1603f7f591096172bfd69a576ae99340afd1bda0d593a6b852e1854`.

Artifact:

`C:\Users\gde00\AppData\Local\Temp\recall-live-graph-readiness-optimization-smoke-20260910.json`

## Verification

Focused tests passed: 44 tests. Ruff passed for the changed files. The readiness mutation proof was
red when the old full graph load was restored, with the failure reason `full semantic graph load is
forbidden during readiness`, then green after restoring the compact marker path.
