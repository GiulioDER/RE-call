# C8 routed specialists with grounded graph, official smoke preparation

## Frozen prediction

Before an official AML Smoke is submitted, configure RE-call with
`C8_routed_specialists_grounded_graph`. Coding requests must route to Voyage Code4. Add must use
the bounded source anchored `openai/gpt-4o-mini` compiler to create grounded graph relations, and
Search must attempt the graph sidecar while returning raw evidence only. The isolated AML
compatibility check must show an authored eligible relation and a graph relation hit without a
graph fallback. This is preparation evidence only and does not authorize submission of an official
Smoke or Full run.

## Atomic rescue activation contract

The C8 service contains the optional generation-bound atomic rescue adapter. It is activated only
when `RECALL_ATOMIC_RESCUE_MODE=active` and the configured artifact root contains a valid manifest
for the exact served generation. The manifest must match the served generation id, corpus SHA-256,
Code4 embedding profile and dimension, plus the configured AML calibration id and pipeline
fingerprint. A missing, malformed, stale, or mismatched artifact preserves ordinary hosted
retrieval and reports an internal fallback. An official AML corpus is mutable during Add, so this
contract does not authorize claiming active rescue until a generation-specific artifact has been
built after the benchmark corpus is frozen.

## OpenRouter graph compatibility check

Before any official AML Smoke, run the isolated C8 graph verifier against the VPS2 service. It
must observe C8 with `graph_sidecar=true`, provider `openrouter`, model `openai/gpt-4o-mini`, a
successful live Add with no compiler fallback and at least one compiled record, at least one
eligible persisted grounded relation, and a Search response with graph attempted, no graph
fallback, and at least one relation hit. The verifier creates then deletes one fresh tenant. A
failed check blocks official Smoke authorization.
