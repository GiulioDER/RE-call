# Query-construction follow-up research — 2026-08-27

## Finding from the completed replay

The 138-row replay is not a clean test of the original-model construction arms. With
`deepseek/deepseek-v4-pro`, reasoning effort `medium`, and `max_tokens=1024`, the provider
frequently ended with `finish_reason=length` before emitting the required JSON. The runner
recorded these as model fallbacks, so the recovery numbers remain valid operational outcomes,
but they under-sample the intended construction protocol.

Among rows with a valid frame, the model mostly repeated current task mechanics rather than
describing the governing invariant or known failure mode. Pyramid recovered 1 of 15 frozen
misses; original-loop recovered 0. Neither met the preregistered adoption bar. Graph expansion
was usually gated or produced no eligible relation, and the benchmark artifact previously did
not preserve the graph admission-rejection taxonomy.

## Focused follow-up

The next exploratory run uses the same frozen population and retrieval generation, but:

- `max_tokens=2048`, with the other DeepSeek/OpenRouter settings unchanged;
- the miss class only for the first apparatus check;
- the challenge wording explicitly asks for the governing invariant, known failure mode, or
  decision rule, and warns against merely repeating the immediate action;
- provider finish reason, reasoning-token usage, graph gate reason, and graph admission
  rejection counts retained in the raw artifact.

This run is a budget/prompt apparatus probe, not a replacement for the preregistered result.
If it shows materially higher valid-frame coverage, the complete 15-miss/31-control run must
be run as a separately identified exploratory artifact before making a decision.

## Interpretation guardrails

Model frames remain proposals only. Gold labels are never sent to the model. Only ordinary
trusted retrieval can create evidence, and graph expansion runs only after newly trusted seed
chunks. A provider failure is reported separately from a retrieval miss.
