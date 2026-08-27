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

The miss-only exploratory run uses the same frozen population and retrieval generation, but:

- `max_tokens=2048`, with the other DeepSeek/OpenRouter settings unchanged;
- the miss class only for the first apparatus check;
- the local challenge wording is staged to explicitly ask for the governing invariant, known
  failure mode, or decision rule, and warn against merely repeating the immediate action. The
  first VPS2-backed budget probe used the already deployed challenge wording; the wording change
  must be deployed before it can be credited as an experimental factor;
- provider finish reason, reasoning-token usage, graph gate reason, and graph admission
  rejection counts retained in the raw artifact.

This run is a budget/prompt apparatus probe, not a replacement for the preregistered result.
If it shows materially higher valid-frame coverage, the complete 15-miss/31-control run must
be run as a separately identified exploratory artifact before making a decision.

## Exploratory result

The complete 2,048-token run finished with 138 rows. Baseline recovered 17/46 overall, including
17/31 controls and 0/15 misses. Original-loop recovered 23/46, including 22/31 controls and
1/15 misses; it produced valid model frames on 45/46 rows, with one provider truncation.
Pyramid recovered 27/46, including 23/31 controls and 4/15 misses; all 46 model calls completed.

The higher budget clearly improved apparatus coverage and recovery, but neither arm met the
registered adoption rule: pyramid requires at least 5/15 miss rescues and 30/31 controls.
Pyramid is promising enough for a prompt-factor follow-up after deployment, but this result does
not justify promotion and does not isolate the staged prompt wording change.

## Interpretation guardrails

Model frames remain proposals only. Gold labels are never sent to the model. Only ordinary
trusted retrieval can create evidence, and graph expansion runs only after newly trusted seed
chunks. A provider failure is reported separately from a retrieval miss.
