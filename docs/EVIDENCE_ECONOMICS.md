# Evidence economics and routing

RE-call reports three separate families of results.

1. Retrieval quality covers recall, nDCG, answer accuracy, citation support, and abstention.
2. Evidence economics covers exact reader tokenizer counts for the rendered evidence and total
   generator input. It is measured with the pinned `cl100k_base` tokenizer and is not a provider
   billing claim.
3. Operations covers staged indexing, snapshot loading, startup, cutover, recovery, and memory.

Each benchmark question records its deterministic query class and fixed routing decision. The
classifier does not inspect gold answers, corpus content, or generated answers.

The public serving path keeps routing, related evidence, and explanations opt in until the
promotion gates in `benchmarks/PREREGISTRATION-evidence-routing.md` are met.

## Reproducible benchmark runs

The LOCOMO harness accepts `--evidence-budget` for one budgeted paired run and
`--routing-mode shadow` or `--routing-mode active` for the routing arm. The fixed ladder is
`128, 256, 512, 1024, 2048, 4096, 8192`. Each budgeted outcome records its applied budget,
exact rendered evidence tokens, total input tokens, query class, routing profile, and raw question
record. A complete curve is assembled by combining paired artifacts at the same question ids.

Operational measurements use `benchmarks.operational.run_operational_benchmark`. Its artifact has
claim family `operational` and cannot be accepted as retrieval quality evidence. Supply callbacks
for lexical readiness, semantic readiness, snapshot loading, warmup, atomic cutover, and recovery.
