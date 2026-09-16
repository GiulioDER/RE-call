# Atomic fact blind source census completion ceiling correction

Status: preregistered after the 120 token run stopped, before any corrected generation call.

## Correction

The first census attempt stopped before returning one usable completion because
`deepseek/deepseek-v4-flash` reached the frozen 120 completion token ceiling. No question, pool,
or retrieval result was produced.

Run a new protocol with maximum 2,048 completion tokens. This is the only changed generation
parameter. The model, temperature, 45 candidate sources, candidate order, manifest SHA256,
prompts, one call per source, validation rules, privacy checks, aggregate reporting, and every
construction gate remain exactly as frozen in
`2026-09-16-atomic-fact-blind-source-census.md`.

The corrected protocol identifier is
`2026-09-16-atomic-fact-blind-source-census-ceiling-correction`. The private pool and public result
must record the corrected identifier and 2,048 token ceiling.

The failed call did not yield content, so it cannot be accepted, reused, or inspected. The
corrected run starts from the first candidate and must still account for exactly 45 usable provider
completions. A second truncation or any other provider exception stops the corrected protocol.

## Prediction

I predict the 2,048 token ceiling eliminates truncation and the unchanged construction reaches
`READY_TO_PREREGISTER_EXPLORATORY_RETRIEVAL` with at least 36 accepted questions. The ceiling is
more than 17 times the failed limit while remaining one eighth of the repository's general
benchmark default.
