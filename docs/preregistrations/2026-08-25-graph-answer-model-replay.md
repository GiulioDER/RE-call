# Pre registration: graph answer model replay

Status: preregistered before the DeepSeek measurement.

## Objective

Test whether a stronger answer model makes better use of the evidence returned by the verified
Evidence Graph V1 run. This is an answer quality test, not a claim that the answer model changes
graph recovery. The graph and retrieval payloads are frozen before model calls.

## Frozen input and baseline

The frozen input is `results/live_tty_answer_ab_graphfix_20260825.json`, containing 50 queries in
the `off` arm and the same 50 queries in the `one_hop` arm. The existing baseline is the paired
`qwen3:4b` artifact `results/deep_answer_graphfix_20260825.json`. Both arms use the same serialized
reasoning responses, query set, tenant, generation, calibration, retrieval profile, and evidence
prompt renderer.

The new model is OpenRouter `deepseek/deepseek-v4-pro`, with temperature 0, reasoning effort
`medium`, JSON object response format, and a 1024 token output ceiling. The returned model id and
usage metadata are retained because the OpenRouter model name is a moving alias.

## Predictions

1. On the 22 answerable queries, DeepSeek will improve human judged supported answer correctness by
   at least 10 percentage points over `qwen3:4b` in at least one arm.
2. On queries where `one_hop` adds evidence, DeepSeek will have a nonnegative graph-use delta in
   supported answer correctness, and its unsupported claim rate will not increase by more than 5
   percentage points versus its own `off` arm.
3. Evidence recall, evidence precision, trusted item count, and graph diagnostics will be exactly
   invariant between models. Any difference is an apparatus failure because the model receives a
   frozen serialized reasoning response.

## Outcomes

The primary outcome is blind human adjudication of each answerable row as `correct_supported`,
`incorrect_supported`, `unsupported_claim`, or `abstained`. A row is `correct_supported` only when
the answer is correct and every material claim is supported by the cited evidence.

Secondary outcomes are structural answer validation rate, citation validity, answer change rate,
graph use delta between `off` and `one_hop`, latency, token usage, and provider failures.

The existing graph run's evidence recall remains a retrieval outcome. It is reported alongside the
answer results, but it is not attributed to the answer model.

## Analysis rules

The comparison is paired by query and arm. No model judge is used. Human labels are added after the
raw artifacts are retained, and the prediction section is never edited after measurement. Missing
or uncertified provider metadata, changed evidence, malformed output, or a missing query is reported
as a failed apparatus check rather than scored as a model result.

The result is not a final causal graph quality claim because this replay does not add the shuffled
relation and removed relation controls from the graph preregistration.
