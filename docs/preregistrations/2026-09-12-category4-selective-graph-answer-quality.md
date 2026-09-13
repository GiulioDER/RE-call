# Preregistration: category 4 selective graph answer quality

Status: protocol locked before answer generation on 2026-09-12.

## Objective

Measure whether the category 4 retrieval improvement from the 0.05 selective graph gate produces
better generated answers and gold citation support than direct retrieval alone. This stage replays
stored contexts and does not rerun retrieval.

## Frozen inputs

* Retrieval artifact: `docs/results/2026-09-12-category4-strict-graph-confirmation.json`.
* Retrieval artifact SHA256: `4E5DB6F5CED6BC368D61F174A19DEB592C587840056CD2882886A02012BAB166`.
* Population: the 841 category 4 questions across all 10 LOCOMO conversations.
* Arms: `baseline` direct ten context items versus `selective_margin_005` with the stored 0.05
  selective graph contexts.
* Contexts: copied exactly from the immutable retrieval artifact. No retrieval, graph expansion,
  gold lookup, or context reordering is performed in this stage.
* Provider: OpenRouter chat completions.
* Model: `deepseek/deepseek-v4-flash`.
* Temperature: 0. Reasoning effort: none. Maximum completion tokens: 512.
* Provider failures remain in the fixed denominator and are recorded per question.

## Primary hypothesis and gate

The 0.05 selective graph arm will improve complete gold citation coverage by at least 2 percentage
points over direct retrieval. The gate passes only when the paired deterministic bootstrap 95%
interval excludes zero, the point estimate reaches 2 points, and valid answer rate declines by no
more than 2 points.

The bootstrap uses 10,000 resamples with seed `20260912` and percentile indices 25 and 9974. Gold
identifiers are used only after generation for scoring.

## Secondary outcomes

Report valid answer rate, nonempty answer rate, any gold citation rate, complete gold citation
coverage, mean gold citation precision, mean gold citation recall, citation validation errors,
provider failures, provider latency, prompt and completion tokens, and paired rescues and
regressions. Report all outcomes for the fixed category 4 population.

If the primary answer gate passes, a separate preregistered blind correctness and citation support
judge will be run before any production promotion decision. The judge will receive no arm labels,
gold identifiers, or retrieval policy names.

## Integrity controls

The answer model receives only the question and the selected stored contexts. The raw answer
artifact is immutable after generation. The answer prompt digest, source revision, model, and input
artifact digest are recorded in the output.

## Reproduction

The runner and category filter were committed in source revision
`83dc89fd7fae20feac276cdbe3df8018f4f964c0`.

```powershell
python scripts/run_voyage4_graph_tail_answer_quality.py docs/results/2026-09-12-category4-strict-graph-confirmation.json docs/results/2026-09-12-category4-selective-graph-answer-quality.json --treatment-arm selective_margin_005 --category 4 --model deepseek/deepseek-v4-flash --workers 8 --source-commit 83dc89fd7fae20feac276cdbe3df8018f4f964c0
```
