# Preregistration: Voyage 4 bounded graph tail answer quality

Status: protocol locked before answer generation on 2026-09-12.

## Objective

Measure whether the Voyage 4 8 direct plus 2 structural graph contexts improve generated answer
quality and gold citation support over the matching Voyage 4 direct 10 contexts. Retrieval is not
rerun in this stage. The exact immutable contexts come from the completed retrieval artifact.

## Frozen inputs

* Retrieval artifact: `docs/results/2026-09-12-voyage4-graph-tail-quality.json`.
* Retrieval artifact SHA256: `D51974F72E19AF2E5B978C3183090CA83C8D12BB134E91AF1A2117A2E46C4B07`.
* Dataset: the same LOCOMO source used by the retrieval artifact, categories 1 through 4 only.
* Population: all 1,536 paired answerable questions.
* Arms: direct 10 versus direct 8 plus graph 2, with contexts and ordering copied exactly from the
  retrieval artifact.
* Provider: OpenRouter chat completions.
* Model: `deepseek/deepseek-v4-flash`.
* Temperature: 0. Reasoning effort: none. Maximum completion tokens: 512.
* One answer request per arm and question. Provider failures are recorded and do not silently
  change the denominator.

## Primary hypothesis and gate

The graph treatment will improve complete gold citation coverage by at least 2 percentage points
over direct 10, with no decline in valid answer rate greater than 2 points. The gate is passed only
when the paired bootstrap 95 percent interval for complete gold citation coverage excludes zero
and the point estimate meets the 2 point threshold.

## Secondary outcomes

Report valid answer rate, nonempty answer rate, any gold citation rate, mean gold citation precision,
mean gold citation recall, citation identity validity, provider failure rate, paired citation rescues
and regressions, and category breakdowns. This stage does not claim factual correctness because no
independent judge is run.

## Integrity controls

The answer prompt receives only the selected context items and the user question. Gold evidence
identifiers are used after generation for scoring and are never supplied to the model. The raw
answer artifact is immutable after the run.

## Reproduction command

```powershell
python scripts/run_voyage4_graph_tail_answer_quality.py docs/results/2026-09-12-voyage4-graph-tail-quality.json docs/results/2026-09-12-voyage4-graph-tail-answer-quality.json --model deepseek/deepseek-v4-flash --workers 8 --source-commit 09327afb14c3cb2b68bb1b0a54ace27eb9ff0a2d
```
