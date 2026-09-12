# Preregistration: selective graph answer quality

Status: protocol locked before answer generation on 2026-09-12.

## Objective

Measure whether the selective graph admission policy converts its retrieval gain into better
generated answers and gold citation support than direct retrieval alone. Retrieval is not rerun in
this stage. Contexts are copied from the immutable selective admission artifact.

## Frozen inputs

* Retrieval artifact: `docs/results/2026-09-12-selective-graph-admission.json`.
* Retrieval artifact SHA256: `E3E729A9DD3D936F80245A477FAFEADB95F12A27E164DA1716626DF0DEE6E824`.
* Population: all 1,536 paired answerable questions in LOCOMO categories 1 through 4.
* Arms: `baseline` direct 10 versus `selective_tail` direct 8 plus selectively admitted graph
  context, with ordering and context items copied exactly from the retrieval artifact.
* Provider: OpenRouter chat completions.
* Model: `deepseek/deepseek-v4-flash`.
* Temperature: 0. Reasoning effort: none. Maximum completion tokens: 512.
* One answer request per arm and question. Provider failures remain in the fixed denominator.

## Primary hypothesis and gate

The selective graph arm will improve complete gold citation coverage by at least 2 percentage points
over direct 10. The gate passes only when the paired bootstrap 95 percent interval excludes zero,
the point estimate reaches 2 points, and valid answer rate declines by no more than 2 points.

## Secondary outcomes

Report valid answer rate, nonempty answer rate, any gold citation rate, mean gold citation
precision, mean gold citation recall, citation identity validity, provider failures, provider
latency, token usage, paired rescues and regressions, and category breakdowns. This stage does not
claim factual correctness because no independent judge is run.

If the answer gate passes, a separate blind correctness and citation entailment judge will be
preregistered over graph rescues, regressions, and a matched unchanged sample. The judge will not
receive arm labels or retrieval policy names.

## Integrity controls

The question and selected contexts are the only task data supplied to the answer model. Gold
evidence identifiers are used only after generation for scoring. The raw answer artifact is
immutable after the run.

## Reproduction

```powershell
python scripts/run_voyage4_graph_tail_answer_quality.py docs/results/2026-09-12-selective-graph-admission.json docs/results/2026-09-12-selective-graph-answer-quality.json --treatment-arm selective_tail --model deepseek/deepseek-v4-flash --workers 8 --source-commit 52b264b061b756f2ef0a9d47ac49351c05553840
```
