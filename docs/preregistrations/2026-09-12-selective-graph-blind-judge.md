# Preregistration: selective graph blind answer judge

Status: protocol locked before judge generation on 2026-09-12.

## Objective

Determine whether the selective graph answer changes are factually better supported and more
complete, rather than merely citing more gold identifiers. The judge compares stored baseline and
selective answers from the completed answer replay.

## Frozen inputs and sample

* Answer artifact: `docs/results/2026-09-12-selective-graph-answer-quality.json`.
* Answer artifact SHA256: `94B5228D0025F3D3AC9A18C2C36067CD5334EB7ADC0CDD0DB7739491E35B277D`.
* Retrieval artifact: `docs/results/2026-09-12-selective-graph-admission.json`.
* Retrieval artifact SHA256: `E3E729A9DD3D936F80245A477FAFEADB95F12A27E164DA1716626DF0DEE6E824`.
* Changed pairs: every complete citation rescue and every complete citation regression in the
  answer artifact, expected to be 85 rescues and 53 regressions.
* Unchanged control: 138 pairs sampled from all other answer pairs using `random.Random(20260912)`
  after sorting by question id and shuffling.
* Total sample: 276 paired questions.

The judge receives the question, the union of both arms' evidence contexts, and two answers labeled
only A and B. A and B are shuffled deterministically per question. Arm identities are removed from
the judge prompt and are restored only after scoring.

## Judge and rubric

* Provider: OpenRouter chat completions.
* Judge model: `anthropic/claude-haiku-4.5`.
* Temperature: 0. Reasoning effort: none. Maximum completion tokens: 256.
* One judge request per paired question.

For each answer, the judge assigns integer scores from 0 through 2 for correctness, evidence
support, and completeness. The judge also counts unsupported factual claims as a nonnegative
integer. A total answer score is the sum of correctness, support, and completeness, ranging from 0
through 6.

## Primary hypothesis and gate

The selective answer will have a higher mean total score than baseline. The gate requires all of
the following:

1. At least 95% of the 276 judge requests produce valid rubric JSON.
2. Selective minus baseline mean total score is at least 0.25 points over valid judgments.
3. A deterministic paired bootstrap 95% interval with seed 20260912 excludes zero.

If the valid judgment rate is below 95%, the result is inconclusive regardless of the point
estimate.

## Secondary outcomes

Report total score by rescue, regression, and unchanged bucket; correctness, support, and
completeness components; unsupported claim delta; category breakdown; judge failures; provider
latency; and token usage. The judge result is not used to alter the retrieval or answer artifacts.

## Integrity controls

The judge does not receive gold answers, gold identifiers, retrieval policy names, or the answer
artifact's arm labels. The combined evidence is provided only as reference material. The raw judge
artifact is immutable after generation.

## Reproduction

The judge implementation was committed before generation in source revision
`13a6fd3b32bc338adbfa5adb0b94e05a69d4bc64`.

```powershell
python scripts/run_selective_graph_blind_judge.py docs/results/2026-09-12-selective-graph-answer-quality.json docs/results/2026-09-12-selective-graph-admission.json docs/results/2026-09-12-selective-graph-blind-judge.json --workers 8 --source-commit 13a6fd3b32bc338adbfa5adb0b94e05a69d4bc64
```
