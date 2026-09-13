# Preregistration: category 4 strict selective graph confirmation

Status: protocol locked before measurement on 2026-09-12.

## Objective

Confirm whether the selective graph policy has a durable retrieval benefit on the category 4 slice
while testing whether a stricter score margin reduces the remaining graph regressions. This is a
retrieval confirmation. Answer generation is a separate follow-up only if the retrieval gate passes.

## Frozen inputs

* Dataset: `locomo10.json`.
* Population: all 841 answerable LOCOMO category 4 questions across all 10 conversations.
* Embedder: `voyage:voyage-4`.
* Candidate pool: 20.
* One shared retrieval result per question with retrieval depth 20.
* Context cap: 10.
* Graph relations: all authored structural benchmark relations.
* Graph candidates: unique, source-backed, one-hop neighbors ranked by their existing retrieval
  score.
* Source revision: `c15f9d89ea85fb48b0d48c5206b180ce27cedeb5`.

## Arms

1. `baseline`: direct retrieval ranks 1 through 10.
2. `selective_margin_005`: activate only on category 4, protect direct ranks 1 through 8, admit
   graph candidates whose score is at least 0.05 above the weaker direct rank 9 or 10, and fill
   rejected slots with direct ranks 9 and 10.
3. `selective_margin_strict`: identical to the existing selective arm, except the score margin is
   0.10.

Both selective arms keep ten context items and preserve the direct prefix. No gold label, answer,
or category other than the fixed activation category is used for candidate selection.

## Primary hypothesis and gate

The strict selective arm will improve complete gold evidence coverage over baseline by at least 2
percentage points. The paired bootstrap 95 percent interval must exclude zero. The strict arm must
also have no more than 8 complete evidence regressions against baseline, which is the preregistered
one percent regression guardrail rounded down for 841 questions.

If the strict arm fails the regression guardrail or the primary uplift gate, it is not a promotion
candidate regardless of secondary metrics. The existing 0.05 arm is a diagnostic comparator.

## Secondary outcomes

Report any gold hit, MRR, evidence precision, graph admission rate, mean additions, direct prefix
identity, paired rescues and regressions, and category 4 results for both margins. Compare the
strict and existing selective arms directly. Record every admitted graph item with its relation and
retrieval score.

## Integrity controls

All arms share the same retrieval result per question. Missing graph candidate scores cannot pass a
selective gate. The raw artifact is immutable after generation. No answer model or independent
judge runs in this stage.

## Reproduction

```powershell
python benchmarks/structural_edge_performance.py --dataset locomo --data locomo10.json --embedder voyage:voyage-4 --candidate-k 20 --seed-k 8 --context-k 10 --edge-budget 2 --neighbor-order retrieval --retrieval-k 20 --selective-gate --selective-margin 0.10 --selective-category 4 --table category4_strict_graph_20260912 --run-id 20260912T000000Z --out docs/results/2026-09-12-category4-strict-graph-confirmation.json
```
