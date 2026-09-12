# Preregistration: selective graph admission by retrieval margin

Status: protocol locked before measurement on 2026-09-12.

## Objective

Test whether a calibrated score gate can keep the useful part of the bounded graph tail while
reducing graph additions that do not improve gold retrieval. The gate is based only on retrieval
scores and graph reachability. Gold labels are not supplied during selection.

## Frozen inputs

* Dataset: `locomo10.json`.
* Population: all 1,536 answerable questions in LOCOMO categories 1 through 4.
* Embedder: `voyage:voyage-4`.
* Candidate pool: 20.
* One shared retrieval result per question with retrieval depth 20.
* Context cap: 10.
* Graph relations: all authored structural benchmark relations.
* Graph candidates: unique, source-backed, one-hop neighbors ranked by their existing retrieval
  score.
* Source revision: `7f0d06b60776277596c7a5646fc61645f1408ec8`.

## Arms

1. `baseline`: direct retrieval ranks 1 through 10.
2. `unfiltered_tail`: protect direct ranks 1 through 8 and append up to 2 retrieval-ranked graph
   candidates, matching the leading prior configuration.
3. `selective_tail`: protect direct ranks 1 through 8. Admit a graph candidate only when its
   retrieval score is at least 0.05 greater than the weaker of direct ranks 9 and 10. Fill any
   unused slots with direct ranks 9 and 10 in retrieval order. The final context therefore remains
   capped at 10 items.

The selective gate is fixed before looking at this run's outcomes. The 0.05 margin is inherited
from the previously preregistered calibrated tail replacement protocol and is not tuned on this
query set.

## Primary hypothesis and gate

The selective arm will improve complete gold evidence coverage over `baseline` by at least 2
percentage points. The paired bootstrap 95 percent interval for the selective minus baseline delta
must exclude zero, and no category may decline by more than 3 points. The unfiltered arm is a
diagnostic comparator, not a promotion requirement.

## Secondary outcomes

Report any gold hit, MRR, evidence precision, mean context size, graph admission rate, graph
candidate scores, direct prefix identity, paired rescues and regressions, and category breakdowns.
Compare selective against unfiltered for complete coverage and graph additions. Record every
addition with its structural relation and retrieval score.

If the selective retrieval gate passes, the resulting immutable contexts will be preregistered for
a separate answer replay with the same model and a blind correctness and citation entailment judge.
No answer quality conclusion is drawn from this retrieval run.

## Integrity controls

The direct prefix is identical across all arms. Selection uses no gold evidence, answer text, or
category-specific activation. Missing retrieval scores cannot pass the selective gate. The raw
artifact is immutable after generation.

## Reproduction

```powershell
python benchmarks/structural_edge_performance.py --dataset locomo --data locomo10.json --embedder voyage:voyage-4 --candidate-k 20 --seed-k 8 --context-k 10 --edge-budget 2 --neighbor-order retrieval --retrieval-k 20 --selective-gate --table selective_graph_admission_20260912 --run-id 20260912T000000Z --out docs/results/2026-09-12-selective-graph-admission.json
```
