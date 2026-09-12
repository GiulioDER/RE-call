# Preregistration: current embedder versus Voyage 3 for gold retrieval

Status: protocol locked before measurement on 2026-09-12.

## Objective

Measure whether the current Voyage 4 embedder improves gold evidence retrieval over the legacy
Voyage 3 embedder on the frozen LOCOMO benchmark. This is a retrieval only experiment. It does not
measure generated answer quality, graph expansion, reranking, latency, or provider cost.

## Frozen inputs

* Dataset: `locomo10.json`, 10 conversations and 1,986 source questions.
* Dataset SHA256: `79FA87E90F04081343B8C8DEBECB80A9A6842B76A7AA537DC9FDF651EA698FF4`.
* Source commit: `6b3a23f0d716b6214bf3897d336e04d6683b4286`.
* Retrieval: RE-call `recall.eval.locomo`, same corpus conversion, trust policy, table isolation,
  `k=5`, `candidate_k=20`, and depth curve `1,3,5,10,20` for both arms.
* Arms: `voyage:voyage-4` as the current arm and `voyage:voyage-3` as the comparison arm.
* Gold labels: LOCOMO answerable categories 1 through 4, using exact evidence turn identifiers.
  Category 5 remains reported only as adversarial abstention and is excluded from gold retrieval.

## Primary hypothesis and gate

The current Voyage 4 arm will improve pooled gold evidence-turn hit@5 by at least 2 percentage
points over Voyage 3, with no answerable category declining by more than 3 points. The gate is
passed only when the paired bootstrap 95 percent interval for the hit@5 difference excludes zero
and the point estimate meets the 2 point threshold.

## Miss attribution

For each answerable question and arm, record the first depth at which any gold evidence turn is
retrieved: ranks 1 through 5, 6 through 10, 11 through 20, or not present by 20. A miss beyond 20
is classified as candidate recall failure for this benchmark configuration. A gold turn found by
20 but not by 5 is classified as ranking or truncation loss. This attribution is descriptive and
does not use gold labels to change retrieval.

## Secondary outcomes

Report pooled and per-category gold item recall, question hit rate, complete evidence coverage,
MRR of the first gold turn, depth curves, paired rescues and regressions, and the miss attribution
counts above. Report the category names as `cat1`, `cat2-temporal`, `cat3`, and `cat4`.

## Reproduction commands

```powershell
python -m recall.eval.locomo --data locomo10.json --embedder voyage:voyage-4 --k 5 --k-curve 1,3,5,10,20 --candidate-k 20 --out docs/results/2026-09-12-voyage4-gold-retrieval.json
python -m recall.eval.locomo --data locomo10.json --embedder voyage:voyage-3 --k 5 --k-curve 1,3,5,10,20 --candidate-k 20 --out docs/results/2026-09-12-voyage3-gold-retrieval.json
```

The runs must use fresh isolated tables or tenants and must not reuse a corpus indexed by another
embedder. The result report and an analysis artifact will be committed after both runs complete.
