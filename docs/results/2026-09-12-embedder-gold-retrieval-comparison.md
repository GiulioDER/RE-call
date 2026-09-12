# Voyage 4 versus Voyage 3 gold retrieval

Measured on 2026-09-12 at `2026-09-12T17:44:42Z` using the preregistered protocol
`2026-09-12-embedder-gold-retrieval-comparison`. The experiment used the frozen LOCOMO dataset,
10 conversations, 1,536 answerable questions, `candidate_k=20`, and retrieval depths 1, 3, 5,
10, and 20. Category 5 adversarial questions were excluded from gold retrieval scoring.

Raw artifact: [2026-09-12-embedder-gold-retrieval-comparison.json](2026-09-12-embedder-gold-retrieval-comparison.json)

Artifact SHA256: `47137C5D6F0FA46D9562CE87109E6F09C4304EA0DEEDE4B01E308123A0A7391E`.

## Primary result

The treatment was `voyage:voyage-4`; the control was `voyage:voyage-3`.

| Metric | Voyage 3 | Voyage 4 | Paired delta | Bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Gold hit@5 | 68.03% | 73.50% | +5.47 points | [+3.26, +7.75] |
| Gold hit@10 | 79.10% | 83.14% | +4.04 points | [+2.21, +5.99] |
| Gold hit@20 | 85.81% | 88.35% | +2.54 points | [+0.78, +4.23] |
| MRR of first gold turn | 0.6242 | 0.6519 | +0.0277 | not preregistered for interval |

The preregistered primary gate passed. Voyage 4 improved hit@5 by more than 2 points, the paired
interval excluded zero, and no category declined.

## Category results

| Category | Questions | Voyage 3 hit@5 | Voyage 4 hit@5 | Delta |
| --- | ---: | ---: | ---: | ---: |
| cat1 | 282 | 63.12% | 70.92% | +7.80 points |
| cat2-temporal | 321 | 74.14% | 79.13% | +4.98 points |
| cat3 | 92 | 44.57% | 51.09% | +6.52 points |
| cat4 | 841 | 69.92% | 74.67% | +4.76 points |

The improvement is broad rather than isolated to the multi hop or temporal slice. Cat3 remains the
weakest category after the improvement and has the smallest sample size.

## Miss attribution

The first depth at which any gold turn appeared was recorded per question.

| First gold depth | Voyage 3 | Voyage 4 | Change |
| --- | ---: | ---: | ---: |
| 1 through 5 | 1,045 | 1,129 | +84 |
| 6 through 10 | 170 | 148 | -22 |
| 11 through 20 | 103 | 80 | -23 |
| absent by 20 | 218 | 179 | -39 |

Voyage 4 moved 84 questions into the top five, reduced the candidate recall failures beyond depth
20 by 39, and reduced the number of questions whose first gold turn appeared only in the tail.

Paired hit@5 outcomes were 119 rescues and 35 regressions, for a net gain of 84 questions.

## Interpretation

The measured bottleneck was candidate generation, and the stronger embedder improved both shallow
ranking and the candidate recall ceiling. This result is materially stronger than the recent graph
one hop replay, which did not improve answer quality and increased prompt tokens. The next quality
experiment should therefore test Voyage 4 as the retrieval base for bounded graph tail competition,
starting with the established 8 direct plus 2 graph configuration. That must be a new preregistered
paired run with graph rescues, regressions, and category results.

This experiment measured gold evidence retrieval only. It makes no claim about generated answer
correctness, citation entailment, or production rollout safety.

## Reproduction

The protocol was committed before measurement in
`docs/preregistrations/2026-09-12-embedder-gold-retrieval-comparison.md`.

```powershell
scp scripts/run_locomo_embedder_comparison.py vps2:/tmp/run_locomo_embedder_comparison.py
ssh vps2 "set -a; . /home/sentiment/recall-repos/.env; set +a; cd /home/sentiment/recall-repos/serving; RECALL_SOURCE_COMMIT=6fec92f1c0bcac1d29e704c9d2ceda84f90e6cf8 /home/sentiment/recall-repos/.venv/bin/python /tmp/run_locomo_embedder_comparison.py --data locomo10.json --out /tmp/2026-09-12-embedder-gold-retrieval-comparison.json"
scp vps2:/tmp/2026-09-12-embedder-gold-retrieval-comparison.json docs/results/2026-09-12-embedder-gold-retrieval-comparison.json
```
