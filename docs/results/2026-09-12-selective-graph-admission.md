# Selective graph admission by retrieval margin

Measured on 2026-09-12 at `2026-09-12T19:07:05.620845Z` using the preregistered protocol
`2026-09-12-selective-graph-admission`. The run covered all 10 LOCOMO conversations and 1,536
answerable questions in categories 1 through 4.

Raw artifact: [2026-09-12-selective-graph-admission.json](2026-09-12-selective-graph-admission.json)

Artifact SHA256: `E3E729A9DD3D936F80245A477FAFEADB95F12A27E164DA1716626DF0DEE6E824`.

## Protocol

All arms used Voyage 4, one shared top 20 retrieval result per question, a context cap of 10, and
all authored structural benchmark relations. The baseline retained direct ranks 1 through 10. The
unfiltered arm protected direct ranks 1 through 8 and appended the two retrieval-ranked graph
candidates. The selective arm protected direct ranks 1 through 8 and admitted a graph candidate
only when its retrieval score was at least 0.05 above the weaker direct rank 9 or 10. Rejected slots
were filled with direct ranks 9 and 10.

## Primary result

| Metric | Direct 10 | Unfiltered 8 plus graph 2 | Selective 8 plus gated graph | Selective delta |
| --- | ---: | ---: | ---: | ---: |
| Complete gold evidence coverage | 69.27% | 71.68% | 71.35% | +2.08 points |
| Any gold evidence hit | 82.88% | 84.11% | 83.98% | +1.11 points |
| MRR of first gold item | 0.5704 | 0.5718 | 0.5716 | +0.0013 |
| Evidence precision | 10.03% | 10.26% | 10.25% | +0.22 points |
| Mean graph additions | 0.00 | 2.00 | 0.80 | -1.20 items |

The preregistered primary gate passed. Selective complete coverage improved by +2.08 points over
baseline. A deterministic 10,000 resample paired bootstrap with seed 20260912 produced a 95%
interval of +1.11 to +3.26 points. No category declined.

The unfiltered arm was slightly higher at +2.41 points, but it made 20 complete evidence
regressions versus 2 for the selective arm. Selective therefore preserves nearly all of the uplift
with substantially lower graph exposure.

## Paired outcomes

Selective graph admission produced 34 complete evidence rescues and 2 regressions. The remaining
1,500 questions were unchanged. Graph candidates were admitted for 805 of 1,536 questions, or
52.41%, with 1,225 total additions. The direct prefix remained protected and each selective context
contained 10 items.

## Category results

| Category | Questions | Direct 10 | Unfiltered 8 plus graph 2 | Selective 8 plus gated graph | Selective delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| cat1 | 282 | 28.37% | 30.50% | 30.14% | +1.77 points |
| cat2-temporal | 321 | 81.00% | 82.56% | 82.56% | +1.56 points |
| cat3 | 92 | 41.30% | 42.39% | 43.48% | +2.17 points |
| cat4 | 841 | 81.57% | 84.54% | 83.95% | +2.38 points |

The selective effect is positive in every category. Category 4 remains the largest contributor,
while category 3 clears the 2 point threshold with the smallest sample.

## Interpretation

The score margin gate is a better controlled graph policy than unconditional tail replacement for
this benchmark. It retains a +2 point retrieval improvement while cutting graph additions by 60%
and reducing complete evidence regressions by 90% relative to the unfiltered arm.

This is still a retrieval result, not proof of factual answer quality. The next measurement should
replay the baseline and selective contexts through DeepSeek V4 Flash and run a blind correctness and
citation entailment judge. The unfiltered arm should remain a retrieval diagnostic rather than a
production candidate.

## Reproduction

The protocol was committed before measurement in
`docs/preregistrations/2026-09-12-selective-graph-admission.md`.

```powershell
python benchmarks/structural_edge_performance.py --dataset locomo --data locomo10.json --embedder voyage:voyage-4 --candidate-k 20 --seed-k 8 --context-k 10 --edge-budget 2 --neighbor-order retrieval --retrieval-k 20 --selective-gate --table selective_graph_admission_20260912 --run-id 20260912T000000Z --out docs/results/2026-09-12-selective-graph-admission.json
```
