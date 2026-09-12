# Voyage 4 bounded graph tail quality

Measured on 2026-09-12 at `2026-09-12T17:57:45.162784Z` using the preregistered protocol
`2026-09-12-voyage4-graph-tail-quality`. The run covered all 10 LOCOMO conversations and 1,536
answerable questions in categories 1 through 4.

Raw artifact: [2026-09-12-voyage4-graph-tail-quality.json](2026-09-12-voyage4-graph-tail-quality.json)

Artifact SHA256: `D51974F72E19AF2E5B978C3183090CA83C8D12BB134E91AF1A2117A2E46C4B07`.

## Protocol

Both arms used Voyage 4, one shared top 20 retrieval result per question, and a context cap of 10.
The baseline retained the top 10 direct results. The treatment protected the top 8 direct results
and appended at most 2 unique retrieval-ranked structural graph neighbors. The direct prefix was
identical in all 1,536 paired questions.

## Primary result

| Metric | Direct 10 | Direct 8 plus graph 2 | Paired delta |
| --- | ---: | ---: | ---: |
| Complete gold evidence coverage | 69.79% | 72.14% | +2.34 points |
| Any gold evidence hit | 82.81% | 84.05% | +1.24 points |
| MRR of first gold item | 0.5713 | 0.5727 | +0.0014 |
| Evidence precision | 10.06% | 10.25% | +0.20 points |

The preregistered primary gate passed. The complete coverage delta was +2.34 points, with a paired
bootstrap 95% interval of +0.78 to +3.97 points. No category declined by more than 3 points.

## Category results

| Category | Questions | Direct 10 | Direct 8 plus graph 2 | Delta |
| --- | ---: | ---: | ---: | ---: |
| cat1 | 282 | 29.79% | 31.21% | +1.42 points |
| cat2-temporal | 321 | 81.31% | 82.55% | +1.25 points |
| cat3 | 92 | 42.39% | 43.48% | +1.09 points |
| cat4 | 841 | 81.81% | 85.02% | +3.21 points |

The effect is positive in every category. Cat4 supplies most of the absolute gain because it is the
largest category. Cat3 remains the weakest category and has the smallest sample size.

## Paired outcomes

The treatment produced 55 complete-evidence rescues and 19 regressions, for a net gain of 36
questions. The other 1,462 questions were unchanged. The graph additions averaged exactly 2 items
per question, and the direct prefix was preserved for every pair.

The converted corpus contained 5,872 conversation-order edges, 5,610 session-date edges, and 5,862
speaker edges. These are structural benchmark relations derived from LOCOMO metadata. This result
does not yet establish that the production semantic graph has the same coverage or causal value.

## Interpretation

Voyage 4 preserves the bounded graph benefit seen with the previous embedder and clears the
preregistered retrieval gate. The gain is modest but broad, while the protected direct prefix avoids
replacing strong direct evidence. This supports a follow-up answer-stage replay using the same fixed
contexts and DeepSeek V4 Flash, measuring citation support, complete citation coverage, unsupported
claims, and answer correctness separately.

This result does not justify changing the production default by itself. The next answer-stage run
must use the immutable contexts in this artifact and the same model configuration for both arms.

## Reproduction

The protocol was committed before measurement in
`docs/preregistrations/2026-09-12-voyage4-graph-tail-quality.md`.

```powershell
python benchmarks/structural_edge_performance.py --dataset locomo --data locomo10.json --embedder voyage:voyage-4 --candidate-k 20 --seed-k 8 --context-k 10 --edge-budget 2 --neighbor-order retrieval --retrieval-k 20 --table voyage4_graph_tail_20260912 --run-id 20260912T000000Z --out docs/results/2026-09-12-voyage4-graph-tail-quality.json
```
