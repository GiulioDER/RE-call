# Category 4 strict selective graph confirmation

Measured on 2026-09-12 at `2026-09-12T20:03:45.208926+00:00` on VPS2.
The raw artifact contains 1,536 aligned answerable cases. The preregistered analysis population is
the 841 category 4 questions across all 10 LOCOMO conversations.

Raw artifact: [2026-09-12-category4-strict-graph-confirmation.json](2026-09-12-category4-strict-graph-confirmation.json)

Artifact SHA256: `4E5DB6F5CED6BC368D61F174A19DEB592C587840056CD2882886A02012BAB166`.

Source revision: `c15f9d89ea85fb48b0d48c5206b180ce27cedeb5`.

Preregistration: [2026-09-12-category4-strict-graph-confirmation.md](../preregistrations/2026-09-12-category4-strict-graph-confirmation.md)

## Category 4 result

| Metric | Baseline | Margin 0.05 | Strict margin 0.10 |
| --- | ---: | ---: | ---: |
| Questions | 841 | 841 | 841 |
| Any gold evidence | 84.07% | 86.33% | 85.14% |
| Complete gold evidence | 81.57% | 84.54% | 82.88% |
| Paired complete delta | reference | +2.97 points | +1.31 points |
| MRR | 0.5848 | 0.5873 | 0.5860 |
| Evidence precision | 8.74% | 9.04% | 8.87% |
| Mean graph additions | 0.000 | 0.823 | 0.451 |
| Questions with graph admission | 0 | 54.82% | 31.15% |
| Complete evidence rescues | reference | 25 | 11 |
| Complete evidence regressions | reference | 0 | 0 |

The strict arm's deterministic 10,000 resample paired bootstrap interval for complete evidence
coverage was `[+0.59, +2.14]` points. The strict arm therefore improved coverage with no observed
regressions, but it failed the preregistered primary gate because the point estimate was below the
required +2 point uplift.

The 0.05 diagnostic comparator met the +2 point uplift criterion with a bootstrap interval of
`[+1.90, +4.16]` points and zero regressions. It admitted 0.823 graph items per question on
average, compared with 0.451 for the strict arm. The stricter margin reduced graph admission and
removed 14 of the 25 comparator rescues without producing any additional protection in this slice.

Both selective arms preserved the direct top eight for every category 4 question and kept the
context size at ten items. Retrieval latency is not a treatment outcome here because all arms use
the same shared retrieval result per question.

## Gate decision

`selective_margin_strict` is not a promotion candidate under the locked protocol. The regression
guardrail passed with zero regressions, and the interval excluded zero, but the required complete
evidence uplift of at least 2 points did not pass.

This result supports the existing 0.05 margin as the more effective category 4 retrieval setting,
but does not establish generated answer quality or production performance. Any answer replay must
be preregistered separately.

## Reproduction

```powershell
python benchmarks/structural_edge_performance.py --dataset locomo --data locomo10.json --embedder voyage:voyage-4 --candidate-k 20 --seed-k 8 --context-k 10 --edge-budget 2 --neighbor-order retrieval --retrieval-k 20 --selective-gate --selective-margin 0.10 --selective-category 4 --table category4_strict_graph_20260912 --run-id 20260912T000000Z --out docs/results/2026-09-12-category4-strict-graph-confirmation.json
```

The command above reproduces the benchmark configuration. The measured artifact was generated on
VPS2 with the source revision recorded above and copied without modification.
