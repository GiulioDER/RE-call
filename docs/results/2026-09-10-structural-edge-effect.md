# Structural edge effect result

Measured 2026-09-10 under protocol commit `04ab5155`, before the effect numbers were collected.
The JSON artifact is [2026-09-10-structural-edge-effect.json](C:/Users/gde00/Documents/recall/docs/results/2026-09-10-structural-edge-effect.json).

## LoCoMo paired routing probe

Input: `locomo10.json`, SHA256
`79FA87E90F04081343B8C8DEBECB80A9A6842B76A7AA537DC9FDF651EA698FF4`.

Population: all 1,536 category 1 through 4 questions with nonempty evidence across 10
conversations. The baseline is the fixed top five lexical turns. The treatment starts with those
same five turns and adds every one hop structural neighbor.

| Measure | Baseline | Structural one hop | Change |
| --- | ---: | ---: | ---: |
| Full gold evidence hit rate | 0.2591 | 0.4245 | +0.1654 |
| Mean candidates | 5.00 | 22.95 | +17.95 |
| Mean evidence precision | 0.0643 | 0.0251 | -0.0393 |

The treatment rescued 254 of the 1,138 baseline misses, or 22.32%. It did not create hit
regressions because the treatment is a superset of the baseline. The improvement is therefore a
real reachability signal, but it is measured with an unbounded expanded set and is not a matched
retrieval budget or end to end answer score.

## Isolated edge types

| Edge type | Hit rate change | Rescues among baseline misses | Mean added candidates | Mean precision |
| --- | ---: | ---: | ---: | ---: |
| `conversation_order` | +0.1230 | 189 | 9.42 | 0.0356 |
| `speaker` | +0.0671 | 103 | 9.50 | 0.0281 |
| `session_date` | +0.1224 | 188 | 8.57 | 0.0378 |
| `reply_continuity` | +0.0000 | 0 | 0.00 | 0.0643 |
| `explicit_entity_repetition` | +0.0000 | 0 | 0.00 | 0.0643 |

The checked out LoCoMo data supplied no explicit reply or entity fields, so those two edge types
were not exercised. It supplied 5,872 conversation order edges, 5,862 speaker edges, and 5,610
session date edges.

## Interpretation

The preregistered numerical gate passes this routing probe: the hit rate gain exceeds 0.02, the
miss rescue rate exceeds 0.05, and the precision drop is within the preregistered 0.10 limit.
That does not justify enabling this expansion as a default production retrieval behavior. The
absolute precision is only 0.0251, the context grows by 4.6 times, and this probe uses a lexical
seed ranking rather than the shipped embedding and trust path.

The safe conclusion is to retain the structural metadata and its tests, but keep one hop
expansion behind an experiment flag until a matched budget, paired embedding retrieval run is
available. This result measures mechanism reachability, not final answer quality.

## ATM status

ATM was not measured. The workspace contains prior ATM result artifacts but not the raw email,
image, video, and question inputs required to construct and score a paired structural edge run.
No synthetic ATM number is substituted.

Result artifact SHA256:
`E8E068F16F3C1200AFE86E9D6F5559F2958E4F3F04E56DFB533B9EF013BDA610`.
