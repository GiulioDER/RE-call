# Live source conditioned chunk admission result

**Measured:** 2026-09-13  
**Decision:** BUILD SOURCE CONDITIONING in shadow form at registered `alpha=0.08`  
**Pinned generation:** `gen_56dce932a4444411b488bc860fea4fb7`  
**Calibration:** `cal_b458bf825c6d47469c5511b3a5038dff`  
**Source commit:** `508b523582ccb3859cdb712fc78de1fd35ca5704`  
**Remote checkout:** `/home/sentiment/recall-repos/source-conditioned-admission-a6f733a9`  
**Raw artifact:** `docs/results/2026-09-13-live-source-conditioned-admission.json`  
**Raw artifact SHA256:** `AD815D774C1C1AB516CD1C9FB77A1E7BAA637D70EC5AB8890DE0BF41248BA0C7`

## Outcome

Source conditioning found a useful safety operating point, and a larger arm showed a small gold
gain. The registered smallest build arm is `alpha=0.08`: it preserved 14 complete queries and 16
of 25 facts, reduced false evidence answers from 4 of 28 to 2 of 28, increased source hits from 17
of 22 to 18 of 22, and raised context precision from 0.4677 to 0.4941.

The `alpha=0.15` arm was the Pareto leader on the registered metrics. It reached 15 complete
queries and 17 facts, retained 2 false answers, reached 19 gold sources, and raised context
precision to 0.5227. The frozen decision rule still selects the smaller `alpha=0.08`, so 0.15 is a
candidate for independent validation rather than the registered production value.

| Arm | Complete | Facts | Source hit | Context precision | False answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| Served baseline | 14/22 | 16/25 | 17/22 | 0.4677 | 4/28 |
| Trust backfill | 14/22 | 16/25 | 17/22 | 0.4444 | 4/28 |
| Cosine backfill | 12/22 | 14/25 | 17/22 | 0.4568 | 4/28 |
| Source `alpha=0.02` | 12/22 | 14/25 | 17/22 | 0.4512 | 4/28 |
| Source `alpha=0.04` | 13/22 | 15/25 | 17/22 | 0.4691 | 4/28 |
| Source `alpha=0.06` | 14/22 | 16/25 | 18/22 | 0.5000 | 3/28 |
| Source `alpha=0.08` | 14/22 | 16/25 | 18/22 | 0.4941 | 2/28 |
| Source `alpha=0.10` | 14/22 | 16/25 | 18/22 | 0.4943 | 2/28 |
| Source `alpha=0.15` | 15/22 | 17/25 | 19/22 | 0.5227 | 2/28 |
| Source `alpha=0.20` | 15/22 | 17/25 | 19/22 | 0.5111 | 3/28 |

## What caused the movement

Deep trust backfill added seven gold context chunks but zero facts, zero complete queries, and zero
safety improvement. Raw cosine ordering then lost two complete queries and two facts. The source
signal, rather than extra depth or cosine reordering alone, caused the useful movement.

At `alpha=0.08`, the treatment removed the false Kafka migration and warehouse schema answers.
The affiliate programme and disaster recovery controls remained false answers. At `alpha=0.15`,
the same two false answers remained.

The one new complete answer at `alpha=0.15` was the Redis backend verdict. Its gold chunk had raw
cosine 0.483937, held out source support 0.702937, and adjusted score 0.514378, which cleared the
certified 0.509 threshold while remaining a per chunk decision.

The maker rebate query gained its gold source at `alpha=0.08`, but not its essential fact. The
selected gold chunk was ordinal 0 while the essential fact remained in another chunk. This is the
specific evidence that licenses the planned query focused selector inside source conditioned
documents.

## Prediction verdicts

| Prediction | Result | Verdict |
| --- | --- | --- |
| One arm reaches 16 complete and 19 facts with at most 4 false answers | Best was 15 complete and 17 facts | falsified |
| One arm reaches at most 3 false answers while preserving 14 complete and 16 facts | `alpha=0.06` reached 3; 0.08 and 0.10 reached 2 | confirmed |
| A qualifying arm keeps context precision at least 0.44 | Qualifying arms ranged from 0.4941 to 0.5227 | confirmed |
| Trust backfill alone does not satisfy the gold gain | It changed neither complete queries nor facts | confirmed |
| Run stays under 15 minutes with no extra work beyond query embeddings | Summed client time was 113.208 seconds; 50 baseline query embeddings were issued and reused inside paired audit arms | time confirmed; cost not independently instrumented |

## Decision and next experiment

The frozen rule says BUILD SOURCE CONDITIONING because `alpha=0.08` is the smallest arm with at
least 14 complete queries, at least 16 facts, at most 2 false answers, context precision above
0.44, and no loss against trust backfill.

This result licenses a shadow implementation, not direct promotion. The source model and alpha
were evaluated on a small leave one query out corpus and the composite score has no certified
serving calibration yet.

The next gold focused experiment should reuse the source support differently. Select the top three
sources by held out support, retrieve eight chunks inside each selected source using the already
computed query vector, then choose five chunks using per chunk cosine plus source support. Do not
use essential fact labels in selection. This repairs the exact failure of the earlier document
expansion, whose correct source could be third and was cut by its two source limit.

An exploratory analysis, not part of the registered decision, found the gold source within the top
three source support ranks for 18 of 22 answerable queries. Among the eight baseline incomplete
queries, five gold sources ranked in the top three: current suite duration, stale branch regression,
Redis, maker rebates, and clean Git status. Guard mutation ranked fifth, paid API policy sixth, and
gate 1774 eighteenth. This bounds where a top three source expansion can help and identifies the
three residual cases that need a different lever.

## Reproduction

```powershell
git checkout 508b523582ccb3859cdb712fc78de1fd35ca5704
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/source-conditioned-admission-a6f733a9'
$env:RECALL_SOURCE_COMMIT='508b523582ccb3859cdb712fc78de1fd35ca5704'
python -u scripts/run_live_source_conditioned_admission.py `
  --generation-id gen_56dce932a4444411b488bc860fea4fb7 `
  --output docs/results/2026-09-13-live-source-conditioned-admission.json
Get-FileHash -Algorithm SHA256 docs/results/2026-09-13-live-source-conditioned-admission.json
```

The runner aborts unless the served baseline reproduces the registered substrate exactly.
