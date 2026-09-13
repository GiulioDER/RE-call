# Production linked tail parity and live quality result

Measured on 2026-09-13. The implementation was preregistered in
[2026-09-12-production-graph-linked-tail-parity.md](../preregistrations/2026-09-12-production-graph-linked-tail-parity.md).

The LoCoMo artifacts use source revision `ae1e3543a21755ec92019a4504f68330fadc0a7b`.
The live artifact uses source revision `d05d3903db55064137dddd33e4083e5b9218ca13`.
The later revision only pins the isolated remote checkout used by the live runner.

## Decision

Do not promote `linked_tail` or `hybrid` to the serving default.

The production selector exactly reproduces the successful LoCoMo reference mechanism, and true
LoCoMo topology passes the topology control. The live memory result does not pass the quality gate.
Against the identical graph retrieval path with relations removed, true linked topology produces
no gold rescue and one distinct gold regression. The regression repeats in all five recorded
passes. The apparent improvement over literal graph off is caused by the graph path retrieving 20
direct candidates instead of 5, not by graph topology.

The highest ROI next step is relation quality and coverage work on a frozen labeled real memory
pilot. First classify the 18 activated live queries and the three distinct context changes. Then
replace generic structural links with typed, query useful relations such as `depends_on`,
`supersedes`, and `caused`. A corrected live harness must compare true topology with a no relation
arm that keeps the same top 20 retrieval and context assembly path.

## LoCoMo parity and topology

All three controls contain 1,536 answerable questions. Production linked tail returned the exact
same ordered context as the frozen selective margin 0.05 reference in all 1,536 cases, with zero
mismatches.

1. True topology moved complete gold evidence from 69.47 percent to 71.35 percent, a gain of 1.89
   percentage points. Any gold evidence moved from 82.62 to 83.79 percent. MRR moved from 0.5748 to
   0.5761. Evidence precision moved from 10.02 to 10.23 percent.
2. Shuffled endpoints moved complete gold evidence from 69.73 percent to 69.60 percent, a loss of
   0.13 percentage points.
3. Removed relations produced no additions and no metric change.

This passes the preregistered implementation parity and topology gates.

Raw artifacts and SHA256 values:

1. [True topology](2026-09-13-production-linked-tail-parity-true.json),
   `8DC3CA5204E3DF1240CE819A47EC5F283EB754A00F18682CD52F23615FE76D39`.
2. [Shuffled topology](2026-09-13-production-linked-tail-parity-shuffled.json),
   `D76D9F5A656367CFB623EFA15BF2B05953CB9E71F3374C94B5FD3B33CC803E72`.
3. [Removed topology](2026-09-13-production-linked-tail-parity-removed.json),
   `380C8081C03A52E542DDF60AE4BDB46D5313AC4436EFB8B47EAA425EADF2A3D5`.

## Live memory result

The frozen population contains 50 queries, one warmup pass, and five recorded passes. It is bound
to generation `gen_6aaffd1f9712404c8fa5cee5a6af748a` and query set SHA256
`63d290a61189758a88b47bfed981ae1331d87fc004b752d632c1f5b58f5aa192`.

| Arm | Any gold | Complete gold | Mean gold recall | Evidence precision | MRR | Server p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Graph off | 22% | 4% | 0.2518 | 0.2333 | 0.2100 | 625.4 |
| Outside pool | 22% | 8% | 0.3178 | 0.1995 | 0.2100 | 1133.6 |
| Linked tail | 22% | 6% | 0.3026 | 0.1930 | 0.2100 | 1109.9 |
| Hybrid | 22% | 6% | 0.3026 | 0.1933 | 0.2100 | 1098.8 |
| Linked tail, shuffled | 22% | 8% | 0.3178 | 0.1992 | 0.2100 | 1501.4 |
| Linked tail, removed | 22% | 8% | 0.3178 | 0.1992 | 0.2100 | 1109.7 |

Linked tail exposed an eligible candidate for 18 distinct queries, which passes the coverage gate.
It changed the final context for three distinct queries. Compared with the same path removed
control over 250 paired observations, it produced zero gold recall gains and five losses. These
five losses are the same query repeated across five passes. Complete gold evidence was 6 percent
with true topology and 8 percent with relations removed. Mean gold recall was 0.3026 versus 0.3178,
and evidence precision was 0.1930 versus 0.1992.

The regressed query is `can two pytest sessions run at the same time here`. True topology replaced
its third gold chunk with unrelated evidence from
`sentiment-agent/feedback-concurrent-sessions-race-the-same-clone-2026-07-24.md:24`, reducing gold
recall from 1.0 to 0.6667. The other two changed queries added irrelevant evidence without changing
gold recall.

The literal graph off comparison is not a valid topology control. Graph off retrieves 5 direct
candidates, while every graph arm retrieves 20 before assembling the context. The removed and
shuffled controls have zero linked candidate activations but improve complete coverage more than
true linked topology. Their result isolates the direct retrieval depth benefit from topology.

Linked tail did reuse all 224 eligible raw scores and chunk payloads. It performed zero candidate
fetches and zero cosine rescoring. Total database statements were 2,064 versus 1,712 for graph off.
Database result bytes were nearly unchanged, but server p95 was 1,109.9 ms versus 625.4 ms, 77.5
percent slower. Hybrid added outside pool fetching and rescoring without improving gold over linked
tail.

Raw artifact: [2026-09-13-production-linked-tail-live.json](2026-09-13-production-linked-tail-live.json).
SHA256: `332A32A307FE7198BA491D18B0137B755CC1CB6001D62FC09921AEC7B492E87D`.

Gold identifiers in the frozen query set were normalized by adding the `recall/` source prefix and
then compared with each returned evidence item as `source:ordinal`. Aggregate values include all
five recorded passes. Pairwise rescue and regression counts match rows by pass and query index.

## Reproduction

Run LoCoMo from the isolated VPS2 checkout, with the serving dataset and shared embed lock free:

```bash
cd /home/sentiment/recall-repos/measure-linked-tail-ae1e3543
set -a
. /home/sentiment/recall-repos/.env
set +a
RECALL_SOURCE_COMMIT=ae1e3543a21755ec92019a4504f68330fadc0a7b /home/sentiment/recall-repos/.venv/bin/python -m benchmarks.structural_edge_performance --dataset locomo --data /home/sentiment/recall-repos/serving/locomo10.json --embedder voyage:voyage-4 --candidate-k 20 --seed-k 8 --context-k 10 --edge-budget 2 --retrieval-k 20 --neighbor-order retrieval --production-parity --relation-control none --relation-control-seed 20260912 --run-id linked-tail-parity-true-20260913 --table linked_tail_parity_true_20260913 --out docs/results/2026-09-13-production-linked-tail-parity-true.json
```

Repeat with `--relation-control shuffled`, run ID `linked-tail-parity-shuffled-20260913`, table
`linked_tail_parity_shuffled_20260913`, and the matching output filename. Repeat with
`--relation-control removed`, run ID `linked-tail-parity-removed-20260913`, table
`linked_tail_parity_removed_20260913`, and the matching output filename.

Run the live comparison from the local worktree:

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/measure-linked-tail-ae1e3543'
$env:RECALL_SOURCE_COMMIT='d05d3903db55064137dddd33e4083e5b9218ca13'
python -m scripts.run_live_graph_candidate_mode_comparison --query-set docs/preregistrations/2026-08-17-memory-queries.json --output docs/results/2026-09-13-production-linked-tail-live.json --generation-id gen_6aaffd1f9712404c8fa5cee5a6af748a --passes 5 --warmup-passes 1
```

Recheck artifact integrity:

```powershell
Get-FileHash -Algorithm SHA256 docs/results/2026-09-13-production-linked-tail-parity-true.json,docs/results/2026-09-13-production-linked-tail-parity-shuffled.json,docs/results/2026-09-13-production-linked-tail-parity-removed.json,docs/results/2026-09-13-production-linked-tail-live.json
```
