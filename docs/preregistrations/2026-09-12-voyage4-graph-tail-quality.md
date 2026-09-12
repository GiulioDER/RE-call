# Preregistration: Voyage 4 with bounded 8 direct plus 2 graph retrieval

Status: protocol locked before measurement on 2026-09-12.

## Objective

Measure whether the established retrieval-ranked 8 direct plus 2 structural graph configuration
improves gold evidence retrieval when the underlying retriever uses Voyage 4. The comparison is
against the matching top 10 direct Voyage 4 result. This is a retrieval only experiment.

## Frozen inputs and apparatus

* Dataset: `locomo10.json`, 10 conversations and 1,536 answerable questions in categories 1 through 4.
* Dataset SHA256: `79FA87E90F04081343B8C8DEBECB80A9A6842B76A7AA537DC9FDF651EA698FF4`.
* Embedder: `voyage:voyage-4` for indexing and query retrieval.
* Production path: `Indexer`, `PgVectorStore`, and `HybridRetriever`.
* Candidate pool: 20 per retrieval leg, with one shared top 20 result per question.
* Baseline: top 10 direct items.
* Treatment: protect the top 8 direct items and append at most 2 unique, source-backed, one-hop
  structural graph neighbors ranked by their existing retrieval score.
* Graph relations: all authored structural relations present in the converted LOCOMO corpus.
* No gold labels, answer text, category gating, or second semantic query may influence selection.

## Primary hypothesis and gate

The 8 plus 2 treatment will improve complete gold evidence coverage over the Voyage 4 top 10 direct
baseline by at least 2 percentage points. The paired bootstrap 95 percent interval for the complete
coverage difference must exclude zero. No category may decline by more than 3 percentage points.

If the gate fails, the configuration remains an experiment and the production graph policy is not
changed.

## Secondary outcomes

Report any gold hit, complete gold evidence coverage, MRR of the first gold item, evidence precision,
mean added items, paired rescues and regressions, category breakdowns, graph edge counts, and every
question level context and addition. Report whether the direct prefix is preserved.

## Reproduction command

```powershell
python benchmarks/structural_edge_performance.py --dataset locomo --data locomo10.json --embedder voyage:voyage-4 --candidate-k 20 --seed-k 8 --context-k 10 --edge-budget 2 --neighbor-order retrieval --retrieval-k 20 --table voyage4_graph_tail_20260912 --run-id 20260912T000000Z --out docs/results/2026-09-12-voyage4-graph-tail-quality.json
```

The run must use a fresh isolated table or tenant and the raw artifact must not be edited after
measurement. The source revision, input hash, and exact configuration are recorded in the result.
