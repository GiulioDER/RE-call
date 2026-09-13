# Live graph candidate headroom result

Measured on 2026-09-13. The experiment was preregistered in
[2026-09-13-live-graph-candidate-headroom.md](../preregistrations/2026-09-13-live-graph-candidate-headroom.md)
before the audit implementation and measurement.

## Decision

Do not proceed to typed graph relation authoring on the current retrieval generation. The first
preregistered gate failed: only 3 of the 50 frozen queries had missing gold evidence in raw ranks
9 through 20, below the required minimum of 5. The graph cannot select gold evidence that the base
retriever does not place in its candidate pool.

The next retrieval quality work should improve base candidate recall and chunk selection. The
strongest existing lead is the separately measured Voyage Context 4 contextualized embedding path.
After a safe production implementation and a frozen live comparison, rerun this exact headroom
audit. Resume typed relation work only if the candidate pool then passes the gate.

Generic `references` should remain ineligible for unrestricted tail promotion. It produced six
distinct non gold promotions, zero gold promotions, and one complete gold regression in this run.

## Gate results

The population contained 50 frozen queries, of which 22 carried gold labels. Direct ranks 1
through 10 were incomplete for 18 labeled queries.

1. Prediction 1 failed. Missing gold appeared in raw ranks 9 through 20 for query indices 0, 2,
   and 5 only. The observed count was 3, while the decision rule required at least 5.
2. Prediction 2 held. Current topology connected zero gold tail candidates, which is fewer than
   half of the three headroom queries.
3. Prediction 3 held. Generic `references` produced six distinct non gold final promotions and
   zero gold final promotions.
4. True topology produced zero gold recall gains and zero complete gold rescues against the same
   path removed topology control.
5. True topology produced one gold recall loss and one complete gold regression, both on query
   index 1, `can two pytest sessions run at the same time here`.

The typed relation pilot required both at least 5 headroom queries and at least 3 headroom queries
unconnected by current topology. Although all three observed headroom queries were unconnected,
the first condition failed, so the pilot does not proceed.

## Artifact verification

The artifact contains 100 rows, exactly two arms for each of 50 queries. Every row contains the
benchmark audit payload and exactly 20 raw hits. Every embedded response reports trusted evidence,
certified calibration, and the same generation, pipeline, corpus, calibration, and query set
identities.

1. Source commit: `3444036cb72401effcfe47a13ff62f2deb766217`.
2. Generation: `gen_6aaffd1f9712404c8fa5cee5a6af748a`.
3. Query set SHA256: `63d290a61189758a88b47bfed981ae1331d87fc004b752d632c1f5b58f5aa192`.
4. Pipeline fingerprint: `77c918cd93f9200b36e505ae874d49a1949f04a8a85ce5b8b72a8c135b472db7`.
5. Corpus fingerprint: `1c4c0f13223a6f6c2da6d1d9632e1383b3a103fefa0466a0175f77e4e2c3f15e`.
6. Calibration: `cal_fb9135955aae4950b22529ae27019b96`.
7. Raw artifact:
   [2026-09-13-live-graph-candidate-headroom.json](2026-09-13-live-graph-candidate-headroom.json).
8. Artifact SHA256: `2103DA9AB5D3D4F1FD5FE8DDFAB67C4484AD25FD238C0AB9E989A2C8F7A966D2`.

## Recommended next sequence

1. Implement the first class Voyage Context 4 profile with source document grouping, generation
   identity, calibration, bounded requests, shadow rollout, and rollback.
2. Compare Voyage Context 4 with the current Voyage 4 profile on the frozen live memory queries.
   Measure gold at ranks 5, 10, and 20 rather than measuring only final context.
3. Rerun this candidate headroom audit on the winning certified generation.
4. Resume typed relation authoring only if at least 5 queries have raw tail gold and at least 3 of
   those remain unconnected.
5. If graph work resumes, exclude generic `references` from promotion until a separate relation
   precision experiment demonstrates positive gold selection without regression.

## Reproduction

First confirm that no embedding or LoCoMo benchmark process is active and that the shared lock is
free:

```powershell
ssh.exe vps2 "pgrep -af 'index_local\.py|run_index_local\.sh|index_memory_one\.py|run_locomo_embedder_comparison\.py|run_locomo_embedder_comparison_followup\.py' || true; if flock -n ~/recall-repos/.locks/embed.lock true; then echo EMBED_LOCK_FREE; else echo EMBED_LOCK_BUSY; fi"
```

Run the frozen audit from the local worktree:

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/measure-linked-tail-ae1e3543'
$env:RECALL_SOURCE_COMMIT='3444036cb72401effcfe47a13ff62f2deb766217'
python -m scripts.run_live_graph_candidate_headroom --query-set docs/preregistrations/2026-08-17-memory-queries.json --output docs/results/2026-09-13-live-graph-candidate-headroom.json --generation-id gen_6aaffd1f9712404c8fa5cee5a6af748a
```

Verify the artifact hash:

```powershell
Get-FileHash -Algorithm SHA256 docs/results/2026-09-13-live-graph-candidate-headroom.json
```
