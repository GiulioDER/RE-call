---
name: temporal-supersession-traversal
description: "Pre registered prediction for temporal and supersession aware graph traversal"
valid_from: 2026-09-10
metadata:
  node_type: experiment
  type: project
  originSessionId: codex-temporal-supersession-traversal
  modified: 2026-09-10T00:00:00Z
---

# Prediction

The temporal and supersession aware traversal will preserve the live neighbor when a stale
neighbor appears earlier in deterministic graph order and the graph node budget is one. The
baseline will spend that slot on the stale neighbor, then produce no live neighbor after trust
rejection. The new path will reject the stale neighbor before scoring and admit the live neighbor.

On the synthetic effect benchmark, I predict stale neighbor admission will fall to zero, live
neighbor admission will rise from zero to one, and candidate cosine scoring calls will fall by at
least one for every stale candidate excluded before ranking. On unrelated current neighbors, I
predict the result order and candidate count will be unchanged.

The measurement command will be:

```text
python benchmarks/temporal_supersession_traversal.py
```

The behavior test will be:

```text
python -m pytest tests/test_semantic_graph.py::test_temporal_and_supersession_neighbors_are_filtered_before_budget -q
```

This record is frozen before either measurement. Results will be appended below without editing
the prediction or any recorded number.

## Result

Pending.

## Measured result, 2026-09-10

Command: `python benchmarks/temporal_supersession_traversal.py`

The prediction held on all three deterministic cases. With one slot, the former order scored
`a_expired` and preserved no live neighbor, while the new order scored `z_live` and preserved it.
With two slots, the former order scored `a_expired` and `b_superseded`, preserved no live neighbor,
and the new order scored only `z_live`, saving one scoring call. The current only control preserved
`z_live` in both paths, with one scoring call in each.

Command: `python -m pytest tests/test_semantic_graph.py::test_temporal_and_supersession_neighbors_are_filtered_before_budget -q`

The required mutation proof was red with the temporal and supersession filter loop replaced by an
empty loop. The intended assertion failed because the result contained only `seed` instead of
`seed` and `live`. Restoring the loop made the exact test green.
