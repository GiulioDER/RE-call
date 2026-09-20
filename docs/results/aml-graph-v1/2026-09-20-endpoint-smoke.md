# AML grounded graph endpoint smoke

Date: 2026-09-20

Initial implementation commit: `29736c1ff43c962618b172c802e6e73d4e46d4eb`

Exact PR-tip confirmation: `00af24fceab9d2cc9eb8b9ab78303f2edb6404c6`

Variant: `G1_grounded_graph`

Endpoint: isolated VPS2 localhost experiment on port 18006. The public `recall-aml.service`
remained active with zero restarts and was not modified. The experiment service was stopped after
verification.

After rebasing onto the current multimodal branch and adding the graph release-manifest binding,
the complete contract suite and graph mechanism probe were repeated against the exact PR tip. The
same checks passed. The isolated experiment service was stopped again, while the public
`recall-aml.service` remained active with its original process and zero restarts.

## External HTTP contract

The authenticated `scripts/aml_hosted_verify.py --mode contract` run passed every check:

- health and version
- durable Add and immediate Search
- identical replay and conflicting replay
- multi-chunk session handling
- exact user isolation
- top-k bounds
- deletion without peer deletion

## Graph mechanism probe

One temporary user and one source session were added through the real HTTP endpoint. The probe
deleted the user in a `finally` block.

| Signal | Result |
|---|---:|
| Add HTTP status | 200 |
| Raw chunks | 1 |
| Compiled chunks | 1 |
| Compiler fallback | false |
| Corpus status | ready |
| Authored relations | 1 |
| Eligible relations | 1 |
| Graph sidecar chunks | 1 |
| Search HTTP status | 200 |
| Graph attempted | true |
| Graph fallback | false |
| Graph profile | `aml-grounded-reference-v1` |
| Relation hits | 1 |
| Invalid relations | 0 |
| Returned only raw evidence | true |
| Top-100 membership changed | false |

The corpus contained one raw item, so the relation target was already inside the protected prefix
and the correct promotion count was zero. The focused deterministic test separately proves that a
valid tail target moves only to rank 9 while the first eight results and complete membership remain
unchanged.

This was a compatibility and mechanism smoke, not an AML quality evaluation. No official AML
evaluation credential was available on the host, so no official Smoke or Full job was started.
