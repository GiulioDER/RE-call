# Preregistration: graph-first retrieval probe

Date: 2026-08-28

## Question

Can the generation-bound semantic graph provide useful query seeds before the first trusted
retrieval, without promoting graph metadata or graph text to evidence?

## Frozen population and snapshot

Use the unchanged `agent-ab-skill-001` population: 15 known misses and 31 hit controls. The
original prompts and initial queries remain fixed, and labels are hidden from the serving tool.
The input SHA256 is:

`a87c7eebe0f7ce7daf45a5d689fa9a4212a938b32297b893cd89c8d89572c2ee`

All modes use the same read-only VPS2 generation snapshot:

* generation: `gen_168361bd2310433e87beda1fc6f4a5e0`;
* corpus fingerprint: `0448c9d4a25d00c4561790b75a0d5286e0480d1b894c2aea4c4c74061db5586`;
* pipeline fingerprint: `77c918cd93f9200b36e505ae874d49a1949f04a8a85ce5b8b72a8c135b472db7`;
* tenant: `memory`;
* embedder: `voyage:voyage-4`;
* retrieval profile: `fast`;
* graph generation: `one_hop` is not used in this probe;
* workers: `2`;
* result size: `k=5`;
* graph query candidates: at most `3`.

## Arms

Each graph-first call first loads the graph, creates deterministic query proposals, and then runs
the original query and every proposal through the ordinary trusted retrieval layer.

1. `entity`: exact canonical entity or alias matches only.
2. `relation`: exact matched entities followed through authored directional relations only.
3. `hybrid`: entity and relation proposals combined in deterministic order.

The baseline is the original query result embedded in every graph-first response. Graph output is
proposal data only. Every result remains subject to tenant, generation, calibration, validity,
supersession, and trust checks.

## Apparatus invariants

The graph must match the requested tenant and generation, pipeline fingerprint, and corpus
fingerprint. Ambiguous entities, candidate relations, diagnostic-only relations, and graph text
are excluded from proposal construction. A graph failure falls back to the baseline result and is
reported as apparatus state. No gold field is sent to the MCP tool.

## Metrics and decision rule

Primary metrics are top-five governing-memo recovery on the 15 misses, misses rescued relative to
the embedded baseline, and retention on the 31 controls. Secondary metrics are graph activation,
candidate count, newly trusted chunks, rejected candidates, retrieval calls, graph latency, and
fallback rate.

Prefer a graph-first mode only if it rescues at least 5 of 15 misses, retains at least 30 of 31
controls, and stays within the three candidate query bound. Otherwise retain the current retrieval
path and close this graph-first route as unsupported for the tested miss class.

## Artifact protocol

Run each mode separately with immutable raw and checkpoint JSON paths. Generate summaries with
`scripts/summarize_graph_first_batch.py`. Record SHA256 digests for every raw and summary artifact.
If a process exits before writing its raw artifact, resume the same mode and checkpoint with
identical settings. Do not combine modes in one checkpoint.
