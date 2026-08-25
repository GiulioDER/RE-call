# Preregistration: graph assisted query construction replay

Date: 2026-08-25

## Question

When retrieval already returns a useful project or file anchor but misses the governing memory, can a bounded graph assisted query construction step recover that memory without harming existing hits?

## Frozen inputs

The benchmark population is frozen from `agent-ab-skill-001`:

* 15 miss sessions: `ts-lf-rewrite`, `ts-worktree-import`, `ts-sample-covers-tail`, and `ts-raise-on-missing`.
* 31 hit controls from the same qualification run.
* For each session, use the first recorded `recall_search` query as the baseline query.
* Gold governing memos and session labels come from the existing qualification artifact and the prior expansion and alias preregistrations. No session may be moved between miss and control after this file is committed.

The archived run did not preserve graph traces. The replay therefore uses the current trusted `re-call-docs` reasoning projection and records its generation and corpus fingerprints in the result. It is a current graph replay, not a claim about the old graph generation.

## Arms

1. `off`: run the baseline query with `graph_expansion=off`.
2. `one_hop`: run the same query with deterministic one hop graph expansion.
3. `graph_query`: only for sessions classified as anchored by the `one_hop` response, construct at most two followup queries from the initial trusted evidence and graph neighbors, then retrieve each with ordinary trusted retrieval. Keep the baseline query in the candidate set.
4. `graph_query_loop`: only when `graph_query` adds no governing memo and returns at least one new trusted graph neighbor. Construct one final query from the new neighbor set. No further loop is allowed.

The query constructor may use only the original query, trusted initial evidence, graph node names, relation labels, and source metadata returned by the reasoning response. It may not use the gold memo text, gold filename, answer text, or a hidden target label.

## Anchor classification

`anchored` means the one hop response contains at least one trusted graph node or relation that is not already in the baseline evidence and is connected to an initial trusted evidence source. `unanchored` means no such node is returned. Unanchored sessions are reported but receive no graph query arm.

## Primary metrics

* Governing memo recovery at top 5, measured at the session level.
* Rescue count among the 15 frozen misses.
* Hit retention among the 31 frozen controls.
* Anchored versus unanchored rescue rates.

Secondary metrics are number of new trusted chunks, query novelty, graph nodes used, provider calls, and total latency.

## Gates and exclusions

* A graph arm is a rescue only when the governing memo appears in top 5 and the response is trusted.
* Any trust refusal, malformed response, missing generation identity, or graph projection mismatch is excluded from rescue counts and reported as an apparatus failure.
* The experiment is informative only if all 15 misses and all 31 controls have valid baseline and one hop responses from one generation.
* The graph query arm is judged only on its predeclared eligible anchored subset. No post hoc eligibility changes are allowed.

## Decision rule

Prefer graph assisted query construction only if it rescues at least 5 of 15 misses, retains at least 30 of 31 controls, and produces no more than two provider calls per eligible session. Otherwise keep the current retrieval path and treat the result as diagnostic evidence for a different query or indexing intervention.
