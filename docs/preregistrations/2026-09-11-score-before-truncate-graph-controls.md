Date: 2026-09-11

Status: locked before implementation and measurement

## Question

Does scoring the complete bounded graph neighborhood before final context allocation make graph
topology causally useful, rather than merely reintroducing candidates that the direct retrieval
cutoff omitted?

## Prediction

I predict that the true edge arm will improve complete evidence recall over retrieval only and
will beat the shuffled endpoint arm by a positive paired delta. I predict that shuffled endpoints
will be no better than retrieval only once the candidate budget and scoring budget are matched. I
predict that removing the graph will match retrieval only, apart from measurement noise. I will
not treat a true edge gain over retrieval only as evidence for graph topology unless true edges
also beat shuffled endpoints.

## Frozen serving contract

All four arms use the same query set, generation, tenant, embedder, retrieval profile, query order,
reasoning budget, and answer assembly mode. The graph arm first collects a bounded one hop
neighborhood, fetches at most `MAX_GRAPH_RESCORING_CANDIDATES`, scores every fetched candidate,
protects the direct prefix, and allocates only the remaining graph slots after scoring.

The controls are:

1. `true_edges`: authored graph relations.
2. `shuffled_endpoints`: object endpoints are deterministically permuted, preserving each subject
   out degree and each object in degree while changing the connections.
3. `removed_graph`: no graph relations.
4. `retrieval_only`: graph expansion is off.

No arm may alter the direct retrieval pool, candidate cap, trust policy, or answer model. The raw
per query artifact is immutable. Quality is computed at query level, with paired bootstrap
intervals and separate reporting of candidate counts, rescored counts, graph allocations, trust
outcomes, latency, and answer metrics.

## Decision rule

I will consider topology supported only if true edges beat shuffled endpoints on the preregistered
primary retrieval metric with a positive paired interval, while the true edge arm also clears the
existing retrieval quality gate and neither control shows a trust or refusal regression. If true
edges do not beat shuffled endpoints, I will report candidate reintroduction as the supported
mechanism and make no topology based rollout recommendation.

## Reproduction

The implementation test is the focused semantic graph unit test suite. The quality run must use the
existing graph performance attribution runner with one recorded artifact per arm and the same
generation pin for all four arms. Results belong in a separate document and must not edit this
prediction section.
