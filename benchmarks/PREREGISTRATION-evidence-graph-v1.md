# Preregistration: RE-call Evidence Graph V1

Status: preregistered before graph quality measurement.

## Objective

Measure whether deterministic, authored one hop semantic graph expansion improves evidence recall
and answer quality without weakening trust or citation behavior. This study does not evaluate a
community report, global GraphRAG search, DRIFT search, or default model extraction.

## Arms

Each query is run against the same tenant, generation, calibration artifact, corpus snapshot,
embedder, answer provider, and budget:

1. Current hybrid retrieval.
2. Authored supersession graph expansion.
3. Deterministic Evidence Graph V1 expansion with `graph_expansion=one_hop`.
4. Shuffled relation control, preserving relation count and endpoint degree distribution.
5. Removed relation control, with semantic relations unavailable.
6. Proposal assisted reasoning without semantic graph expansion.

The graph is disabled for the baseline and proposal only arms. Relation controls are generated
from the same persisted graph snapshot and are never mixed with the production graph rows.

## Primary outcomes

The primary outcomes are evidence recall at fixed candidate budget, answer accuracy, citation
precision, unsupported claim rate, correct abstention rate, and false abstention rate.

Secondary outcomes are entity disambiguation accuracy, contradiction detection precision, p50 and
p95 latency, graph build time, graph storage size, model calls, and token usage.

## Per query observation record

The evaluation artifact records one row per query and arm with:

* query identifier and preregistration version;
* tenant, generation, pipeline, corpus, calibration, and query set identities;
* arm, graph expansion mode, graph readiness, graph fingerprint, and relation control seed;
* initial trusted chunk identifiers and appended trusted chunk identifiers;
* rejected candidate count, graph diagnostic count, entity and relation counts inspected;
* answer, abstention outcome, citations, gold evidence identifiers, and adjudication labels;
* unsupported claims, contradiction decisions, latency stages, graph build latency, model calls,
  and token usage.

Corpus text is stored only in the evidence artifact with the existing access controls. Refusal,
diagnostic, and advice fields contain identifiers and sanitized codes, not corpus text.

## Analysis rules

The same query set and deterministic ordering are used for every arm. No graph quality claim is
made unless the semantic graph arm beats both the shuffled relation and removed relation controls
on the preregistered primary outcome without increasing unsupported claim rate. Missing, stale, or
uncertified calibration is a failed run, not an implicit degraded result. Queries with unresolved
gold labels are reported separately and are not silently reclassified.

All deviations, failed runs, and per query observations are retained beside the aggregate report.
