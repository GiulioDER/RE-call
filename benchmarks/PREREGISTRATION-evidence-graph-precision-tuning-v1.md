# Preregistration: RE-call Evidence Graph Precision Tuning V1

Status: prediction locked before the VPS2 quality run.

## Objective

Measure whether five deterministic admission changes improve one hop semantic graph precision
without reducing evidence recall or weakening trust and citation behavior.

The experiment does not change embeddings, calibration, authored corpus metadata, ordinary
retrieval, or the public graph expansion modes.

## Prediction

The combined precision policy is expected to improve evidence precision by at least 0.05
absolute over the current graph precision baseline, while keeping evidence recall within 0.02
absolute of the baseline. The largest precision gain is expected from directional traversal and
hub suppression. Selective expansion is expected to reduce graph latency and the number of
candidate trust evaluations.

## Arms

Every arm uses the same tenant, generation, graph snapshot, Voyage 4 profile, calibration, answer
provider, query set, budget, and deterministic ordering.

1. Graph off.
2. Current graph precision baseline.
3. Directional traversal only.
4. Corroboration scoring only.
5. Hub suppression only.
6. Relative cosine admission only.
7. Selective expansion only.
8. All five changes combined.
9. Shuffled relation control using the combined policy.
10. Removed relation control using the combined policy.

The controls are detached in memory from the loaded graph projection. Production graph rows are
never modified by the control arms.

## Primary outcomes

Evidence precision, evidence recall, citation precision, unsupported claim rate, correct
abstention rate, false abstention rate, and manually adjudicated answer accuracy.

The combined arm qualifies only if it beats the current graph baseline on evidence precision,
does not reduce evidence recall by more than 0.02, does not increase unsupported claims, and
beats both relation controls on the primary outcome.

## Secondary outcomes

Graph candidates discovered, rejection counts by reason, relation and entity counts inspected,
graph latency p50 and p95, total retrieval latency, answer provider latency, answer changes,
citation changes, and trusted evidence changes.

## Selection rules

Hub thresholds 16, 32, and 64 and cosine margins 0.05, 0.10, and 0.15 are evaluated as declared
secondary policy settings. Select the highest evidence precision subject to the primary outcome
constraints. Break ties by evidence recall, then latency, then the smallest hub threshold and
smallest cosine margin.

All reported confidence intervals and tests use the query as the unit of analysis. Chunks are
not treated as independent observations. Report paired bootstrap intervals and an exact paired
permutation test.

## Manual review

The reviewer receives every query where an arm changes the answer, citations, trusted evidence,
abstention, contradiction result, or graph gate. The reviewer labels answer correctness, evidence
support, citation precision, unsupported claims, and abstention correctness.

## Reproduction commands

The live retrieval runner is invoked once per arm with the declared policy environment. The local
answer provider then processes every raw response. The aggregate script records raw paths,
configuration identity, policy fingerprint, and every per query observation.

The preregistration is committed before the first quality measurement and is never edited after
the measurement. Results are appended in a separate artifact.
