# AML standard entailment screen decision

Date: 2026-09-19

## Decision question

Test the standard shipped RE-call entailment judge as one isolated addition to the selected AML
coding baseline. The experiment asks whether the judge removes unsupported evidence in the
`absent` and `adjacent` conditions without losing answer-bearing evidence in `present`.

This is a deliberately small screen. A positive retrieval result authorizes a larger confirmation;
a tie, an identity failure, or any present-evidence regression keeps entailment OFF.

## Fixed arms

1. `B0_raw`: raw Voyage Context 4 dense plus exact lexical retrieval, reranker OFF, graph OFF,
   entailment OFF.
2. `B3_raw_entailment`: identical corpus, query, candidate construction, ordering, and result
   budget, followed by the standard pinned QNLI judge at threshold 0.5.

The judge identity is fixed to `cross-encoder/qnli-distilroberta-base` at Hub revision
`7dd04ee0a6040c06fb381ad7edcb8585f4d937fd`. It runs locally and makes no DeepSeek, OpenRouter,
OpenAI, or Voyage generation call.

## Prior and expected failure mode

Earlier RE-call work found that the standard QNLI boundary can fail on section-sized chunks: it
may reject the correct chunk together with nearby distractors. Other measured corpora also found
weak answerability discrimination. The prior for this AML screen is therefore negative. The test
is still warranted because the AML coding chunks and conditions differ, and the existing cached
tenants make the check inexpensive.

No threshold tuning is authorized. If the standard judge fails, the standard option remains OFF.
A sentence-window judge, recalibration, or a different model would be a new feature and a new
preregistration.

## Hosted semantics

The normal RE-call entailment stage runs after trust evaluation and judges only verdict-`ok` hits.
The AML Hosted raw path does not call `trusted_search`, so the environment flag alone cannot affect
AML Search. The experimental Hosted adapter must therefore expose the same decision semantics over
the raw eligible candidate pool:

1. Judge every fused candidate before the Top 100 boundary.
2. Preserve the relative order, score, content, and provenance of candidates that pass.
3. Exclude candidates that fail.
4. Return an empty `data` array when no candidate passes.
5. Fail the experimental request on judge failure or wrong output cardinality. There is no silent
   B0 fallback.

The public AML Search response schema remains unchanged. Model identity, candidate counts,
decisions, abstention, and latency are emitted only through diagnostic headers and immutable
experiment traces.

## Frozen small screen

Use the twelve-task roster already frozen for the clean reranker executable screen:

1. `xs-evolve-lease`
2. `xs-join-batch`
3. `xs-widen-manifest`
4. `fa-dedup-key`
5. `ts-mig-name`
6. `ts-semver-pin`
7. `ts-retry-cap`
8. `ts-config-layer`
9. `ts-atomic-write`
10. `ts-idempotent-run`
11. `ts-glob-hidden`
12. `ts-quote-shell`

Run `present`, `absent`, `superseded`, `contradictory`, and `adjacent`, one fresh capture per arm.
The population is 12 tasks by five conditions by two arms, for 120 Search requests. B0 and B3
reuse the five immutable clean-reranker tenants and perform zero Add, Delete, indexing, or dense
embedding operations.

`present` is the benefit control. `absent` and `adjacent` are the conditions where QNLI should help.
`superseded` and `contradictory` are negative controls because a stale or conflicting statement can
still answer the literal question.

## Promotion gate

B3 advances only when all of the following hold:

1. Every expected B3 request attempts and completes the exact pinned judge at threshold 0.5.
2. There are zero judge errors, wrong-cardinality outputs, fallbacks, identity mismatches, missing
   pairs, invalid cells, and unexplained corpus-hash mismatches.
3. Present complete coverage at 100 does not decline.
4. Present mean reciprocal rank does not decline.
5. At least two paired task-condition cells in `absent` or `adjacent` improve unsupported or planted
   evidence admission, with no paired regression on that endpoint.
6. The paired Youden J delta is strictly positive.
7. Search p95 is below 5,000 milliseconds.

A tie or failure selects B0 and stops. The gate is intentionally conservative because only one AML
Full run is available.

## Authorized continuation

If the small screen passes, a separately committed append-only continuation may run all 34 tasks,
all five conditions, and three retrieval captures on the same cached tenants. Only a positive full
retrieval replay may authorize a 216-cell DeepSeek executable screen over `present`, `absent`, and
`adjacent`, using exactly three concurrent task workers. The 1,020-cell confirmation remains a
later gate.

No task solver, official Smoke, or official Full run is authorized by this decision record.

## Graph boundary

Graph remains outside this experiment. Current AML Add requests contain messages and session
identity but no relation metadata, while RE-call's conservative graph derives edges from authored
or explicit relations. Turning graph ON over a zero-edge tenant is not an experiment.

A future graph arm must build query-blind, evidence-backed relations during Add from file paths,
symbols, imports, calls, tests, errors, and explicit revision language. It must never use Search
queries, answer options, task labels, checkers, gold facts, or Full-run results. That feature needs
its own preregistration and organizer confirmation before it can affect an official run.

## Artifact and evidence rules

The canonical preregistration is AMB `preregistration/091-recall-standard-entailment-screen.md`.
Implementation and result commits are separate and signed. Existing artifacts are immutable;
every attempt uses a new path. Tests use exactly three pytest workers, and every new behavioral
test needs a plausible mutation red proof before its green run.

No two embedding or indexing processes may run concurrently. This screen authorizes no embedding
or indexing process at all.

## Implementation checkpoint

The isolated B3 Hosted path now applies the pinned judge once to the complete fused pool before
the Top 100 boundary, preserves passing order, returns an empty data array when every candidate is
rejected, and raises on judge failure or wrong cardinality. B0 remains isolated even if a judge
object exists in the process. The public AML body is unchanged.

Diagnostic headers and structured service logs record attempted and completed state, exact model,
revision, threshold, input, output, accepted and rejected counts, judge latency, total Search
latency, served commit, generation, corpus hash, and variant. A dedicated service uses port 18006
and refuses any variant outside B0 and B3.

Five production mutations established red proof for full-pool ordering, wrong-cardinality refusal,
B0 isolation, variant registration, startup readiness, and isolated service configuration. The
unchanged focused tests then passed with exactly three workers. The complete Hosted module passed
with 68 tests and one expected skip. Ruff and focused mypy passed. Full-tree mypy found two
pre-existing errors in `benchmarks/atm_bench.py`, which this implementation does not modify.

## Premeasurement telemetry closure

Commit `be6158b2c298d3506d87cc851bbd53fb5bf1e06e` adds the explicit
`X-Recall-Entailment-Abstained` header required by the frozen telemetry contract. The value is
true only when the B3 judge completed and admitted no candidate. B0 always reports false. The
behavior was mutation red, then the full Hosted suite passed with exactly three workers before
any measured Search request.
