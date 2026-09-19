# AML code-aware raw retrieval decision

Date: 2026-09-19

## Question

Does a deterministic code-token ranking leg plus bounded raw source-neighbour restoration improve
the validated AML raw dense plus exact lexical baseline without changing the stored corpus or
calling another model?

This is the first stage of the AML Coding multi-view plan. It selects a local candidate and does
not authorize an official AML run.

## Arms

| Arm | Fixed behavior |
| --- | --- |
| `M0_raw` | Raw evidence, Voyage Context 4 dense retrieval, exact PostgreSQL lexical retrieval, code-aware stage OFF, reranker OFF, graph OFF. |
| `M1_code_neighbors` | The identical M0 corpus and dense plus lexical candidates, followed by the code-aware stage defined below. Reranker and graph remain OFF. |

M1 reuses M0's tenant, corpus hash, generation, and embeddings. It performs no Add, compilation,
embedding, SPLADE backfill, or reranking request.

## Frozen code-token stage

The query and every fused candidate are tokenized locally. Tokens are normalized with Unicode
case folding and slash normalization. The extractor recognizes only these deterministic forms:

1. Repository paths and known software filenames.
2. Command-line flags beginning with `--`.
3. Uppercase configuration or environment names containing at least three characters.
4. Exception names ending in `Error` or `Exception`.
5. Function-call identifiers, snake case identifiers, and camel case identifiers.
6. Code-like atoms contained in Markdown inline-code spans.

Blank tokens, tokens shorter than three characters, and the raw rendering labels `role`,
`content`, and `timestamp` are excluded. A token receives the maximum weight of every form it
matches: three for a path, filename, flag, environment name, or exception; two for an inline-code
atom; and one for another identifier.

M0 first produces its ordinary dense plus lexical reciprocal-rank-fused list with candidate width
100 per leg and RRF constant 60. M1 then creates one additional ranking containing only candidates
with at least one exact normalized query-token match. That ranking is sorted by descending sum of
distinct matched-token weights, then M0 rank, then chunk identifier. It contributes a bounded
weighted RRF term of `0.5 / (60 + rank)`. No fuzzy, substring, semantic, or model-generated token
match is permitted.

## Frozen neighbour stage

After code fusion, M1 considers at most the first eight code-matched raw chunks as seeds. For each
seed it reads raw chunks from the same source session, orders them by message ordinal, segment,
and identifier, and restores the immediate predecessor and immediate successor when present.
The served order for a seed is the seed, then predecessor, then successor. Identifiers are
deduplicated globally, non-raw records are ineligible, and the final Search response is truncated
to the requested Top K. A missing or malformed ordinal makes only that seed ineligible and must be
reported in telemetry; it may not trigger a corpus-wide scan or a generated substitute.

## Frozen execution sequence

1. Run the 34-task present-condition retrieval screen for both arms with three captures per query.
   M0 owns one serial dense embedding pass. M1 reuses that exact tenant.
2. If and only if every retrieval and mechanism gate passes, run the frozen 12-task executable
   screen for both arms at seeds 0, 1, and 2, for 72 paired cells.
3. If and only if M1 wins the executable screen, run a fresh 34-task present-condition executable
   confirmation at seeds 0, 1, and 2, for 204 paired cells.
4. If and only if M1 wins confirmation, run fresh retrieval robustness replays in present, absent,
   adjacent, contradictory, and superseded conditions before it can become the raw base for M2 or
   M3.

Task execution uses exactly three concurrent workers, a 600 second session timeout, and
`deepseek/deepseek-v4-flash`. The coding model is a local diagnostic control, not an internal
Search model and not the model for an official AML run.

## Promotion gates

M1 reaches executable screening only when all of these are true:

1. Version and response telemetry prove the exact code profile, weight, neighbour radius, and
   seed limit, with the stage OFF for every M0 Search and ON for every M1 Search.
2. Corpus, query, task-set, generation, embedding, candidate-width, and RRF identities match.
3. M1 performs zero Add and embedding calls, has zero code-stage fallbacks, and reports valid
   bounded neighbours only.
4. The mechanism changes Top 10 order or membership on at least 17 of 34 distinct present queries
   and restores at least one neighbour on at least 10 distinct present queries.
5. Present mean reciprocal rank improves, complete coverage at 10 and 100 do not decline,
   source-session recall does not decline, p95 Search latency remains below 1,000 milliseconds
   and below twice M0 p95, and all 204 expected retrieval cells are valid.

M1 reaches confirmation only if every executable screen cell is valid and paired, Search is
called and evidence is delivered, candidate-only cell wins exceed baseline-only cell wins, M1
task wins are at least its task losses, and no task with three valid seeds regresses by more than
one seed. A tie or loss selects M0.

M1 wins confirmation only if every expected cell is valid and paired, its overall Task Solve is
higher than M0, present task wins exceed task losses, the task-clustered 95 percent bootstrap
confidence interval lower bound is strictly positive, wrong-fact damage does not increase, and
the retrieval and latency gates still pass. Failure of any gate selects M0.

Retrieval diagnostics cannot promote a candidate that loses executable Task Solve.

## Exclusions and invalidation

The existing offset compiler, compiled records, facets, task routing, SPLADE, Voyage reranking,
graph expansion, entailment, confidence abstention, and context packing are excluded. M5 is
already resolved for the current raw pool by the clean reranker result, which selected raw and
stopped before task execution. It is not repeated here.

Invalidate an affected arm when corpus bytes, chunk order, query bytes, task population, provider
or model identity, prompts, checker, timeout, condition, seed, candidate width, RRF constant, code
profile, or neighbour rule drifts. Never overwrite or repair a partial artifact. Continue only in
a fresh run-specific directory and preserve the failed attempt.

## Expected artifacts

The benchmark preregistration owns the exact task roster, formulas, selectors, immutable artifact
paths, and append-only results marker. Every selector records SHA-256 hashes for all input
artifacts and both served commits.

No AML Smoke or Full run is authorized. The live Coding contract, GPT 4o mini requirement,
Voyage eligibility, and RE-call licence eligibility remain external gates.

