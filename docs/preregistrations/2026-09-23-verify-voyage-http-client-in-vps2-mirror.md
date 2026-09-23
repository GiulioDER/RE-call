# Pre-registration: does PR #705's Voyage HTTP client hold up in a VPS2-shaped environment?

**Date:** 2026-09-23   **Status:** predicted, not yet measured

## The question

PR #705 (`recall._voyage_http`, head `f4031c0c`) replaces the `voyageai` SDK on RE-call's Voyage
paths and was measured on this workstation and on a VPS3 venv without `torch`. Does it hold in a
venv that mirrors VPS2's serving environment, the CUDA build of `torch` included: that is, do the
Voyage paths stop loading `torch`, and do they return the same results as the SDK?

Answerable by: which modules are in `sys.modules` after each run (yes/no), stage wall times and
peak resident memory per arm, and element-wise differences between the two arms' vectors and
rerank scores.

## What I predict

Two arms in ONE venv (the VPS2 mirror built for
`2026-09-23-voyage-cold-start-profile.md`: torch 2.13.0 CUDA build, transformers 5.15.0,
sentence-transformers 5.7.0, fastembed 0.8.0, voyageai 0.5.0), selected by `PYTHONPATH` and
confirmed by `recall.__file__` in every run:

- **base**: `fd7d0df4`, the parent of #705's first P2 commit `f966914c`, so the only differences
  are the P2 commits;
- **pr**: `f4031c0c`, #705's head.

The probe runs five stages in one process: (1) `import recall_mcp.server`, (2) build
`voyage-context:voyage-context-4`, (3) its `embed_query`, (4) build `voyage:voyage-4` and
`embed_query`, (5) `VoyageReranker` (`rerank-2.5`) over three fixed documents.

| id | claim | predicted |
|---|---|---|
| V1 | `torch`, `transformers`, `sentence_transformers`, `voyageai` in `sys.modules` at the end, pr arm | absent in every run |
| V1b | the same, base arm | `torch` present in every run |
| V2 | stage 2 warm median, pr arm | 0.05 to 0.6 s |
| V2b | stage 2 warm median, base arm | 6 to 10 s |
| V3 | process peak RSS median, pr arm | 80 to 160 MB |
| V3b | process peak RSS median, base arm | 900 to 1,000 MB |
| V4 | whole probe warm wall median: pr / base | 1.5 to 3.5 s / 10 to 14 s |
| V5 | context-4 query vector, pr against base in the same pair | bit-equal (max abs difference 0) in every pair |
| V6 | voyage-4 query vector, pr against base | bit-equal in every pair where base agrees with itself |
| V7 | rerank result (index order and scores), pr against base | identical in every pair |

The base arm figures are anchored on the measurement already recorded in
`2026-09-23-voyage-cold-start-profile.md` (stage 2 7.49 s warm, peak 954 MB), not guessed. The pr
arm band is set near the structural ceiling (no SDK import at all) rather than discounted from
it, because #705's own record found that discounting a structural effect under-predicted it.

## What would falsify this

- `torch` present in any pr-arm run: the HTTP client does not remove the dependency in a VPS2-shaped
  environment, whatever it did elsewhere.
- Any pr/base difference on a case where base agrees with itself: #705 is not a drop-in.
- pr stage 2 above 1.5 s warm, or pr peak RSS above 300 MB: something else still loads the heavy
  stack.

## How it will be measured

VPS3 only, under `~/recall-tb/sessions/pyspy-profile/` (session-owned): git worktrees of the
session's own clone at `fd7d0df4` and `f4031c0c`, the existing mirror venv, `PYTHONPATH` set to the
arm's worktree, `recall.__file__` printed and checked per run. Nothing on VPS2 is touched.

- n = 5 runs per arm, alternating base, pr, base, pr, ... Run 1 of each arm is reported as "first";
  runs 2 to 5 are warm, reported as median and range.
- Each run: wall time and peak RSS from `/usr/bin/time`; per-stage `perf_counter` and
  `ru_maxrss`; the four module flags; the full context-4 and voyage-4 query vectors and the rerank
  result written to a per-run JSON file.
- Parity is computed pair by pair (base run i against pr run i), and each arm against its own
  next run, to separate provider nondeterminism from a client difference. A case where base
  disagrees with itself is reported, not counted, the same rule #705's record used.
- About 3 Voyage calls per run, 30 in total.

## What I already know

- `2026-09-23-voyage-cold-start-profile.md`: in this mirror venv the SDK import takes stage 2 to
  7.49 s warm and +742 MB, through `voyageai.chunking` → `langchain_text_splitters` →
  `sentence_transformers` → `transformers` → `torch`.
- `2026-09-23-voyage-http-client.md` (on #705's branch): bit-equal on 6 of 6 SDK-self-consistent
  cases; workstation 46.18 s → 1.29 s and 558.8 → 52.3 MB; VPS3 without `torch` 1.02 → 0.47 s.
  voyage-code-3 queries were not deterministic at the provider.

## Confounds I can name now

- Another session's C8 run may still be active on VPS3 (network bound), hence ranges.
- `PYTHONPATH` precedes the venv's editable `.pth` entry; if it did not, both arms would run the
  same code. `recall.__file__` per run is the check, and a mismatch voids the run.
- The mirror venv resolves `mcp==2.1.0`; VPS2 has 2.0.0.
- #705 is stacked on the P1 branch; `fd7d0df4` includes P1, so P1 is in both arms equally.

## Result (2026-09-23)

**Status:** measured

VPS3, mirror venv, 10 runs (base, pr alternating, i = 1 to 5), 09:11 to 09:13 UTC. `recall.__file__`
pointed into the arm's own worktree in all 10 runs, so the `PYTHONPATH` selection worked. Every run
exited cleanly and wrote its JSON. Warm = runs 2 to 5.

| id | predicted | measured | held |
|---|---|---|---|
| V1 | heavy modules and `voyageai` absent, pr | `torch` 0/5, `transformers` 0/5, `sentence_transformers` 0/5, `voyageai` 0/5 | yes |
| V1b | `torch` present, base | 5/5 (all four modules 5/5) | yes |
| V2 | pr stage 2: 0.05 to 0.6 s | 0.327 s (0.318 to 0.339); first 0.357 s | yes |
| V2b | base stage 2: 6 to 10 s | 7.849 s (7.662 to 8.037); first 8.028 s | yes |
| V3 | pr peak RSS: 80 to 160 MB | 96.1 MB (96.1 to 97.0) | yes |
| V3b | base peak RSS: 900 to 1,000 MB | 954.4 MB (954.1 to 954.8) | yes |
| V4 | whole probe warm: pr 1.5 to 3.5 s / base 10 to 14 s | 3.22 s (3.04 to 3.27) / 12.07 s (11.89 to 12.34) | yes |
| V5 | context-4 query bit-equal in every pair | **4 of 5 pairs bit-equal; pair 2 differs by up to 0.00636** | **no** |
| V6 | voyage-4 query bit-equal where base agrees with itself | base never agreed with itself (adjacent base runs differ by 8.1e-4 to 9.4e-4) | not countable |
| V7 | rerank identical in every pair | same order in 5 of 5 pairs | yes, order only (see below) |

The other stages did not move: server import 1.51 s base against 1.56 s pr; the first context query,
the voyage-4 query and the rerank each within 0.07 s between arms.

**V5, the falsified prediction.** The base arm returned the same context-4 query vector in all 5
runs. The pr arm returned that vector in runs 1, 3, 4 and 5, and a different one in run 2 (max
absolute difference 0.00636 against every other run of either arm). Under the registered rule, base
agreed with itself, so the case counts and the prediction fails. What the data does not settle is
why: a client that sent a different request would differ on every run, not one in five, and
#705's own record found the provider nondeterministic for another model's queries. So "provider
variance that five base samples happened not to show" and "an intermittent client difference" both
fit. Deciding between them needs more samples per arm, which is a new measurement and is not done
here.

**V6.** The SDK arm itself returned two or more different voyage-4 query vectors across its five
runs, while the pr arm returned one vector five times. That is provider variance on this input
or an SDK-side difference, not evidence against #705, and it is reported, not counted.

**V7 caveat.** `VoyageReranker` reorders and never rescores, so every score in both arms is the
input 0.0. The comparison therefore checks ORDER only (`d2, d0, d1` in all 10 runs), not rerank
scores. #705's record compared the raw (index, score) pairs, which this probe could not see.

**Gap.** Nine of ten claims held or were set aside by the rule. The cost side is at the structural
ceiling in a VPS2-shaped venv: stage 2 falls from 7.85 s to 0.33 s, peak memory from 954 MB to
96 MB, and none of `torch`, `transformers`, `sentence_transformers` or `voyageai` loads. The one
miss is parity. I predicted bit-equality everywhere because #705's record found context-4 queries
deterministic; one pr run in five disagreed, and I cannot yet tell provider variance from a client
difference.

## Follow-up pre-registration (2026-09-23, written after the result above, before this run)

**Status:** predicted, not yet measured

**Question.** Is the context-4 mismatch in V5 (pr run 2, max abs 0.00636) provider variance or a
difference in what the two clients send?

**Design.** Same VPS3 mirror venv, same two worktrees (`fd7d0df4`, `f4031c0c`), `PYTHONPATH`
selection checked by `recall.__file__` per run. n = 20 runs per arm, alternating base, pr. Each run
builds `voyage-context:voyage-context-4` and calls `embed_query` 5 times on the same fixed query,
so 100 calls per arm. `requests.Session.request` (the one call both clients send through) is
wrapped to record the method, the URL, the JSON body and the header NAMES (never values, since
one is the API key) of every request. Each returned vector is hashed (SHA-256 of its float64
bytes).

| id | claim | predicted |
|---|---|---|
| W1 | the context-4 query request body, pr against base, compared as parsed JSON | identical in every run |
| W2 | method and URL, pr against base | identical |
| W3 | calls whose vector differs from the arm's modal vector, base arm, out of 100 | 0 to 10 |
| W4 | the same, pr arm | 0 to 10 |
| W5 | the two arms' variant counts, two-sided Fisher exact test | p > 0.05 |
| W6 | the modal vector hash, pr against base | the same hash |

**Decision rule, fixed now.** If W1 and W2 hold, the two clients send the same request, and any
variant vector is provider-side; #705 is a drop-in on this path. If W1 fails, the named body
difference is the explanation to investigate, whatever W3 to W5 show. W3 to W5 alone cannot
convict or clear the client, because the provider is known to vary.

**Confounds.** Headers are compared by name only; `User-Agent` values are expected to differ and
are not compared. Calls inside one process are sequential and may be served by the same backend
instance, so within-run variants may be rarer than across-run ones. About 200 Voyage calls.

## Follow-up result (2026-09-23)

**Status:** measured

40 runs, 09:18 to 09:22 UTC, base and pr alternating, i = 1 to 20. `recall.__file__` pointed into
the arm's own worktree in all 40. `torch` loaded in 20 of 20 base runs and 0 of 20 pr runs. Each
run sent 6 requests: 1 while building the embedder and 5 queries. 200 query calls in total.

| id | predicted | measured | held |
|---|---|---|---|
| W1 | request bodies identical in every run | identical in 20 of 20 runs (all 6 requests of each pair) | yes |
| W2 | method and URL identical | 20 of 20: `POST https://api.voyageai.com/v1/contextualizedembeddings` | yes |
| W3 | base variants: 0 to 10 of 100 | 2 of 100 (run 3 call 2, run 14 call 3) | yes |
| W4 | pr variants: 0 to 10 of 100 | 1 of 100 (run 19 call 1) | yes |
| W5 | Fisher exact p > 0.05 | p = 1.0 | yes |
| W6 | same modal hash | yes, `a6a739bc8fa24029` in both arms | yes |

The body both clients send, parsed: `{"encoding_format": "base64", "input_type": "query",
"inputs": [["where does the first query spend its time"]], "model": "voyage-context-4",
"output_dimension": 1024, "output_dtype": "float"}`. Header names were `Authorization` and
`Content-Type` in both. The `requests` keyword arguments differ (the SDK also passes `files`,
`proxies` and `stream`), which are transport options, not request content.

**Every variant, in either arm, is the SAME second vector, `d4d0011b5e02415f`.** The provider
returns one of two vectors for this query, about 1 to 2 times in 100, to whichever client asks.

**Decision, by the rule fixed before this run:** W1 and W2 hold, so the two clients send the same
request and the variants are provider-side. #705 is a drop-in on the context-4 query path, and
V5's single mismatch above is best explained as this provider variance. (That V5 variant was not
hashed, so matching it to `d4d0011b5e02415f` by value is an inference, not a measurement.)

**Gap.** All six held. What I did not know beforehand: the provider variance is not noise spread
over many vectors but a two-valued answer, which makes the variant immediately recognisable
across arms and is why 100 calls per arm were enough to settle the question.
