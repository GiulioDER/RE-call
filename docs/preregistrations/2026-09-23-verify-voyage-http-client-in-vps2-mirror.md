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
