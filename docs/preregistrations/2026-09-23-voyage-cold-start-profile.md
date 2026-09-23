# Pre-registration: where the MCP server's first-query cold start goes (py-spy on VPS3)

**Date:** 2026-09-23   **Status:** predicted, not yet measured

## The question

When a `recall_mcp.server` process first builds a Voyage embedder, does the `import voyageai`
inside `VoyageContextualizedEmbedder.__init__` (`recall/embeddings.py`) pull in `transformers` and
`torch`, and is that import the largest single cost between process start and the first embedded
query?

Answerable by: `"torch" in sys.modules` after the embedder is built (yes/no), and the wall time
and resident memory each stage adds, per venv.

## What I predict

Three stages, timed in one process: (1) `import recall_mcp.server`, (2)
`resolve_embedder("voyage-context:voyage-context-4")`, (3) one `embed_query` call.

Two venvs with identical code (`674d9a8b`): **mirror**, which adds VPS2's model packages
(torch 2.13.0, transformers 5.15.0, sentence-transformers 5.7.0, fastembed 0.8.0), and **lean**,
which does not have them.

| id | claim | predicted |
|---|---|---|
| P1 | stage 1, mirror, warm wall time | 1.0 to 4.0 s |
| P1m | `torch` in `sys.modules` after stage 1, mirror | **no** (voyageai is imported lazily) |
| P2 | stage 2, mirror, warm wall time | 5 to 15 s |
| P2m | `torch` in `sys.modules` after stage 2: mirror / lean | **yes / no** |
| P2r | max RSS growth over stage 2, mirror | +400 to +900 MB |
| P3 | stage 2, lean, warm wall time and RSS growth | under 1.5 s and under 150 MB |
| P4 | share of stage 2 (mirror) `-X importtime` cumulative time under `transformers` or `torch` | at least 60% |
| P5 | stage 3 wall time, either venv | 0.2 to 2 s, network bound |
| P6 | py-spy, mirror, non-idle samples over the whole probe falling under a `transformers` or `torch` import | at least 50% |

The anchor is a measurement, not a guess: the same import measured on VPS2 on 2026-09-22 at
11.43 s and 9.97 s and 778 MB (`2026-09-23-core-optimize-p1-serving-correctness` in memory).
Because [[i-over-predict-effect-magnitudes]] says my COST predictions have run about 5 times too
LOW, I have not shaded P2 downward from that anchor; the band is wide on the low side only because
VPS3's CPU is a different machine.

## What would falsify this

- P2m false (torch absent after stage 2 in mirror): the P2 premise is wrong for this code path,
  and the 10 s measured on VPS2 comes from somewhere else.
- Stage 2 mirror under 3 s warm: the import is not the cold start that matters.
- Stage 1 larger than stage 2 in mirror: the priority for P2 is the server import, not voyageai.

## How it will be measured

On VPS3 only (VPS2 is hands off during the official AML run), under
`~/recall-tb/sessions/pyspy-profile/`, a fresh clone at `674d9a8b` and two venvs built for this
session. Nothing in the shared testbench repo, venv or databases is changed.

- Probe script: records `time.perf_counter()`, `resource.getrusage(RUSAGE_SELF).ru_maxrss` and
  `"torch" in sys.modules` after each stage, printed as JSON.
- Wall time: 4 runs per venv without any profiler attached. Run 1 is reported as "first" and runs
  2 to 4 as warm; the warm figure is the median and the range is quoted.
- Import attribution: `python -X importtime` on the probe, once per venv, aggregated by top-level
  package.
- Sampling profile: `py-spy record --rate 250 --idle --format speedscope` and a flame graph, one
  run per venv.
- n: 4 wall-time runs per venv, 1 importtime and 1 py-spy run per venv. One embed call per run,
  so 12 Voyage calls in total.

## What I already know

- VPS2 serving venv (read 2026-09-23, `pip list`): torch 2.13.0, transformers 5.15.0,
  sentence-transformers 5.7.0, fastembed 0.8.0, voyageai 0.5.0, langchain-text-splitters 1.1.2,
  mcp 2.0.0, Python 3.12.3.
- VPS3 testbench venv: no torch, transformers or fastembed. That is why a mirror venv is needed at
  all: profiled as it is, VPS3 cannot reproduce the VPS2 cost.
- `recall-cli-import-costs-six-seconds` (memory, 2026-08-25, Windows): `import recall.cli` 6.09 s,
  of which the psycopg chain through `recall.calibration_v2` was about 2.8 s.
- `CLAUDE.md`, Testing: in tests, `voyageai` pulls `langchain_text_splitters`, which pulls
  `transformers` and `torch`; moving it out of module scope took one test file from 45.33 s to
  1.04 s on Windows.

## Confounds I can name now

- VPS3 is a 4 core host and VPS2 has 12, possibly different CPU generations: absolute seconds do
  not transfer, the mirror/lean ratio and the attribution shares do.
- Another session's C8 run is live on VPS3 (load average 0.00 when checked, network bound), so a
  warm run can still be slowed by it; hence the range, not a single number.
- The page cache: run 1 may be cold for the new venv's files. I will not drop caches, because that
  would disturb the other session's run.
- The mirror venv resolves `mcp==2.1.0` from `pyproject.toml`, where VPS2 has 2.0.0.
- py-spy samples perturb timing; no wall-time figure is taken from a profiled run.
- Code is `674d9a8b` (origin/master today); VPS2 serves `714d4a81`.
