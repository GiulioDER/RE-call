# Third party text in `benchmarks/beam/`

This directory is a research harness. It is not part of any package RE-call ships. Two of its
files carry evaluation prompts copied from other projects, so that a score is produced with the
benchmark's own text rather than a paraphrase of it. Sources and licences were read on 2026-10-20
at the commits named below.

## `prompts.py`: mem0ai/memory-benchmarks, Apache-2.0

| | |
|---|---|
| Source | `benchmarks/beam/prompts.py` in https://github.com/mem0ai/memory-benchmarks |
| Commit | `4b61c5d31b9c668a12b4f5e78064248a02c82d2b` (2026-05-13) |
| Copyright | Copyright 2025 Mem0.ai |
| Licence | Apache License 2.0. A copy of the licence, byte for byte the file at that commit, is in [LICENSE-mem0-memory-benchmarks.txt](LICENSE-mem0-memory-benchmarks.txt). The repository has no NOTICE file at that commit. |
| Changes | A header docstring was added at the top of the file. The prompt bodies are unchanged. |

Vendoring details and the reason for copying verbatim are in [UPSTREAM.md](UPSTREAM.md).

## `aml_c9_probe.py`: AML's public BEAM pipeline, which copies BEAM

`aml_c9_probe.py` copies, from `data/beam/pipeline.py` in
https://github.com/AML-memory/agent-memory-leaderboard at commit
`1b8142bfe0f20f1c5218d6b554aa0012de34e504`, these strings:

| Constant in `aml_c9_probe.py` | Origin, as `pipeline.py` itself records it | Licence |
|---|---|---|
| `ANSWER_PROMPT` | verbatim `src/prompts.py:answer_generation_for_rag` of https://github.com/mohammadtavakoli78/BEAM at `3e12035532eb85768f1a7cd779832b650c4b2ef9` | MIT, Copyright (c) 2025 Mohammad Tavakoli; copy in [LICENSE-BEAM.txt](LICENSE-BEAM.txt) |
| `JUDGE_BASE` | verbatim `src/prompts.py:unified_llm_judge_base_prompt` of BEAM at the same commit | MIT, as above |
| `BATCH_OUTPUT_FORMAT`, `EQUIVALENCE_SYSTEM` | written in AML's pipeline | **none declared**: agent-memory-leaderboard has no LICENSE file at that commit and GitHub detects no licence for it; its README states that adapted evaluators remain subject to their upstream licences |

Changes from the source text: trailing spaces at line ends were removed from `ANSWER_PROMPT` and
`JUDGE_BASE`, and `JUDGE_BASE` stops after the `## OUTPUT FORMAT:` heading, leaving out the
single item JSON instruction that follows it there. AML's batch judge drops the same text and
puts `BATCH_OUTPUT_FORMAT` in its place. The words are otherwise unchanged.

The scoring rule the probe reproduces (the 0, 0.5 and 1 rubric scale, and event ordering scored as
alignment F1 times normalised Kendall tau) follows the same pipeline. `COVERAGE_PROMPT` and
`QUOTE_PROMPT` are RE-call's own.

BEAM's MIT licence text, byte for byte the file at the commit above, is in
[LICENSE-BEAM.txt](LICENSE-BEAM.txt).
