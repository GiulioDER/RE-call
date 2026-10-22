# Third party text in `benchmarks/benchd/`

This directory is a research harness for the Bench'd benchmark. It is not part of any package
RE-call ships. Two files carry text from the Bench'd harness. Source and licence were read on
2026-10-20.

| | |
|---|---|
| Source | https://github.com/benchdai/harness |
| Commit | `bd4824fe5efc095151115c45219c1ef0baa88132` (2026-06-06), the head of `main` when it was cloned on 2026-08-23 |
| Copyright | Copyright (c) 2026 benchdai |
| Licence | MIT. A copy, byte for byte the file at that commit, is in [LICENSE-benchd-harness.txt](LICENSE-benchd-harness.txt). |

What is used from it:

| File here | What it holds | Changes |
|---|---|---|
| `count_tokens.py` | `ANSWERER_PROMPT` and `JUDGE_PROMPT`, copied from `benchd_harness/scoring/llm_judge.py` so that token counts re-render the harness's exact prompts | none; both strings occur verbatim in that file at the commit above (checked 2026-10-20) |
| `harness-workers.patch` | a patch to `benchd_harness/cli.py` and `benchd_harness/runner.py` adding a `--workers` option | it is RE-call's change to those files; its `index` lines name the blobs of both files at the commit above |
