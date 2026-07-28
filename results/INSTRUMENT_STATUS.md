# Instrument status — which abstention claims are checkable

Written 2026-07-28 against `origin/master` @ 9eb3bc1. Read from the artifacts in `results/`,
not from `FINDINGS.md`'s account of itself.

**current** — measured on the shipped pipeline, artifact retained.
**stale** — measured on a superseded configuration.
**unfalsifiable** — no artifact retained; the claim cannot be checked at any cost short of a re-run.

| claim | status | artifact | notes |
|---|---|---|---|
| §9b LOCOMO abstention, 4 modes | current | `locomo/postfix_abstention.json` | post-#81/#84. `locomo_abstention.py:168` passes `calibration=cal` explicitly, so #101's auto-load bug never reached it |
| §9b abstention with rerank on | **unmeasured** | — | #103 measured the default mode only (0.00, unchanged, confirmed identical across `baseline.json`/`rerank_modern.json`/`rerank_shipped.json`). The calibrated and judge modes have never been crossed with a reranker |
| §9c entailment ROC sweep | stale, no retained artifact | — | the re-run (`postfix_entailment_sweep.log`) died after 9 conversations (conv-26, 30, 41, 42, 43, 44, 47, 48, 49); no JSON was written and nothing noticed. That log is gitignored (`results/locomo/*.log`) and worktree-local — not present in this checkout, not retained in git. The original pre-#81/#84 measurement has no retained raw artifact either |
| §10 LongMemEval, all rows | **unfalsifiable** | — | pre-#81/#84; indexes and output discarded. 6h39m to rebuild the merged index alone |
| §7 private-corpus abstention | current, not independently checkable | — | corpus is private |
| §8 PEP abstention | current | — | public corpus and questions; cheap to re-establish |
| every row above | **no row count** | — | no pre-Task-2 artifact records the corpus it measured |

## What this gates

No combined signal, entity-mismatch feature or abstention-policy change is fit against a row
marked **stale** or **unfalsifiable** until that row is re-measured or explicitly demoted.
