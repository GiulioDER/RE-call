# Handoff: run the late-interaction measurement

Paste everything below the line into a new session. It is self-contained.

---

## Task

Execute Tasks 10 and 11 of
[`docs/superpowers/plans/2026-08-07-late-interaction-rerank.md`](../plans/2026-08-07-late-interaction-rerank.md):
score the frozen MTRAG pools with ColBERT, run the validation gates, compute the preregistered
contrasts, and append a RESULTS section to the preregistration.

**The instrumentation is already built, reviewed and merged.** Tasks 1 to 9 are done. You are
running an experiment, not writing a library.

## Read these first

- Preregistration: `docs/superpowers/specs/2026-08-07-late-interaction-rerank-design.md`.
  **Everything above its RESULTS heading is a commitment and must not be edited.**
- Plan: `docs/superpowers/plans/2026-08-07-late-interaction-rerank.md`, Tasks 10 and 11.

## What already exists

| thing | where |
|---|---|
| `LateInteractionReranker`, `maxsim`, `maxsim_or_last`, licence registry | `recall/rerank.py` |
| arms, `holm_family`, `score_stream`, `score`/`validate` CLI | `benchmarks/mtrag/late_interaction.py` |
| power calculation | `benchmarks/mtrag/buried_gold_power.py` |
| `compare_orderings`, `score_delta`, `rerank_order` | `benchmarks/mtrag/rerank_offload.py` |

153 tests pass across 8 files. `fastembed` is already a declared extra and already in `uv.lock`.

## ⚠️ Task 9 already ran, and it changed the experiment

**Family B is DEMOTED TO DESCRIPTIVE. It carries no p-value.** Do not compute one for it.

| | |
|---|---|
| joint 2x2 over the 123 | both bury 78, only MiniLM 12, only BGE 13, neither 20 |
| rho | 0.7967 |
| minimum detectable bury rate at 0.80 power | 0.2917, against a control of 0.7317 |

Detecting anything in Family B would need ColBERT to rescue more than half of the 90 buried
documents. BGE, 25x MiniLM's size, rescued essentially none (91 against 90). **Family A carries the
verdict alone.** The demotion followed a rule fixed before the number was seen; it is not
renegotiable now that it is inconvenient.

**Family C's V1 width is 100.** The 2026-08-07 equal-width protocol capped every arm at 100. The
383 figure is `mq_nested2_nogold`'s WHOLE-POOL depth, from a table explicitly flagged as
width-confounded. Do not use 383.

## 🛑 The blocker nobody wrote down

**`docs.jsonl` was never archived. Only its SHA256 was.**

`/var/lib/recall-benchmarks/2026-08-07-mtrag-rerank-conversion/payload/` contains `pairs.jsonl`,
`queries.jsonl` and `docs.jsonl.sha256`, but not `docs.jsonl` itself. The document texts must be
regenerated from the `recall_splade` database on VPS2 before anything can be scored.

**This is recoverable and, unusually, verifiable**: regenerate, hash, and compare against the
archived `docs.jsonl.sha256`. A match proves the regeneration is byte-identical to what the earlier
run scored, which is a stronger guarantee than the archive itself would have given. **If the hash
does not match, stop.** A near-identical corpus silently produces numbers that look fine and are
not comparable to the baseline.

## Environment

VPS2: `root@100.91.148.25`, key `~/.ssh/contabo_sentiment`. Verified reachable 2026-08-07.

| path | what |
|---|---|
| `/var/lib/recall-benchmarks/2026-08-07-mtrag-rerank-conversion/` | pairs, queries, scores for MiniLM and BGE, rankings |
| `/var/lib/recall-benchmarks/2026-08-06-mtrag-splade-learned-sparse/` | frozen pools in `results/pools/` |
| `/var/tmp/re_call_mtrag_20260803/mt-rag-benchmark/` | corpora and qrels. ⚠️ `/var/tmp` is NOT durable, check it exists |
| database `recall_splade` | `recall_mtrag_bge_v1_{clapnq,cloud,fiqa,govt}`, 366,479 passages |

⚠️ **Another session shares this host.** It was at load 33.78 on 2026-08-07. Check load before a
long run and coordinate before killing anything. Never run DDL against `sentiment_agent`, which is
the money-path database.

## The order to do things in

1. Regenerate `docs.jsonl` from `recall_splade` and **verify against the archived SHA256.**
2. **G1 reproduction:** re-run `rr_minilm` through `rerank_offload` and require 0.7603 R@100 and
   0.3769 nDCG@5 to four significant figures. **If G1 fails, STOP.** It means the path is wrong,
   not that reproduction is hard: the 2026-08-07 run already reproduced these on different hardware
   through a different code path.
3. Score `li_colbertv2`, then `li_answerai`. G3 completeness runs automatically at the end of each.
4. **G2 validate** each arm. A MISMATCH stops the run.
5. Score `li_jina` LAST, and only with `--accept-noncommercial`. Its numbers are diagnostic only.
6. Contrasts via `analyse_contrasts.py`, unchanged. **Build the Holm family with
   `holm_family()`**, never by hand: it is the gate that makes `li_jina` mechanically unable to
   enter a shipping-relevant family, and until you call it, it protects nothing.
7. Append RESULTS. Archive under `/var/lib/recall-benchmarks/YYYY-MM-DD-mtrag-late-interaction/`
   with a SHA256 manifest and a `NOTE.md`.

## Traps, each of which has already cost time here

1. **`--scores` is written non-atomically.** A crash mid-run leaves a partial file with a
   valid-looking header and no checkpoint. Every downstream consumer raises rather than producing a
   number, so this is a wasted-run risk and not a wrong-number risk, but on a multi-hour job
   consider writing to a temp path and renaming on success before you start.
2. **`ef_search` silently truncates.** A retrieval path once returned 6 candidates of 100 for
   weeks. No test caught it; a timing anomaly did. Any new retrieval path must assert it returned
   the k it asked for.
3. **`|user|: ` prefixes every MTRAG-human turn.** `strip_speaker` in `benchmarks/mtrag/run.py`
   removes it. This does NOT apply if you reuse the existing dump, where it is already stripped.
4. **`pkill -f <pat>` matches its own ssh command line** and kills the shell running it. Use
   `[p]at`.
5. **Heredocs over ssh mangle quotes.** Use `scp` for anything non-trivial. `scp` takes `-P`,
   `ssh` takes `-p`.
6. **Check liveness of the JOB, not one of its stages.** A tarball was once hashed mid-write
   because the encoder had exited while `tar` was still running.
7. **Never run multiple pytest files in one invocation on the dev machine.** A shared local
   Postgres produces spurious setup errors that are not real failures. One file at a time, and
   re-run a red file alone before believing it.
8. **Query side and document side must come from the same checkpoint and the same `fastembed`
   version.** The header records both. Re-verify after any venv change.

## The decision rule, which is already fixed

- **POSITIVE**, follow-on project justified: C1 (`li_colbertv2 − rr_minilm`, nDCG@5) point estimate
  **≥ +0.010** and Holm-significant within Family A, with no veto tripped.
- **Vetoes**, on "CI excludes zero" rather than Holm: a regression in R@100 or nDCG@10.
- **NULL**: C1 below the bar or its CI includes zero.
- **The capacity reading**, `li_jina`'s only permitted use: a shared null across all three licenses
  "capacity does not appear to be the binding constraint over 33M to 560M", **and nothing
  stronger**. The spread here is 5x against the primary arm and 17x at its widest, not the 25x that
  closed the reranker lever.

## Predictions recorded before any score

P1: C1 positive but small, +0.005 to +0.015, straddling the bar. P2: withdrawn in effect, since
Family B is descriptive. P3: V1 smaller than C1. P4: no GPU rental needed.

The stated distrust, which is the thing to watch: the 123 are already lexically anchored, so
ColBERT's token maxima may simply reproduce SPLADE's judgment and fail P1 and P3 together. **If
that happens, the correlated failure is the finding**, and it is a stronger result than either
prediction landing.

## Known minor items, triaged as "fine to leave" by the final review

Recorded so you do not rediscover them: `LATE_INTERACTION_MODELS` is a mutable dict; `score_stream`
silently drops a `pairs` doc_id absent from the `docs` stream (`assert_complete` catches it one
layer up); `rerank_offload.cmd_validate`'s printed verdict ignores `worst_delta` while its exit code
checks it, so it can print MATCH and exit 1 (pre-existing, and the exit code is what CI reads).
