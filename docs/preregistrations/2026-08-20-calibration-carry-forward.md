# Pre-registration: calibration carry-forward across a bounded corpus delta

**Date:** 2026-08-20   **Status:** measured 2026-08-20; every prediction below is unedited

Tenant `memory` on VPS2 (`recall_repos`, port 55432), embedder `fastembed:BAAI/bge-large-en-v1.5`.

## The question

When a generation is rebuilt to absorb a small corpus delta, does the certified calibration
already published for the parent generation still clear the certification bar when its stored
labelled query set is re-scored against the child generation, without refitting the threshold?

Answerable by a number: the separability CI lower bound on the child generation, against
`MIN_SEPARABILITY = 0.90`.

## Why this is being asked

Today the answer is structurally "no", and not because of a measurement.
`CalibrationRepository.resolve` looks up a published calibration by the ACTIVE `generation_id`
and then re-checks `pipeline_fingerprint` and `corpus_fingerprint`. A rebuild produces a new
`generation_id` with no calibration row, so resolution returns `STALE`, which maps to
`CALIBRATION_STALE`: strict refuses every query, development degrades every query. There is no
rebind, no carry-forward and no delta tolerance anywhere in the tree.

The proposed mechanism is not a looser gate. It re-runs the SAME stored evidence against the new
lineage and publishes a carried-forward artifact only if that evidence still certifies. The
invariant is preserved: a certified threshold remains one that has been shown to separate on the
generation it actually serves.

## State measured today, before any change

Measured 2026-08-20 by direct SQL against `recall_repos` and by driving the MCP server over stdio.

| fact | value |
|---|---|
| active generation, tenant `memory` | `gen_f15666c398f7488fbcdc9327ee0d24ae`, `corpus_version memory-2026-08-17` |
| published calibration | `cal_17ae06a8eb5a4504a06ac3e2565dc880`, threshold **0.712**, AUC **0.9886**, CI low **0.9566**, n = 22 answerable / 28 unanswerable, certified |
| chunks in the generation | 8,671 from 1,080 sources, last indexed 2026-08-17 21:30 UTC |
| stored labelled query set | 50 queries, `recall_calibration_query_sets`, each with a boolean `answerable` |
| live memo files on this workstation | **1,094** |
| memo files added or modified since the staged snapshot (2026-08-16 14:35) | **118**, i.e. **10.8%** of 1,094 |

The corpus delta under test is therefore **118 of 1,094 files, 10.8%**.

Two facts about the serving path, recorded here because they change what "it does not work" means
and both were measured today rather than assumed:

- VPS2 runs the MCP server with `RECALL_ENV=development`, so `recall_mcp.server` builds a plain
  `PgVectorStore` on the legacy `chunks` table and never reads `recall_generations` or
  `recall_calibrations`. Probe result: `failure_code INDEX_NOT_READY`, `trust_state degraded`,
  `abstained False`. The same probe with `RECALL_ENV=production` returns `failure_code None`,
  `trust_state trusted`, `abstained True`. Both published calibrations are inert on the served
  path today.
- The legacy `chunks` table holds **8,716 rows for tenant `memory`**, not zero. The memory note
  `generation-tenants-are-invisible-to-the-legacy-store` records "0 in chunks" for both calibrated
  tenants; that is correct for `re-call-code-gen` (a tenant name that exists only in
  `recall_chunks_v1`) and **wrong for `memory`**. This matters because it is why development mode
  returns plausible stale hits rather than nothing at all.

## What I predict

**P1. Carry-forward succeeds at this delta.** Re-scoring the 50 stored labelled queries against
the child generation, holding the threshold at 0.712, the separability CI lower bound stays at or
above **0.94** and therefore clears the 0.90 bar. Point AUC lands in **0.96 to 0.99**, i.e. at or
slightly below today's 0.9886.

**P2. The error moves in one direction, and I can name it in advance.** A top-1 cosine is a max
over the indexed set, so adding documents can only leave it unchanged or raise it. Holding the
threshold fixed:

- false abstains (answerable scoring below 0.712) **can only fall or stay**, from the recorded
  4.5% to somewhere in **0% to 4.5%**;
- false confirms (unanswerable scoring at or above 0.712) **can only rise or stay**, from the
  recorded 3.6% to somewhere in **3.6% to 11%**.

If false abstains RISE, the monotonicity argument is wrong and something other than corpus
addition changed, most likely one of the 118 files being a modification that replaced better text.

**P3. The rebuild is cheap now, and the warning in `build_generation.sh` is obsolete.**
`_reuse_source` copies chunks across generations on matching `source_sha256` and
`pipeline_fingerprint`, and the parent generation now exists in `recall_chunks_v1`, which it did
not on 2026-08-17. Predict **at least 89% of the 1,094 sources are reused rather than re-embedded**,
and the child build takes **no more than 15% of the parent build's wall clock**.

**P4. The mechanism is needed at all.** Before the change, a query against the child generation
returns `CALIBRATION_STALE` in strict mode and no corpus text. Predicted with high confidence:
this is a code-path certainty, not a measurement, and it is stated so that a green result cannot
be claimed without demonstrating the red one first.

## What would falsify this

- **P1 falsified** if the CI lower bound on the child generation is below 0.90, or if the point
  AUC falls below 0.96. Either means a 10.8% delta is already too large to carry a threshold
  across, and the feature must either bound the delta far tighter or refit rather than rebind.
- **P2 falsified** if false abstains rise above 4.5%, or if false confirms exceed 11%.
- **P3 falsified** if fewer than 89% of sources are reused, or the child build exceeds 15% of the
  parent's wall clock.
- **P4 falsified** if the pre-change child generation answers rather than refusing under strict
  policy, which would mean the gate I am describing is not the gate that runs.

## How it will be measured

1. Ship the 118 changed memo files to VPS2 and re-manifest, then build a child generation with
   `recall.cli generation build` against the same pipeline identity as the parent. Record reused
   versus re-embedded source counts from the build output, and wall clock for both builds.
2. **Verify the apparatus before believing the outcome.** Confirm the child generation is served,
   by asking it a question whose answer exists ONLY in one of the 118 new files and checking that
   the answering chunk is that file. Today's negative control is recorded above: the query
   "why is a substring test wrong for deciding directory identity" does NOT retrieve
   `a-substring-test-for-a-directory-identity.md` from the parent generation, and abstains at
   `trust_state trusted`. If the child generation abstains on the same query, the ingest failed
   and no calibration number from that run means anything.
3. Under strict policy, query the child generation before publishing any carried-forward
   artifact. Record the failure code. This is P4.
4. Re-score all 50 stored labelled queries against the child generation, at the same retrieval
   profile the server uses. Metrics, each named by its denominator:
   - **separability (AUC)** over 22 answerable x 28 unanswerable pairs, with the Hanley and
     McNeil interval already implemented in `recall.calibration.separability_interval`;
   - **false abstain rate**, over the 22 answerable queries, at a threshold held at 0.712;
   - **false confirm rate**, over the 28 unanswerable queries, at the same threshold;
   - **source reuse rate**, over the 1,094 sources in the child manifest.
5. Only then decide whether the carried-forward artifact certifies.

## What I already know

- `calibrated-thresholds-and-the-overlap` (measured 2026-08-17): memory threshold 0.7100,
  separability 0.989, LOO false-confirm 3.6% and false-abstain 4.5%. Critically, **in 4 of 4
  corpora the answerable and unanswerable distributions OVERLAP**: memory
  min(answerable) - max(unanswerable) = **-0.048**. The threshold is a least-bad cut sitting
  INSIDE an overlap region, not a boundary. That is the reason to expect it to be sensitive to
  which documents are present, and the reason not to read a high AUC as a clean separation.
- `index-not-ready-needs-a-generation` (confirmed 2026-08-17): there is no config switch that
  makes a calibration apply without a generation, and at that time `_reuse_source` had nothing to
  reuse because the corpus lived in the legacy `chunks` table with zero generations. **That
  precondition has since changed**, which is exactly what P3 tests.
- `docs/preregistrations/2026-08-15-bge-large-voyage-splade-memory-corpus.md` and
  `2026-08-16-generated-calibration-query-sets.md` hold the fits these numbers come from.

## Confounds I can name now

1. **Label rot is the dangerous one.** The 50 queries were labelled against the 2026-08-17 corpus.
   A query labelled `answerable: false` may have become genuinely answerable BECAUSE one of the
   118 new memos now answers it. That would present exactly as "false confirms rose", i.e. as the
   feature failing, when the label is what changed. **Every unanswerable query that crosses 0.712
   on the child generation must be read by hand and its retrieved chunk inspected before it is
   counted as a false confirm.** A count taken without that inspection is not a measurement.
2. **The labels were authored by me and have never been operator-reviewed.** Carried forward
   unretired from the 2026-08-17 record.
3. **Sample sizes sit on the bar.** 22 answerable and 28 unanswerable, against
   `MIN_CALIBRATION_SAMPLES = 20`. The interval is wide by construction and the smaller class
   dominates it, so a point AUC moving inside the interval is not evidence of anything.
4. **Monotonicity in P2 is a top-1 dense argument.** The served path is hybrid with a gap
   threshold, so the guarantee is directional rather than strict. A small violation is not by
   itself a falsification of the mechanism, only of the clean form of the argument.
5. **One tenant, one delta, one embedder.** Nothing here licenses a general claim about how large
   a delta a calibration survives. It measures 10.8% on `memory` with bge-large, and that is all.

## Result (2026-08-20)

**Status:** measured

Measured on VPS2, tenant `memory`. Parent `gen_f15666c398f7488fbcdc9327ee0d24ae`
(`memory-2026-08-17`), child `gen_c5c87c56c24048cb8b8e6296656150e5` (`memory-2026-08-20`), both
built with `fastembed:BAAI/bge-large-en-v1.5`. The scoring pass wrote nothing and used only the
code already deployed on that host, so it is independent of whether the carry-forward artifact
path is installed.

**Apparatus check, step 2, done before reading any calibration number.** The stored query set
re-canonicalised to its digest `504255095b09fe01…`, 50 queries, and the child reports 1,149
sources against the parent's 1,080. The negative control recorded in the prediction still holds:
the parent generation does not retrieve `a-substring-test-for-a-directory-identity.md`.

### The corpus delta, which the prediction had to estimate and the manifests now state

| | predicted | measured |
|---|---|---|
| changed sources | 118 of 1,094 = **10.8%** | 69 added, 38 modified, 0 removed of 1,149 union = **9.31%** |

The prediction's denominator counted this workstation's live stores; the measured one is the
difference between the two committed manifests, which is the quantity the mechanism actually uses.
They are close enough that the prediction was aimed at the right magnitude, and the difference is
not a correction to anything: it is two different populations, counted on purpose.

### P1. Carry-forward succeeds at this delta — **CONFIRMED**

| | predicted | measured |
|---|---|---|
| separability CI lower bound | >= 0.94 | **0.9528** |
| point separability | 0.96 to 0.99 | **0.9870** |

Both inside the predicted band, and 0.9528 clears the 0.90 certification bar. The parent's own AUC
was 0.9886, so the ordering degraded by 0.0016 across a 9.31% corpus delta.

### P2. The direction of the error — **FALSIFIED, and the prediction named its own cause**

| at the inherited threshold 0.712 | predicted | measured |
|---|---|---|
| false abstain, over 22 answerable | 0% to 4.5%, **can only fall** | **9.09%** (2 of 22) |
| false confirm, over 28 unanswerable | 3.6% to 11%, can only rise | **3.57%** (1 of 28) |

**False abstains doubled and false confirms did not move.** That is the opposite of the predicted
direction on both halves.

The argument was that a top-1 cosine is a max over the indexed set, so adding documents can only
raise it. The argument is sound and the premise is false: **38 of the 107 changed sources were
modifications, not additions**, and a modified file replaces its earlier text and re-chunks. A memo
whose answering passage was rewritten, or merely split differently, can score lower than it did.
The prediction stated this in advance as the thing that would explain a rise, so what is falsified
is the monotonicity claim, not the diagnosis.

The two near misses are both genuine answerable queries the child now abstains on:

```
0.6678  why did writing a file with python change every line ending
0.7100  why did my branch look like it had regressed
```

The second misses the threshold by **0.0020**. A refit on the same scores chooses **0.707**, which
would catch it. So the inherited threshold is now measurably worse on this corpus than a fresh fit,
by one query, while still certifying — which is exactly the drift `refit_threshold` was added to
make visible rather than to act on.

**The single crosser is a real false confirm, not label rot.** The pre-registration named label rot
as the dangerous confound and required every crosser to be read by hand. The one crosser is:

```
0.7158  how did the GraphQL schema migration affect latency
```

There is no GraphQL anywhere in these projects, so the `answerable: false` label is still correct
and this is an honest false confirm rather than a query the new memos started answering. The
confound was checked and did not fire.

### P3. The rebuild is cheap now — **CONFIRMED, both parts**

| | predicted | measured |
|---|---|---|
| sources reused | >= 89% | **1,042 of 1,149 = 90.7%** |
| child wall clock as a fraction of the parent's | <= 15% | **73.2 min / 722 min = 10.1%** |

The reuse count is exactly `1,149 - 107`, i.e. every unchanged source was copied and every changed
one re-embedded, with nothing else touched. Parent wall clock is taken from
`generation_build.log`'s create-to-last-write span (2026-08-17 09:28:25Z to 21:30:35Z), not from
`ready_at`, which includes a `validate` run started the following day and would have overstated it
at 1,422 minutes.

**The warning in `bin/build_generation.sh` is now obsolete and should be corrected in place**: it
says the build re-embeds all 1,080 files, which was true only while the corpus lived in the legacy
`chunks` table with no generation to reuse from.

### P4. The mechanism is needed at all — **CONFIRMED**

```
gen_f15666c398f7488fbcdc9327ee0d24ae  status=certified  artifact=cal_17ae06a8eb5a4504a06ac3e2565dc880
gen_c5c87c56c24048cb8b8e6296656150e5  status=stale      artifact=None
failure code for the child: CALIBRATION_STALE
```

Measured before any new code was installed. Strict policy refuses every query against the child.

### What the implemented mechanism decides on these numbers

Certified. Separability CI low 0.9528 >= 0.90, false abstain 0.0909 <= 0.10, false confirm
0.0357 <= 0.10. **The false-abstain rate sits at 91% of its bound**, so on this trajectory the next
comparable delta is the one that fails, and that is the intended behaviour rather than a margin to
be widened.

### The gap that teaches something

The prediction treated a corpus delta as a single quantity and reasoned about it as if it were all
additions. It is not: **additions and modifications move the error in opposite directions**, and
only additions are monotone. 69 additions pushed false confirms up, 38 modifications pushed false
abstains up, and the two effects landed on different classes. A delta bound that does not separate
them cannot predict which way a threshold will fail, and `corpus_delta` as implemented returns the
counts separately for exactly this reason, even though the bound is applied to their sum.

### Not measured here, and it matters more than any number above

The live MCP servers on VPS2 query tenant `memory` with **`voyage:voyage-4`**, while the legacy
`chunks` table those servers read holds **8,716 chunks under `bge-small-symmetric-v1`** and 2 under
`voyage:voyage-4`, and the calibrated generation measured above is **`bge-large`**. Three models on
one tenant, all at 1,024 dimensions, so nothing raises. Every number in this record concerns the
bge-large generation path and says nothing about what those servers currently return. Recorded here
because a green carry-forward result must not be read as "vps2 memory search is working".
