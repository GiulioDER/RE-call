# Pre-registration: X-1, C9's baseline on the AML sources it has never been measured on

**Date:** 2026-09-25   **Status:** predicted, not yet measured

Branch `claude/mm1-mm3`. System: served C9 at `3eb447c4` (the round-two day-zero baseline). Local
directional measurement over public data; not an AML hosted evaluation.

## Why

Every round-two Textual decision so far rests on LoCoMo and BEAM, and every Multimodal one on
MemEye. AML's Textual track names seven sources (LoCoMo-Refined, ScriptMem, LongMemEval-Refined,
LongMemEval-S, CLBench, PersonaMem-v2, BEAM; memory `2026-09-18-aml-textual-track-evidence`) and its
Multimodal track seven columns (MobileMem-Omni, MemLens at four lengths, MemEye open and MCQ;
memory `2026-09-18-aml-multimodal-track-evidence`). C9 has never been run on LongMemEval,
ScriptMem, CLBench, PersonaMem, MobileMem or MemLens. Two consequences, both measured today:

- **Findings do not transfer by default.** 47 of 48 real Textual smoke Searches took the Code4
  route, while LoCoMo sends 27% to Context4; MM-1's retrieval prediction failed by an order of
  magnitude because it rested on the one MemEye scenario that least needed the image leg.
- **Round-two candidates have no held-out set.** T-1, MM-1 and MM-4 are measured on the same
  public sets that motivated them.

X-1 is not a treatment. It measures where C9 stands on each unmeasured source, names its weakest
categories there, and becomes the held-out check every round-two candidate must pass before it is
recommended.

## Scope

| Track | Source | AML pipeline in the pinned checkout (`1b8142bf`) | Scoring |
|---|---|---|---|
| Textual | LongMemEval-S | `data/longmemeval-s/pipeline.py` | binary judge, strict time granularity |
| Textual | ScriptMem | `data/scriptmem/pipeline.py` (4 script files) | exact option match, including multiselect and ordering |
| Textual | CLBench | `data/clbench/pipeline.py` | strict all-or-nothing rubric judge, plus rubric share |
| Textual | PersonaMem-v2 | `data/personamem/pipeline_v2.py` | mapped exact answer |
| Multimodal | MemLens (32K) | none public | MemLens's own evaluator, choice and open items |
| Multimodal | MobileMem-Omni (English, filtered) | none public | MobileMem's own evaluator |

LongMemEval-Refined is AML's refinement of LongMemEval and is not public; LongMemEval-S stands in for
it. Data are fetched from each source's public release at a pinned revision; a source that cannot be
obtained or parsed is reported as unavailable, never imputed.

## Subsets, fixed now

To stay inside the budget, each source is sampled once with `random.Random(20260925)`, stratified by
its own category field, and the sample (ids plus revision hash) is committed before any Add:

- LongMemEval-S: 120 of 500 questions, 20 per question type; one tenant per question (its own
  haystack).
- ScriptMem: all four scripts, all questions.
- CLBench: 200 items.
- PersonaMem-v2: 200 questions.
- MemLens 32K: 120 of 789 questions, stratified by question type.
- MobileMem-Omni filtered English: 2 of the 8 English users, all their questions.

If a source's ingest volume alone would exceed a quarter of the spend cap at the measured
DeepSeek compile rate, its sample is halved before anything is added, and the record says so.

## Arms

- **C9**: the served configuration.
- **C9′**: the same retrieval answered and judged a second time, interleaved per question (the
  reader and judge noise floor; retrieval is shared through the query-embedding cache).

No treatment arm: X-1 is the baseline that treatments are later checked against.

## What I predict

Reader and judge: `deepseek/deepseek-v4.1-flash`, pinned to one provider, reasoning off, bounded
output, AML's own prompts from the pinned checkout where they exist. AML's own reader differs (BEAM's
is Qwen3-14B), so these are reader-relative baselines. My past predictions ran too high for effects
and too low for retrieval headroom; for baselines I have no record either way, so the bands are wide.

| Source | Metric | Predicted |
|---|---|---|
| LongMemEval-S | accuracy, all 120 | 0.55 to 0.75 |
| LongMemEval-S | accuracy, temporal-reasoning (20) | 0.30 to 0.60, the lowest type |
| LongMemEval-S | evidence-session Recall@10 | 0.70 to 0.92 |
| ScriptMem | exact match | 0.30 to 0.60 |
| CLBench | strict accuracy | 0.10 to 0.35 |
| CLBench | rubric share | 0.50 to 0.75 |
| PersonaMem-v2 | accuracy | 0.40 to 0.65 |
| MemLens 32K | accuracy | 0.30 to 0.55 |
| MobileMem-Omni (EN) | accuracy | 0.25 to 0.50 |
| any source | \|C9′ − C9\| | at most 0.03 |

Directional predictions, which matter more than the levels: on LongMemEval-S, temporal-reasoning
scores lowest and knowledge-update second lowest; on MemLens and MobileMem, the questions whose
evidence is an image score below those whose evidence is text, since served C9 returns images only
on the 4.4% of questions that name the medium (MM-1 Stage 0).

## What would falsify this

X-1 has no treatment to falsify. What would falsify my picture of C9:

- LongMemEval-S temporal-reasoning not among its two lowest types.
- On MemLens or MobileMem, image-evidence questions scoring at or above text-evidence ones under the
  served route gate (which would mean MM-1's finding does not generalise).
- Any \|C9′ − C9\| above 0.05, which would make that source unusable as a held-out check at this
  sample size.

## How the result is used, fixed now

1. **Weakest categories.** For each source, the categories below the source's mean by more than the
   C9′ noise floor are listed as round-two targets, with their question counts.
2. **Held-out rule.** A round-two candidate is recommended only if, on at least one X-1 source it was
   not developed on, it is non-inferior to C9 (no drop larger than the noise floor) on that source's
   overall score. This applies to T-1, MM-1, MM-3 and MM-4 before any of them serves.

## How it will be measured

1. **Stage A (free):** fetch each source at a pinned revision; write an adapter per source mapping
   its sessions to AML Add requests and its questions to Search requests, following the pinned AML
   pipeline where one exists; dry-run each adapter and report Add, session and question counts and
   category labels; commit the samples.
2. **Stage B (Voyage and ingest compile):** on VPS3, one fresh C9 ingest per source subset, Search
   every question once at `top_k` 100, store the returned items; delete and verify every tenant.
3. **Stage C (answers):** C9 and C9′ answered and judged with the pinned prompts, interleaved per
   question, two workers.

**Spend.** Cap USD 25 for X-1 across ingest compile, answers and judges, DeepSeek only, at the USD 5
balance floor the user set on 2026-09-25. Never alongside another OpenRouter job: X-1's Stage B
starts only after the MM-1/MM-3 Stage 2 and T-1 LoCoMo runs have finished.

## Apparatus checks, fixed now

1. Every adapter's dry run matches the source's own published counts for the sampled items.
2. Every question's evidence (where the source labels it) is present in its tenant after ingest.
3. Valid-answer rate at least 98% per source; judge parse failures counted as not correct and
   reported separately.
4. Every tenant deleted and verified empty after its source.

## Confounds I can name now

- **These are AML's own public sources.** Nothing is tuned on them in X-1, and a later candidate
  checked against them is checked, not tuned; but a result here is not held out from AML itself.
- **Reader mismatch.** DeepSeek V4.1 Flash reads and judges; AML's reader and judge differ per
  dataset. Differences between sources are more trustworthy than absolute levels.
- **Subsets.** 120 to 200 questions per source resolve differences of roughly 0.10, not 0.03; small
  categories (20 questions) only flag, they do not measure.
- **LongMemEval-S is not LongMemEval-Refined.** AML's refined version is not public.

<!-- FROZEN PREREGISTRATION ENDS HERE. APPEND RESULTS BELOW. -->

## Results

No measurement had run when this record was committed.

### Stage A result, 2026-09-25

`scripts/aml_x1_sources.py` (`fetch`, `draw`, `materialize`, `dryrun`), run on VPS3 under
`nice -n 10` with an RSS watchdog (peak 380 MB; dry run 52 s). No model, embedding or C9 service
call; nothing sent. Every built Add and Search request validates against C9's own `AddRequest` and
`SearchRequest` from served `3eb447c4`: 0 failures. The draw is identical on VPS3 (Python 3.12) and
locally (Python 3.14). Outputs: `results/aml-x1/draw.json`, `results/aml-x1/dryrun.json` (with a
per-source `adds_sha256` so Stage B can prove it sends these exact requests); the materialised data
(1.2 GB) stays on VPS3 under `~/mm1-mm3/x1/`.

| Source (pin) | Drawn | Tenants | Adds | Text-only Adds | Image parts | Evidence present |
|---|---:|---:|---:|---:|---:|---|
| LongMemEval-S (HF `98d7416c`) | 120 | 120 | 5,705 | 5,705 | 0 | 118/118 |
| ScriptMem (GH `22ac7e7e`) | 457 | **unavailable** | | | | |
| CLBench (HF `b28a5832`) | 200 | 200 | 106 | 106 | 0 | not labelled |
| PersonaMem-v2 (HF `ed956dea`) | 200 | 128 | 15,041 | 15,041 | 0 | 179/179 |
| MemLens 32K (HF `afa101a1`) | 120 | 120 | 1,626 | 562 | 1,546 | 106/106 |
| MobileMem-Omni EN (HF `14c08631`) | 263 (users 11, 15) | 2 | 416 | 0 | 2,309 | 248/248 |

Apparatus check 1 (published counts) holds for every available source, with one README
discrepancy recorded (MobileMem's 48.2 turns per session is 48.02 in the data; totals match).
ScriptMem's per-script and per-type counts (457) match too, but its manifest's four sha256 values
do not match the released files.

Departures from the record, stated rather than silently taken:

1. **ScriptMem cannot be measured.** The public release carries only a two-utterance synthetic
   `format_example` per script, and its README says the script conversations are not released.
   Questions, options and answers are public; the memory to Add is not. It is reported unavailable,
   as the record requires, never imputed.
2. **CLBench barely tests memory under the only mapping available.** AML's CLBench ingest mapping
   is not in the pinned checkout; X-1 maps every turn before the final user turn to Adds. 129 of
   the 200 drawn tasks are single-turn, so their tenants are empty and the reference document sits
   in the query itself; 27 queries exceed 20,000 characters, which C9 cuts to head and tail.
   **Open, for the user before Stage B:** keep CLBench as mapped, drop it, or restrict it to its 71
   multi-turn tasks.
3. **Stratification.** LongMemEval-S is 20 per type as fixed. For CLBench, PersonaMem-v2 and
   MemLens the record gives totals only, so each is drawn by proportional largest-remainder
   stratification over its own category field.
4. **Mappings the record did not fix:** PersonaMem-v2 histories have no dates or sessions, so one
   Add per user-led round with no timestamp; its options are shuffled by a sha256 seed because
   AML's shuffle uses Python `hash()`, which changes per process. Timestamps are read as UTC
   everywhere, since no source states a zone. MobileMem images are read from the 6.25 GB
   `image.zip` by HTTP range request for the two drawn users only (2,309 images, CRC-checked).
5. **Data quirks kept as they are:** 21 drawn PersonaMem `sensitive_info` questions have their
   snippet deliberately absent from the history (counted unlabelled, not missing); 14 MemLens
   questions contain a literal `<image>` with no image; 22 MemLens and 416 MobileMem files carry an
   extension that disagrees with their bytes (MIME taken from the bytes); 2 LongMemEval-S items
   repeat a haystack session id.

**Ingest volume against the halving rule.** 22,414 text-only Adds carrying 86.3M characters would
each take one DeepSeek compile; LongMemEval-S holds most of the characters and PersonaMem-v2 most
of the Adds. Whether either exceeds a quarter of the USD 25 cap is decided at the start of Stage B
from a measured per-Add compile cost, before any source is ingested, and the record will say which
were halved.

### Amendment 1, 2026-09-25, after Stage A and before Stage B

**CLBench is restricted to the drawn tasks with at least one prior turn, on the user's decision
(open point 2 of the Stage A result).** The draw is not changed or topped up: of the 200 drawn
tasks, the 129 single-turn ones are dropped, because they store nothing and carry their reference
document in the query, so they do not test memory. CLBench's X-1 sample is therefore 71 tasks,
not 200, and its predicted bands (strict accuracy 0.10 to 0.35, rubric share 0.50 to 0.75) stand
unchanged, now read on those 71. With 71 items it resolves only large differences, so it enters the
held-out rule as a direction check. Implemented in `tenants_for` in `scripts/aml_x1_sources.py`;
the dry run is re-run so the per-source `adds_sha256` Stage B checks against covers exactly these
tasks. No Stage B request had been sent.

Amendment 1 applied, 2026-09-25: the dry run was re-run on VPS3 with the amended script (the Stage A
files kept beside it as `*.stageA.*`). CLBench now has 71 items and 71 tenants, none empty; its Adds
(106) and `adds_sha256` are unchanged, as they must be, since the dropped tasks had no Adds. Every
other source's counts and digest are identical to Stage A. `results/aml-x1/dryrun.json` is the
amended run.

### Amendment 2, 2026-09-26, before any Stage B ingest: Add-time compile off

**What happened.** The Stage B cost probe (`scripts/aml_x1_stageb.py probe`, arm B on VPS3, the
Add-time compiler set to `deepseek/deepseek-v4.1-flash` because X-1 is DeepSeek-only) showed the
anchored compile failing on long LongMemEval-S Adds: the model fills the compiler's 2,400-token
output bound, the JSON is cut off, the compile raises `ValueError`, retries three times and falls
back, about 85 seconds and about USD 0.013 per Add for nothing kept. Projected over LongMemEval-S
alone that is about USD 75 and a day of ingest, three times this record's cap, and it would not
measure the served C9 either, which compiles with gpt-4o-mini. The probe was allowed to finish so
the failure rate per source is on record; its numbers are appended when it ends.

**What changes, decided by the user 2026-09-26.**

1. **Stage B ingests with the Add-time compile off** (`RECALL_AML_COMPILER=0`, new, experiment
   only). Adds store raw windows only; atomic views are still built from them. Compiled records
   are therefore absent from the graph and from the Context 4 specialist index. X-1's baseline is
   **C9 without the Add-time compile**, called C9-raw below; the closest measured precedent is
   "remove compiled records from the context route", −0.95 inside a −2.38 drift (2026-09-24).
2. **gpt-4o-mini is not used** for the compile (the user's DeepSeek-only rule stands).
3. **The halving rule is applied as written**: a source is halved if its ingest alone would exceed
   a quarter of the USD 25 cap. With the compile off, ingest spends nothing from the cap (Voyage
   embedding is outside it, as before), so the rule is not expected to trigger. If it did, the
   kept half would be, within each category, the first half of the drawn ids ordered by
   `sha256("x1-halve:" + id)`, rounded up, which depends on ids only.
4. **The raised compile bound is measured separately, not used in Stage B.** A new experiment
   override, `RECALL_AML_COMPILER_MAX_TOKENS` (default 2,400, unchanged when unset), is probed on
   20 LongMemEval-S Adds at 8,000 tokens with DeepSeek. Prediction, written now: fallback share
   falls from above 0.50 at 2,400 to at most 0.20 at 8,000, and cost per Add is USD 0.005 to
   0.02. It informs A-1 and any later DeepSeek compile run; it does not change X-1.

**What it does to the rest of the record.** Predictions are unchanged and are now read against
C9-raw. The held-out rule is unchanged: candidates (T-1, MM-1, MM-3) act at Search time, so each is
compared with C9-raw on the same stored retrieval. A statement about the served C9's absolute level
on these sources cannot be made from X-1; a statement about a Search-time candidate's difference
can.
