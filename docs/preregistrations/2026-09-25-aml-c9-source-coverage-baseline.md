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
