# Pre-registration: MM-4, machine-read image text as a sidecar at Add

**Date:** 2026-09-25   **Status:** predicted, not yet measured

Branch `claude/mm1-mm3`. Baseline: served C9 at `3eb447c4`, with the multimodal scope MM-1 selects
(see Arms). Local directional experiment over public data; not an AML hosted evaluation.

## Why

The official Multimodal Full scored 31.64, with information_extraction 22.13, direct_recall 23.60
and visual_reasoning 19.35. C9 stores an image message's supplied text and the image, and nothing
read from the image: multimodal Adds skip the compiler entirely (`HostedService.add`), and
`content_text` keeps raw windows free of generated claims by design. So a question about what is
written on a receipt, how many chairs are in a room or what a screen showed can only be answered if
the reader is handed the image and reads it itself, which MM-1 found happens on 4.4% of public
questions under the served route gate.

The repository already describes the missing piece: `docs/MULTIMODAL.md` specifies a
`MultimodalSidecar` of bounded text (OCR, captions) beside the original object, and the C8
production-direction memo (`2026-09-22-c8-multi-tenant-production-direction`) recommends a
modality-aware text sidecar with OCR, caption, entities, time and location. ATM-Bench reports that
schema-guided text memory holds up better under distractors than raw media; MemEye's authors report
that captions lose instance- and pixel-level detail. Nothing in the records rejects the idea; it
was never built for C9.

## The mechanism, fixed now

1. **Extraction at Add.** For every image part of an Add, one vision call with a fixed prompt that
   asks for, as short labelled lines and only what is visible: all legible text verbatim; objects
   with counts, colours and positions; people by appearance only (no identity guesses); setting;
   any visible date or time; the app or screen state if it is a screenshot. Output capped at 300
   tokens. Model in this experiment: `deepseek/deepseek-v4.1-flash`, pinned to one provider,
   temperature 0, reasoning off (user instruction 2026-09-25: no gpt-4o-mini until experiments
   finish; it accepts images, checked the same day). AML's Open-source division expects gpt-4o-mini
   at Add, so any recommendation is confirmed with gpt-4o-mini before it could serve (Decision rule).
2. **Storage.** One sidecar row per image in its own namespace (`image_text_tenant(tenant)`,
   parallel to the existing media and multimodal namespaces), text = the extraction, `primary_id` =
   the image message's raw parent id, embedded with the primary profile (`voyage-code-4-v1`). Raw
   windows are untouched, so `content_text`'s rule still holds for everything C9 already stores.
3. **Retrieval, M4r** (flag `image_text_leg`). At Search, a dense query over the sidecar namespace,
   top 20, fused into the main ranking by the multimodal visual leg's RRF (`fuse_hits`), each hit
   standing for its parent. A tenant with no sidecar rows gets exactly today's Search output.
4. **Shown, M4s** (flag `image_text_shown`, implies M4r). Each returned image-bearing item also gets
   its sidecar text appended as one text part, labelled `[machine-read from the image]`, so a reader
   that is not given the image, or reads it at low detail, still gets the text in it.

Both default off, with experiment overrides, as for MM-1 and MM-3. No part of 1 to 4 is tuned on a
measured set.

## Arms

MM-1 and MM-3 are still being measured. MM-4 is compared against whatever scope C9 would serve after
them, fixed before MM-4 runs:

- **S** = MM-1's recommended scope (preserve or dual) with MM-3 if recommended; otherwise the served
  route scope. Written into an amendment the moment MM-1's result is recorded.
- **S′** = S again, answered separately (reader noise floor; retrieval is identical through the
  shared query-embedding cache, as MM-1 found).
- **M4r** = S + the sidecar leg.
- **M4s** = S + the leg + the shown text. Same retrieval as M4r; built offline from M4r's stored
  responses by the production render function, so the two are paired on identical evidence.

All four arms Search one ingest built with sidecars; S and S′ run with the leg off, which queries
nothing in the sidecar namespace.

## What I predict

**Extraction (apparatus and cost):**

| Metric | Predicted |
|---|---|
| Images with a non-empty extraction | at least 0.95 |
| Extraction length, median words | 60 to 180 |
| Extraction spend for MemEye's 438 images | USD 0.10 to 0.60 |

**Stage 1, retrieval (MemEye, 8 scenarios, 371 questions, four option rotations):**

| Contrast | Predicted |
|---|---|
| M4r − S, any-clue Recall@10 by session | +0.01, band 0.00 to +0.04 (MemEye is near its ceiling: 0.9655 on Brand) |
| M4r − S, MRR | +0.02, band −0.01 to +0.05 |

**Stage 2, answers** (debiased exact match, MemEye MCQ prompt, DeepSeek V4.1 Flash reader pinned to
one provider, arms interleaved per question):

| Contrast | Set | Predicted |
|---|---|---|
| M4s − S | all 371 | **+0.03**, band 0.00 to +0.07 |
| M4r − S | all 371 | +0.01, band −0.02 to +0.03 |
| (M4s − S on X3∪X4) − (M4s − S on X1∪X2) | all 371 | positive: instance- and pixel-level questions gain more |
| S′ − S | all 371 | within ±0.03 |

The shown text should matter more than the retrieval leg: on MemEye Brand a clue round was already
in the top 10 on 0.9655 of questions (the other seven scenarios are measured by MM-1's Stage 1, not
yet reported when this was written), so what the reader lacks is more likely the content of the
image than its location.

## What would falsify this

- Extraction empty or failed on more than 5% of images (the sidecar is not reliably built).
- M4s − S at or below 0, or its paired CI lower bound below −0.03.
- The X-axis gap not positive: machine-read text is not helping where captions are known to lose
  detail.
- Any difference between S and the same Search on a tenant with no sidecars (the leg leaks).

## Decision rule

Recommend M4s (or M4r if M4s adds nothing over it) if M4s − S is at least +0.03 with a CI lower
bound above −0.02 and |S′ − S| is smaller. Before it could serve: re-run the extraction with
gpt-4o-mini on a sample of 100 MemEye images and check the answer gain holds on those questions,
and estimate the extraction cost for the official Multimodal data volume (about 55,000 Adds in the
first Full). Any change to C9 needs an explicit user decision and never during an AML job.

## How it will be measured

1. **Build** extraction, sidecar storage, the leg and the shown render behind flags; unit tests seen
   red against deliberate mutations: a text-only tenant is byte-identical with both flags on; a
   sidecar hit fuses into its parent; the shown part is appended once, labelled, only to
   image-bearing items; extraction output is capped.
2. **Stage 1** on VPS3: one C9 ingest of all eight MemEye scenarios with sidecars built at Add (the
   MM-1 harness, `scripts/aml_mm_scope_stage1.py`, with arms S and M4r); retrieval metrics as in
   MM-1 (`scripts/aml_mm_scope_report.py`).
3. **Stage 2**, only if extraction passes and Stage 1 shows no retrieval loss beyond −0.02 on
   Recall@10: answers for S, S′, M4r and M4s through `scripts/aml_mm_scope_stage2.py`'s reader
   path, paired bootstrap over questions, 10,000 resamples, seed 20260925.

**Spend.** Extraction under USD 1 (DeepSeek, 438 images). Ingest compile as in MM-1 (DeepSeek, small).
Answers about USD 8 to 12 for four arms on 371 questions with prompt caching. Cap USD 15 for MM-4,
the USD 40 balance floor, and never alongside another OpenRouter job (user instruction 2026-09-25).

## Apparatus checks, fixed now

1. Every image has exactly one sidecar row after ingest, with its parent's `primary_id`.
2. S's responses are byte-identical to a Search of the same questions on a table without the
   sidecar namespace, on a 20-question sample.
3. The leg ran (logged per Search) on every question in M4r and on none in S.
4. The shown render adds exactly one labelled text part per image-bearing item and changes nothing
   else (checked on every stored M4r row).
5. Reader valid-answer rate at least 98% in every arm.

## What I already know

- MemEye Brand v2 (29 questions): supplied text only 0.4655, preserved images 0.4310, dual 0.4741;
  any-clue Recall@10 0.9655 in all three (`docs/preregistrations/2026-09-20-aml-multimodal-memeye-brand-v2.md`).
  Handing the reader images at `low` detail did not beat text there, which is part of why the shown
  text is the arm expected to matter.
- MM-1 Stage 0: 21.6% of MemEye questions reach the image route; 4.4% across three AML sources.
- DeepSeek V4.1 Flash reads images: shown a Burger King logo, it named it; without the image it
  guessed Coca-Cola (MM-1 amendment 2).
- ATM-Bench's text-modality ceiling (memory `atm-answer-selection-status-2026-08-20`): some visual
  questions cannot be answered from text descriptions at all. MM-4 keeps the image; it adds text.

## Confounds I can name now

- **Machine-read text can be wrong.** A misread number or a hallucinated object reaches the reader as
  evidence. The label says where it came from; losses where M4s is wrong and S right are counted and
  reported, not only the net.
- **Extraction model is not the production one.** DeepSeek here, gpt-4o-mini under AML's rules; the
  confirmation step in the Decision rule exists for this.
- **MemEye only.** MobileMem and MemLens (the other AML Multimodal sources) have no local harness yet,
  and MemEye is itself one of AML's sources.
- **Arm S depends on MM-1.** If MM-1 changes the scope, MM-4 measures on top of that change, which is
  the intended comparison but means MM-4 cannot run before MM-1 is decided.

<!-- FROZEN PREREGISTRATION ENDS HERE. APPEND RESULTS BELOW. -->

## Results

No measurement had run when this record was committed.
