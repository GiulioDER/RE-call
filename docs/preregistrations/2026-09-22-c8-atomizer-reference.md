# C8 atomizer reference on CAMBench

Status: frozen when committed. Nothing below the "Result" heading may be written before the
measurement, and nothing above it may be edited afterwards; corrections are appended.

## Question

Does atomic rescue, which C8 (`C8_routed_specialists_grounded_graph`) carries but has never
activated, recover exact gold windows on a real C8-shaped corpus once the atomizer can segment
that corpus at all? Which of two deterministic window-aware atomizers is the better reference?

## Why the current atomizer cannot answer

Measured 2026-09-22, before this record, with no embedding: C8 stores each CAMBench session as
one-line, content-only 160-word windows with a 120-word stride. The memo atomizer
(`build_source_views` in `scripts/audit_atomic_fact_auxiliary_view.py`) needs blank-line
paragraphs and one parent chunk per paragraph. On the frozen corpus it produces **1 view for
1,220 windows**. The C8 qualification corpus had 3 chunks, below the selector's floor of five
distinct dense parents, so the stage has only ever failed closed.

## Frozen apparatus

- Corpus: CAMBench coding manifest SHA-256
  `58055df1828b2c1e51bc3c7f9f82e916145c67aa58332f22ce1b86b2d849b814`, 196 sessions, 1,220
  windows, 34 task prompts, loaded and verified by `load_frozen_corpus` in
  `scripts/aml_c7_qualification.py`.
- Harness: `scripts/c8_atomizer_reference.py` at the commit that adds this record. It replays C8's
  Code4 path offline: exact dense top 100 (profile `voyage-code-4-v1`), the production
  `insert_atomic_rescue_dense` on an artifact written and loaded by the production writer and
  loader, canonical BM25, and unweighted RRF (constant 60) with stable window ties. Graph sidecar
  promotion is not replayed; it preserves top-100 membership and the protected top 8 by contract.
- Atomizer: `recall/atomizer.py`. Arms:

  | arm | settings | views | parents covered |
  |---|---|---:|---:|
  | `off` | control, no rescue | 0 | 0 |
  | `sentence` | sentence split, 6 to 40 words | 6,579 | 1,216 of 1,220 |
  | `micro` | 24-word windows, stride 12 | 11,749 | 1,220 of 1,220 |

  The view counts were computed before this record without embedding anything.
- Probes: 390 seeded spans (seed 20260922, 12 to 30 words, 2 per session), drawn with no
  reference to either atomizer. One question per span from `openai/gpt-4o-mini-2024-07-18` via
  OpenRouter, provider `openai`, fallbacks off, temperature 0, strict JSON schema. Rejection rules
  are frozen in `probe_rejection`: not answerable, fewer than 5 or more than 45 words, or 5 or
  more consecutive span tokens copied. Gold is every window that fully contains the span
  (55 spans have two). Split by session hash: 204 dev spans from 103 sessions, the rest confirm.
  Probe texts and per-query rows are private and stay outside the repository. The probe file's
  SHA-256 is appended below once generated, before any evaluation.
- Sentinel: the 34 task prompts, gold = every window of the task's own sessions.
- Execution: VPS3 only, because VPS2 is serving the official AML Full run. Budget for this record
  USD 1.00 (probe generation capped at USD 1.00 in code); the program cap is USD 10.

## Metrics

Primary, on the **dev** probes: paired exact-gold gains and losses at rank 8 against `off`
(C8 protects the top 8). Secondary: exact@1/5/6/10, exact MRR@10, source@k, the share of
rescues that are exact gold, rescues from outside the dense top 100, selector latency, and the
activation counters `attempted`, `active`, `candidate_available`, `fallback`.

## Predictions

Written before any probe exists. My recorded bias is to over-predict effect sizes by two to four
times, so these are deliberately set at a quarter to a half of first instinct.

1. **Activation.** For both arms, on every dev probe and every task prompt:
   `attempted = active = candidate_available = queries`, `fallback = 0`.
2. **Control headroom.** `off` exact@8 on dev probes between 55% and 75%.
3. **Sentence arm, primary.** Net exact@8 gain on dev probes of **+2 to +6** (centre +4, about
   2 percentage points), with **at most 2 losses**.
4. **Micro arm, primary.** Net exact@8 gain of **+1 to +5** (centre +3), at most 3 losses. I expect
   it no better than `sentence`: twice the views means more near-duplicate competition for the
   single slot.
5. **Mechanism.** Among dev probes that `off` misses at rank 8, the rescued parent is exact gold
   in **8% to 20%** of them for `sentence`.
6. **Sentinel.** On the 34 task prompts, **0** source losses at rank 8 for either arm, and 0 gains
   (the control is expected at or near ceiling).
7. **Cost.** Selector p95 below 10 ms per query for `sentence` and below 20 ms for `micro` on VPS3.

## Decision rules

- An arm is **eligible** if, on dev, net exact@8 gain is at least +3, losses at most 2, and the
  task sentinel has 0 source losses at rank 8.
- The single eligible arm with the larger net dev gain (tie: `sentence`, fewer views) is run
  **once** on the confirm split. It **passes** if confirm net exact@8 gain is at least +2 with at
  most 2 losses and 0 sentinel losses. Confirm probes are not inspected before that run.
- If no arm is eligible, the reference is recorded as "activates, no measured gain", and the next
  step is a changed selector or atomizer under a new record, never a retuned version of these
  arms against the same dev split.
- Passing makes the arm the C8 atomizer reference and the input to a generation-bound artifact
  builder for the live C8 service. It does not authorize production enablement or any AML
  submission.

## Result

Not yet run.

## Apparatus appendix, appended 2026-09-22 before any evaluation

This section records apparatus only, not a result. It sits below the Result heading because the
record forbids editing anything above that heading after commit; no vector had been embedded and
no ranking computed when it was written.

- Probe file SHA-256 `c2d6190933f17e2a6b8250e14ee4a791dc672357abc79229f69df4f4374244e1`
  (private, on VPS3). 390 spans, **302 kept**: 57 rejected for copying the span, 30 not
  answerable, 1 for length.
- Split: **153 dev** probes from 97 sessions, **149 confirm** from 89; 21 dev and 21 confirm
  probes have two gold windows.
- Cost USD 0.036432, every call served by `openai/gpt-4o-mini-2024-07-18` via provider OpenAI.
- Prediction 3's centre of +4 is now about 2.6 percentage points of 153 dev probes rather than the
  "about 2 points" written above, because fewer probes survived than spans were drawn. The
  prediction's range in probes (+2 to +6) stands as written.

## Result, dev split, measured 2026-09-22 on VPS3

Appended after the measurement. Nothing above has been edited. Harness commit `d36c989c`, probe
file `c2d61909…`, 153 dev probes and 34 task prompts, Code4 cache of 1,220 windows plus both view
sets. The confirm split was **not** run, per the decision rule below.

| | `off` | `sentence` | `micro` |
|---|---:|---:|---:|
| probe exact@1 | 21 | 22 | 25 |
| probe exact@5 | 54 | 55 | 59 |
| probe exact@8 (primary) | 68 | 69 | 70 |
| probe exact@10 | 75 | 76 | 77 |
| probe exact MRR@10 | 0.2264 | 0.2382 | 0.2545 |
| gains / losses @8 vs `off` | | 2 / 1 | 5 / 3 |
| **net @8** | | **+1** | **+2** |
| rescued window is exact gold, of 85 control misses @8 | | 3 (3.5%) | 5 (5.9%) |
| rescued from outside dense top 100 | | 39 | 46 |
| attempted / active / candidate / fallback | | 153/153/153/0 | 153/153/153/0 |
| selector p50 / p95 ms | | 0.86 / 1.13 | 1.31 / 2.00 |
| task sentinel source@8 (off 33) | | 34 | 34 |
| task sentinel exact@1 (off 27) | | 25 | 25 |

Predictions, scored:

1. Activation: **confirmed.** Every probe and task query attempted, active, with a candidate and
   no fallback, in both arms. This is the activation receipt C8 never had.
2. Control headroom 55 to 75%: **falsified low**, 44.4%. The probes are harder than predicted.
3. `sentence` net +2 to +6, at most 2 losses: **falsified**, net +1 (2 gains, 1 loss).
4. `micro` net +1 to +5, at most 3 losses: **confirmed**, net +2 (5 gains, 3 losses). The
   expectation that it would be no better than `sentence` was wrong.
5. Mechanism 8 to 20% gold rescues among control misses (`sentence`): **falsified**, 3.5%.
6. Sentinel 0 losses and 0 gains at rank 8: losses **confirmed** (0), gains **falsified** (+1 in
   each arm).
7. Selector p95 below 10 and 20 ms: **confirmed**, 1.13 and 2.00 ms.

**Decision: no arm is eligible** (net dev gain below +3 for both). Recorded outcome: the C8 atomic
stage **activates, with no measured gain** on this reference. The confirm split stays untouched.

What the rows show, for the next record rather than as a claim of this one:

- Rescue precision is the binding constraint: a wrong rescue is chosen for 94 to 96% of misses.
- Gains are large jumps (fused 34 to 1, 57 to 2, 38 to 2, 27 to 1); losses are mostly one-place
  slips of the gold window behind a wrong rescue.
- Under C8's RRF with BM25, inserting at **dense** rank 6 does not protect the **fused** top
  five. A wrong rescue with a strong BM25 rank took fused rank 1 on two task prompts, both rescued
  from outside the dense top 100. The production statement "preserves the dense top five" is true
  of the dense list and must not be read as a statement about the final C8 ranking.

Spend for this record: USD 0.036 OpenRouter plus Voyage Code4 for 1,220 windows, 11,764 view
vectors and 182 queries (about 0.9M tokens).
