# C8 atomizer round 3: fresh dev questions and a view-level admission gate

Status: frozen when committed. Nothing below the "Result" heading may be written before the
measurement, and nothing above it may be edited afterwards; corrections are appended.

## Question

Round 2 (`2026-09-22-c8-atomizer-reasoning.md`) found no eligible arm. Its best lead was grounded
gpt-4o-mini atoms (net exact@8 +3 with 3 losses, 16 of 150 admitted rescues exact gold, twice the
`micro` precision, no rank-1 losses on the task sentinel). Its gate did not gate: comparing a short
view with a 160-word window admitted 98.7% of probes. Round 3 asks, on **fresh** dev questions:
does a gate that compares views with views turn grounded atoms (or micro-windows) into an eligible
arm?

## Frozen apparatus

- Corpus, windows, control path, scoring, sentinel: exactly as in the reference record, via
  `scripts/c8_atomizer_round3.py`, which imports the reference and reasoning harnesses unchanged.
- **Gate** `select_view_gated_atomic_rescue` (`recall/atomic_rescue.py`): admit the best view
  outside the served dense top five only if its score is at least the **weakest protected
  parent's best view** score under the same query vector; refuse when no protected parent has a
  view.
- **Atoms**: the round-2 file, unchanged (SHA-256 `840bf815…`, 5,305 grounded gpt-4o-mini facts).
- **Fresh dev questions** (`dev3`): seed 20260923, spans of 12 to 30 words, two per session, drawn
  from **dev-half sessions only** and disjoint from every span of the round-1 probe file, so the
  **149 confirm probes remain untouched** and are the only confirmation set. Writer
  `meta-llama/llama-3.3-70b-instruct` through OpenRouter (`require_parameters`, temperature 0,
  strict schema), with the reference record's prompt and rejection rules. A non-OpenAI writer
  addresses the round-2 threat that one model wrote both the atoms and the questions. The confirm
  probes were written by gpt-4o-mini, so the threat still applies to confirmation and is stated
  rather than removed. Probe file hash appended below before evaluation. Cap USD 0.50 in code.
- **Arms**: `off`; `micro` (reference, ungated); `llm` (grounded atoms, ungated); `micro_vgate`;
  `llm_vgate`.

## Metrics

Primary: paired exact-gold net gain at rank 8 against `off` on `dev3`. Secondary: exact@1/5/10,
MRR@10, admitted rescues, admitted rescues that are exact gold, the task sentinel.

## Predictions

Set at a quarter to a half of first instinct, per my recorded over-prediction bias.

1. **Writer yield.** 140 to 190 of the drawn spans become kept questions.
2. **Headroom.** `off` exact@8 on `dev3` between 35% and 60%.
3. **Selectivity.** Each gated arm admits a rescue on 25% to 70% of `dev3` probes. For
   `llm_vgate`, 12% to 30% of admitted rescues are exact gold.
4. **Ungated arms.** `micro` net +0 to +3; `llm` net +1 to +5.
5. **`micro_vgate`.** Net +1 to +4, at most 2 losses.
6. **`llm_vgate`.** Net +2 to +6, at most 2 losses. It is the arm I expect to be eligible, and I
   put that at a little under even odds.
7. **Sentinel.** Gated arms: 0 source losses at rank 8 and at most 1 exact loss at rank 1.
8. **Writer threat.** The atom advantage survives a non-OpenAI writer: `llm`'s exact-gold rescue
   share is at least 1.3 times `micro`'s.

## Decision rules

- **Eligible** on `dev3`: net exact@8 at least +3, at most 2 losses, 0 sentinel source losses at
  rank 8.
- The eligible arm with the largest net gain runs **once** on the 149 confirm probes against `off`.
  Ties go to fewer model calls (`micro_vgate` before `llm_vgate`). It **passes** with confirm net
  exact@8 at least +2, at most 2 losses, and 0 sentinel source losses at rank 8.
- If none is eligible, record it; nothing is retuned on `dev3`, and the confirm probes stay
  untouched.
- A pass names the atomizer and gate to take to the live C8 instance on VPS3 and then to fresh
  memory-tenant misses. It authorizes neither production enablement nor AML use. VPS2 is not used.

## Result

Not yet run.

## Apparatus appendix, appended 2026-09-23 before any evaluation

Apparatus only; no arm of this record had been evaluated when it was written.

- Fresh dev probe file SHA-256 `493d15d282a346712716839a43090cfafa69944eaff0cd16bcce78e124a1b927`
  (private, VPS3): 204 spans drawn, **181 kept**, 19 rejected for copying the span, 4 not
  answerable. USD 0.020862.
- Served by `meta-llama/llama-3.3-70b-instruct` through **six providers** (AkashML, CoreWeave,
  DeepInfra, Parasail, SambaNova, Together), because routing allowed fallbacks among providers
  that honour the strict schema. The file is frozen, so this affects only a regeneration, not this
  record.
- Prediction 1 (140 to 190 kept) is already visible and **confirmed** (181).
