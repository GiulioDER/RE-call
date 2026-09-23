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

## Result, fresh dev split (`dev3`), measured 2026-09-23 on VPS3

Appended after the measurement; nothing above has been edited. Harness commit `e8a856c5`, probe
file `493d15d2…`, atoms `840bf815…`. New vectors: 180 queries.

| dev3 (181 probes) | exact@1 | exact@5 | exact@8 | exact@10 | MRR@10 | gains/losses @8 | **net @8** | admitted | rescued = gold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `off` | 27 | 74 | 97 | 101 | 0.2640 | | | | |
| `micro` | 35 | 80 | 102 | 106 | 0.3017 | 5 / 0 | **+5** | 181 | 23 |
| `llm` | 38 | 83 | 102 | 105 | 0.3104 | 5 / 0 | **+5** | 181 | 20 |
| `micro_vgate` | 35 | 80 | 102 | 106 | 0.3017 | 5 / 0 | **+5** | 180 | 23 |
| `llm_vgate` | 38 | 83 | 102 | 105 | 0.3104 | 5 / 0 | **+5** | 181 | 20 |

Task sentinel (34): 0 source losses at rank 8 in every arm (the `micro` arms gain 1). Exact losses
at rank 1: `micro` and `micro_vgate` 2, `llm` and `llm_vgate` 0. No fallback anywhere.

Predictions, scored:

1. Writer yield 140 to 190: **confirmed** (181).
2. `off` exact@8 35% to 60%: **confirmed** (53.6%).
3. Gated arms admit 25% to 70%; `llm_vgate` 12% to 30% of admissions exact gold: **falsified**.
   The view gate admitted 180 and 181 of 181, and `llm_vgate`'s share is 11.0%. The reason is
   structural and should have been predicted: the maximum over about 11,000 outside views almost
   always exceeds the maximum over the few views of one protected parent.
4. Ungated `micro` net +0 to +3: **falsified high** (+5). `llm` net +1 to +5: **confirmed** (+5).
   The `micro` miss is my first recorded under-prediction in this program.
5. `micro_vgate` net +1 to +4 with at most 2 losses: net **falsified high** (+5), losses confirmed.
6. `llm_vgate` net +2 to +6 with at most 2 losses: **confirmed** (+5, 0 losses).
7. Gated sentinel, 0 source losses at 8 and at most 1 exact loss at 1: `llm_vgate` **confirmed**;
   `micro_vgate` **falsified** on rank 1 (2 losses).
8. Atom advantage survives a non-OpenAI writer (gold-rescue share ratio at least 1.3): **falsified**.
   `llm` 20 against `micro` 23, ratio 0.87. Round 2's precision advantage for gpt-4o-mini atoms
   (16 against 9, on gpt-4o-mini questions) did not reproduce on Llama-written questions, which is
   what the stated writer threat predicts.

**Decision.** All four rescue arms are **eligible** (net +5, 0 losses, 0 sentinel source losses
at rank 8), tied on net gain. The rule breaks ties by fewer model calls and names `micro_vgate`
before `llm_vgate`, but does not separate `micro` from `micro_vgate`, which both make none.
Resolved here, **before the confirm split is touched**, in favour of **`micro`** (ungated):
the view gate is inert on this set (180 of 181 admitted), `micro` needs no new selector code, and
it is the arm already validated live (`2026-09-22-c8-atomizer-reference.md`, live receipt). The
confirm run uses `off` and `micro` only. Because `micro` is deterministic, the confirm probes'
gpt-4o-mini authorship cannot favour it through shared phrasing with atoms.

## Result, confirm split, measured once 2026-09-23 on VPS3

Appended after the measurement. Arms `off` and `micro` only, as resolved above before this run.
Probe file `c2d61909…`, the 149 confirm probes that no arm of any round had touched. Harness
commit `fb5e048b`. New vectors: 143 queries.

| confirm (149 probes) | exact@1 | exact@5 | exact@8 | exact@10 | MRR@10 | source@8 |
|---|---:|---:|---:|---:|---:|---:|
| `off` | 21 | 61 | 72 | 77 | 0.2538 | 93 |
| `micro` | 25 | 70 | 76 | 80 | 0.2860 | 97 |

Paired against `off` at exact@8: **6 gains, 2 losses, net +4**. At exact@1: 5 gains, 1 loss.
Admitted and candidate available on 149 of 149, no fallback; 14 rescues were exact gold.
Task sentinel (34): source@8 33 to 34, **0 source losses at rank 8**; exact@1 27 to 25, 2 losses.

**Decision: PASS.** Net +4 is at least +2, 2 losses is at most 2, and the sentinel has 0 source
losses at rank 8. The ungated `micro` window atomizer is the confirmed C8 atomizer reference.

What the pass does and does not establish:

- It establishes a modest, replicated retrieval gain for C8 atomic rescue with the `micro`
  atomizer on CAMBench: net +2 on the original dev split, +5 on fresh dev, +4 on confirm, with the
  live service reproducing the offline replay (live receipt in the reference record).
- It does **not** clear the rank-1 cost on task prompts: both `micro` runs lose 2 task prompts at
  exact@1 (source ranks unaffected). The rule did not gate on it; a production decision should.
- It does **not** establish anything about the memory tenant, production, AML task success or the
  gpt-4o-mini atoms, whose round-2 advantage failed to reproduce on non-OpenAI questions.
- The admission gates of rounds 2 and 3 are both inert as built and are not part of the reference.

Program spend for rounds 1 to 3 is recorded in the program memo.
