# C8 atomic rescue with gpt-4o-mini reasoning and an admission gate

Status: frozen when committed. Nothing below the "Result" heading may be written before the
measurement, and nothing above it may be edited afterwards; corrections are appended.

## Question

The C8 atomizer reference (`2026-09-22-c8-atomizer-reference.md`) established that atomic rescue
activates on every query but gains too little (net exact@8 +1 and +2), because the rescued window
is gold for only 3.5 to 5.9% of control misses, and a wrong rescue can take fused rank 1 under
C8's RRF. Three changes attack that, and this record measures them:

1. **Admission gate** (deterministic): admit a rescue only if, under the same query vector, the
   best view scores at least as high as that vector's fifth-best whole window
   (`select_gated_atomic_rescue`, `recall/atomic_rescue.py`).
2. **gpt-4o-mini grounded atoms** (Add-time reasoning): up to six self-contained fact statements
   per window, each kept only if its verbatim quote grounds to an exact word range of the window
   (`ground_quote`, `recall/atomizer.py`). The statement is embedded; the window is the parent.
3. **gpt-4o-mini query decomposition** (Search-time reasoning): one to three search statements
   per query, each an extra probe vector, each gated against its own dense ranking.

C8 already calls gpt-4o-mini at Add (anchor compiler) and allows it at Search, so options 2 and 3
stay within the model boundary the AML Open-source track describes. This record authorizes no AML
use.

## Frozen apparatus

- Corpus, windows, probes, sentinel, control path and scoring: exactly as in the reference
  record (manifest `58055df1…`, probe file `c2d61909…`, 153 dev and 149 confirm probes, 34 tasks),
  via `scripts/c8_atomizer_reasoning.py`, which imports the reference harness unchanged.
- Model for both reasoning steps: `openai/gpt-4o-mini-2024-07-18` through OpenRouter, provider
  `openai`, fallbacks off, temperature 0, strict JSON schemas, prompts as committed in
  `ATOMS_PROMPT` and `DECOMPOSE_PROMPT`. Caps in code: USD 1.50 for atoms, USD 0.50 for
  decomposition.
- Atoms are generated once for all 1,220 windows and decompositions once for all 302 probes and
  34 tasks, before any evaluation; their file hashes are appended below before evaluation.
- Arms: `off`; `micro` (the reference's ungated arm, re-run as a parity check); `micro_gate`;
  `llm_gate`; `micro_gate_decomp`; `llm_gate_decomp`. Gate rank 5.
- Known threat, stated in advance: the probes were also written by gpt-4o-mini (different prompt,
  from random spans). Shared phrasing between probe questions and LLM atom statements could favour
  `llm_*` arms here in a way that human queries would not. This record cannot remove that; the
  memory-tenant confirmation the program requires is where it gets tested.

## Metrics

As in the reference record: primary is paired exact-gold net gain at rank 8 against `off` on
**dev**; plus exact@1/5/10, MRR@10, admitted rescues (`candidate_available`), exact-gold rescues,
selector latency, and the task sentinel. New: atom grounding rate and window coverage; the share
of admitted rescues that are exact gold; decomposition call latency.

## Predictions

My recorded bias is to over-predict effect sizes two to four times; these are set accordingly.

1. **Parity.** `micro` reproduces the reference dev result exactly (net +2, 5 gains, 3 losses),
   since vectors are cached and the path is deterministic.
2. **Grounding.** 60% to 85% of proposed atoms ground; 85% to 97% of windows keep at least one.
3. **Gate selectivity.** `micro_gate` admits a rescue on 20% to 60% of dev probes, and the share of
   admitted rescues that are exact gold is 8% to 20% (ungated `micro`: 9 of 153, 5.9%).
4. **`micro_gate` primary.** Net exact@8 +1 to +4, at most 1 loss.
5. **`llm_gate` primary.** Net exact@8 +2 to +6, at most 2 losses.
6. **Decomposition.** Each `_decomp` arm is 0 to +3 above its non-decomposed counterpart, never
   below it by more than 1.
7. **Sentinel.** Gated arms: 0 source losses at rank 8 and at most 1 exact loss at rank 1 on the
   34 tasks (ungated arms lost 2 at rank 1).
8. **Cost.** Atoms USD 0.20 to 0.45 in total; decomposition under USD 0.05 in total with call
   latency p50 between 0.8 and 2.0 seconds.

## Decision rules

- An arm is **eligible** if, on dev, net exact@8 gain is at least +3, losses at most 2, and the
  sentinel has 0 source losses at rank 8.
- The eligible arm with the largest net dev gain is run **once** on the confirm split, which no
  arm has touched. Ties go to the arm with fewer model calls (deterministic, then Add-time only,
  then Search-time). It **passes** if confirm net exact@8 gain is at least +2, losses at most 2,
  and the sentinel has 0 source losses at rank 8.
- If none is eligible, the outcome is recorded as such and no arm is retuned against this dev split.
- A pass names the C8 atomizer and selector to take to the live C8 service and then to fresh
  memory-tenant misses. It does not authorize production enablement or AML use.

## Result

Not yet run.
