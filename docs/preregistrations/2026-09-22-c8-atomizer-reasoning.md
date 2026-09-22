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

## Apparatus appendix, appended 2026-09-22 before any evaluation

Apparatus only, written below the Result heading because nothing above it may be edited. No arm
of this record had been evaluated and no vector for its inputs embedded when it was written.

- Atoms file SHA-256 `840bf815b84fca13a5bfe863f7d2335ef2284ab90f221a563c73a6c8ec334839` (private,
  VPS3): 1,220 windows, 0 malformed, **6,236 facts proposed, 5,305 grounded (85.07%)**, 923
  ungrounded, 8 over the statement length; **1,213 of 1,220 windows** keep at least one fact.
  USD 0.23398, every call served by `openai/gpt-4o-mini-2024-07-18` via OpenAI.
- Decompositions file SHA-256 `597d04c863036ebbd17203f95161d76365bb45198c1c55b391ccad9b056f54a2`:
  336 queries (302 probes, 34 tasks), 0 empty, mean 2.982 sub-queries, call latency p50 883 ms and
  p95 1,431 ms. USD 0.012638, same snapshot and provider.
- Already visible against prediction 2, stated now rather than in the result: grounding 85.07% is
  just above the predicted 60 to 85%, and window coverage 99.4% is above the predicted 85 to 97%.
  Both are falsified on the high side. Prediction 8's atom cost (USD 0.20 to 0.45) and
  decomposition cost (under USD 0.05) are within range; its latency p50 (0.88 s) is within the
  predicted 0.8 to 2.0 s.

## Result, dev split, measured 2026-09-22 on VPS3

Appended after the measurement; nothing above has been edited. Harness commit `589d6ad2`, inputs as
in the appendix. New vectors this run: 4,938 atom statements and 542 sub-queries. The confirm
split was **not** run, per the decision rule.

| dev (153 probes) | exact@1 | exact@5 | exact@8 | exact@10 | MRR@10 | gains/losses @8 | **net @8** | admitted | rescued = gold | selector p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `off` | 21 | 54 | 68 | 75 | 0.2264 | | | | | |
| `micro` | 25 | 59 | 70 | 77 | 0.2545 | 5 / 3 | +2 | 153 | 9 | 2.1 |
| `micro_gate` | 24 | 58 | 70 | 77 | 0.2488 | 5 / 3 | +2 | 151 | 8 | 1.8 |
| `llm_gate` | 24 | 61 | 71 | 78 | 0.2603 | 6 / 3 | **+3** | 150 | 16 | 0.9 |
| `micro_gate_decomp` | 22 | 56 | 68 | 76 | 0.2419 | 4 / 4 | 0 | 153 | 8 | 6.3 |
| `llm_gate_decomp` | 21 | 55 | 70 | 78 | 0.2380 | 5 / 3 | +2 | 153 | 8 | 5.6 |

Task sentinel (34): every arm has 0 source losses at rank 8. Exact losses at rank 1:
`micro` 2, `micro_gate` 2, `llm_gate` **0**, `micro_gate_decomp` 1, `llm_gate_decomp` 2.
No arm fell back on any query.

Predictions, scored:

1. Parity: **confirmed**. `micro` reproduces the reference dev result exactly (5 gains, 3 losses).
2. Grounding: **falsified high** (85.07% and 99.4%), as already stated in the appendix.
3. Gate selectivity 20 to 60% admitted, 8 to 20% of admitted exact gold: **falsified**. 151 of 153
   admitted (98.7%); 8 of 151 exact gold (5.3%).
4. `micro_gate` net +1 to +4 with at most 1 loss: net **confirmed** (+2), losses **falsified** (3).
5. `llm_gate` net +2 to +6 with at most 2 losses: net **confirmed** (+3), losses **falsified** (3).
6. Decomposition 0 to +3 above its counterpart, never more than 1 below: **falsified**.
   `llm_gate_decomp` is 1 below `llm_gate` (inside the tolerance) but `micro_gate_decomp` is 2
   below `micro_gate`.
7. Sentinel, gated arms 0 source losses at rank 8 and at most 1 exact loss at rank 1: source part
   **confirmed** for all four; rank-1 part confirmed for `llm_gate` (0) and `micro_gate_decomp` (1),
   **falsified** for `micro_gate` and `llm_gate_decomp` (2 each).
8. Cost: **confirmed**, as stated in the appendix.

**Decision: no arm is eligible.** `llm_gate` reaches the +3 net bar but has 3 losses against a
limit of 2. Per the rule, no arm is retuned against this dev split, and the confirm split remains
untouched.

What the rows show, for the next record rather than as a claim of this one:

- **The gate does not gate.** A short view's cosine is systematically higher than a 160-word
  window's under the same query vector, so "beats the fifth window" is true for about 99% of
  probes. A usable gate must compare like lengths: a view against other views, or against its own
  parent window's score.
- **Grounded gpt-4o-mini atoms roughly double rescue precision** (16 of 150 admitted rescues
  exact gold, against 9 of 153 for `micro`) and are the only arm with zero rank-1 losses on the task
  sentinel. This is the direction worth a fresh record, with the threat stated above (the probes
  were also written by gpt-4o-mini) still unaddressed.
- **Decomposition makes rescue worse and costs a model call per search.** More probe vectors
  mean more chances for a wrong view to win the single slot.

Spend for this record: USD 0.247 OpenRouter (atoms 0.234, decomposition 0.013) plus Voyage Code4
for 4,938 statements and 542 sub-queries.
