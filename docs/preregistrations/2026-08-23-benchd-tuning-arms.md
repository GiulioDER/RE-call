# Pre-registration: Bench'd tuning arms for RE-call (LongMemEval slice)

**Date:** 2026-08-23   **Status:** predicted, not yet measured

## The question

Which RE-call configuration maximises the Bench'd LongMemEval nuance score, and by how much do
the three candidate additions (Voyage rerank-2.5, SPLADE sparse leg, DeepSeek v4 pro synthesis)
each move it over the dense-plus-lexical baseline?

## What I predict

On a 60-question stratified slice (`--max-items 60`, the harness's own seed-42 sampler), judged
by the official pipeline (gpt-4o-mini answerer and judge, temperature 0):

- **A0 baseline** (voyage-4 dense + lexical fusion, top_k=10, session granularity, no rerank,
  no synth, abstention suppressed): nuance overall **60 ± 5** (their leaderboard top is 59.0
  with weaker retrieval; the oracle variant makes retrieval easy).
- **A1 = A0 + Voyage rerank-2.5**: **+2 to +5 points** over A0.
- **A2 = A1 with SPLADE replacing the lexical leg** (prithivida/Splade_PP_en_v1): **0 to +3
  points** over A1. Conversational paraphrase is where learned sparse is weakest, so I predict
  the smallest effect here; this is the "is the SPLADE arm useful" question, and a null is a
  usable answer.
- **A3 = best retrieval arm + DeepSeek v4 pro synthesis** (`deepseek/deepseek-v4-pro-0813`,
  distilling top_k=10 chunks into a digest): **+3 to +8 points** over the best retrieval arm,
  and mean recall tokens drop from several hundred to **under 100**, which alone moves the BMI
  efficiency term by tens of points.
- Best arm overall lands at **66 to 72** nuance.

Predictions are set at a quarter to a half of plausible ceilings, per the standing correction
that I over-predict effect magnitudes (eleven of twelve past predictions high by 2 to 4 times).

## What would falsify this

Per arm: a point estimate at or below zero on the paired per-question comparison against its
baseline arm, or a positive estimate whose 95% CI includes zero AND is under +2 points. For A3
specifically, a synthesis arm that wins accuracy but pushes mean recall tokens above the raw
concatenation would falsify the efficiency half of the claim.

## How it will be measured

Bench'd harness (github.com/benchdai/harness, cloned 2026-08-23), custom adapter
`benchmarks/benchd/recall_adapter.py`, run on VPS2 (keys in `/opt/sentiment_agent/.env`, local
model inference under the embedding rules):

```bash
benchd run -a re-call -b longmemeval-v1 -n 60 --judge --key ./keys/private.key
```

with the arm's env knobs recorded per run. Metric: `scores.nuance.overall` from the signed
manifest (n=60, rate over judged questions), plus per-dimension nuance, mean recall tokens
(`efficiency.mean_recall_tokens`), and abstention count (predicted 0 with threshold 0.0).
Paired per-question deltas computed from the manifests' shared question ids. Token and cost
accounting via `benchmarks/benchd/count_tokens.py` (manifest recount plus OpenRouter usage
snapshots before and after).

Apparatus verification before any scored arm: `smoke-memory-v0` run end to end with the hashing
embedder ($0) checking ingest, retrieval, judging, manifest signing, and the token counter on a
case whose plumbing failures are visible.

## What I already know

- Their leaderboard: LlamaIndex and LangChain 59.0, LLM baseline 57.6, Mem0 OSS 32.4 on the
  full 500 (signed manifests in their repo). Nothing about RE-call on this harness has ever
  been measured; no prior benchd artifacts exist on VPS2 (searched 2026-08-23).
- The judge scores "insufficient information" as INCORRECT, so abstention is a forfeit; the
  suppress default and threshold 0.0 follow from that, and the threshold question for the
  official run is registered separately once the tuning arm is chosen.
- The ATM work found rerankers real but modest on retrieval-limited tasks, and the mem0-harness
  work found RE-call's retrieval competitive at matched budgets. Memory:
  `benchd-official-rules-2026-08-23`, `atm-answer-selection-status-2026-08-20`.

## Confounds I can name now

- **Tuning on a subset of the test set.** LongMemEval has no dev split; the 60-question slice
  is inside the official 500. Config chosen here is tuned on data it will be reported on. Named
  rather than solved: every system on this leaderboard has the same exposure, and the official
  number will come from the full 500, 88% of which the tuning never saw.
- **Judge variance.** Temperature 0 via OpenRouter is not bit-reproducible; their methodology
  claims under 0.3% variation on repeats. Differences under ~2 points on n=60 (about 1
  question) are noise regardless.
- **Synthesis leakage.** DeepSeek could answer from parametric knowledge rather than the
  excerpts; the prompt forbids it, but per-item enforcement is not verifiable. LongMemEval
  facts are synthetic personal history, which limits what parametric knowledge could supply,
  and the falsification criterion compares against the same retrieval, so leakage would show as
  an implausible jump on questions whose retrieval missed.
- **Ingest cache.** A cache bug would silently reuse a stale index across items; guarded by the
  hash-clearing fix in e4808a1b and by the smoke run, whose items each ingest distinct turns.
- **Provider routing.** OpenRouter may route deepseek-v4-pro to different backends between
  arms; single-provider pinning is available if A3's variance looks anomalous.
