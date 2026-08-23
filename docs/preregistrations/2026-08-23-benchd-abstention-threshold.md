# Pre-registration: what honoured abstention costs on Bench'd, at the default threshold

**Date:** 2026-08-23   **Status:** predicted, not yet measured

## The question

On the winning tuning configuration (A6: voyage-4 dense + lexical, rerank-2.5, DeepSeek v4 pro
synthesis with extended thinking off, top_k=10), how many points does honouring abstention at
RE-call's default gap threshold cost on the 60-question LongMemEval slice, relative to A6's
measured 75.0?

## What I predict

- Abstention fires on **6 to 18 of 60 questions** (10 to 30%): the default threshold is
  uncertified for voyage-4 on this corpus, and every fire is a forfeit because the locked judge
  scores "insufficient information" INCORRECT and every question is answerable.
- Score drops to **60 to 70** (a 5 to 15 point cost).
- Conclusion registered in advance: for the official run, threshold 0.0 with abstention
  suppressed is the correct configuration, and the honest artifact says so explicitly rather
  than hiding it. A corpus-calibrated threshold would converge on the same near-zero abstention
  because the calibration set would contain answerable questions only.

## What would falsify this

Zero abstentions at the default threshold (then suppression changes nothing and the knob is
moot), or an abstention-honouring score at or above A6's 75.0 (then abstention is somehow free
here and the framing above is wrong).

## How it will be measured

Same harness, same 60-question seed-42 slice, same A6 knobs except
`RECALL_BENCHD_ABSTAIN=honour` and `RECALL_BENCHD_THRESHOLD` set to RE-call's
`DEFAULT_GAP_THRESHOLD`. Metrics: count of empty recalls (abstentions), nuance overall, paired
per-question flips against the A6 manifest (run_2cd53886fb68).

## What I already know

A6 measured 75.0 with 0 fallbacks, 35.2 mean recall tokens, 46.9 tokens per correct. The A0
diagnosis showed the answerer already produces "insufficient" on genuinely missing evidence, so
suppression does not fabricate answers; it only stops RE-call from refusing before the answerer
sees the evidence.

## Confounds I can name now

- Judge variance on n=60 (~1 question either way).
- The default threshold was tuned for bge-class cosine geometry; voyage-4 cosines sit in a
  different range, so the abstention rate measures threshold miscalibration as much as
  abstention policy. That is the point of the measurement: it is the number that justifies
  calibrating rather than inheriting the default.
