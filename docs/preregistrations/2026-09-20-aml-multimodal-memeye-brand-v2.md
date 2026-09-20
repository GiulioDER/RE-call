# AML multimodal MemEye Brand completion v2

Date: 2026-09-20

## Purpose

This is a newly frozen completion experiment after the v1 apparatus established that original
image reconstruction is substantially more expensive and occasionally slower than the initial
budget assumptions. It does not edit, reinterpret, resume, or score the failed v1 MM1 artifact.
It remains a local directional experiment over public MemEye data, not an AML hosted evaluation or
leaderboard score.

The base protocol is the frozen registered section of
`docs/preregistrations/2026-09-20-aml-multimodal-memeye-brand-pilot.md` at RE-call commit
`052ad3a60595d9ac0b5b9fb69ef26ced15275da2`. Every source, dataset member, arm, ingestion rule,
Search request, Answer model and prompt, retrieval metric, prediction, admission threshold, selector
verdict, privacy rule, and host safety rule from that section remains frozen except for the explicit
amendments below.

## Frozen amendments

1. The provider cost ceiling is USD 25 across the v1 attempts and this v2 run together. The v2
   spend ledger must begin at the already incurred v1 total of exactly USD `2.10026`. The runner
   stops before the next Answer request at USD 25 and records an incomplete arm if one completed
   request crosses the ceiling. Prior spend is not attributed to any v2 arm score.
2. The hosted-memory HTTP client uses a 600-second per-request timeout so a valid response within
   the registered 30 MiB decoded-media allowance can arrive. The Answer provider client remains
   explicitly fixed at 180 seconds and three retryable-status attempts.
3. All three arms are rerun from the beginning under one new exact application commit. The valid v1
   MM0 result is descriptive prior evidence only and is not copied, selected, or scored in v2.
4. Every arm receives a fresh table, user ID, and cleanup verification. V2 uses the fresh immutable
   result namespace `results/aml-multimodal-memeye-brand-v2/<commit>-live1`. No v1 table or result
   file is reused.
5. The four-hour ceiling applies independently to each arm. The controller remains sequential, so
   at most one experiment worker can Add, Search, reconstruct media, or call the Answer proxy at a
   time.
6. The mechanical selector rejects a negative or missing arm cost and rejects a sum of valid v2 arm
   costs above USD 25. Runtime enforcement uses the carried-forward ledger, so the stricter combined
   v1 plus v2 ceiling is enforced even though the selector reports only v2 arm costs.

The user explicitly approved the new ceiling after adding provider credit. No official AML private
data or hosted challenge may be run under this authorization.

## Frozen gate

The v1 scientific gate is unchanged:

1. MM1 minus MM0 mean debiased exact match is at least `0.05`.
2. MM2 minus MM1 any-clue Recall at 10 is at least `0.05`.
3. MM2 minus MM1 mean debiased exact match is at least `0.02`.
4. MM2 any-clue Recall at 100 is not lower than MM1.
5. MM2 has no contract, isolation, response-budget, cleanup, identity, or cost regression.

The possible verdicts remain `ADMIT_FULL_MEMEYE`, `RETRIEVAL_ONLY`, `PRESERVATION_ONLY`,
`NO_GAIN`, and `INVALID`. Only `ADMIT_FULL_MEMEYE` authorizes a later, separately preregistered run
over all eight public MemEye MCQ scenarios.

## Frozen failure handling

Every partial artifact is preserved. An infrastructure or apparatus defect may be repaired only
after a focused intended red then green proof and an amendment appended below this marker. A repair
requires a new application commit, table set, worktree, and immutable result directory. Scientific
arms, prompts, labels, metrics, predictions, thresholds, and selection rules may not change after
this commit.

<!-- FROZEN PREREGISTRATION ENDS HERE. APPEND RESULTS BELOW. -->

## Results

No v2 measurement had run when this record was committed.
