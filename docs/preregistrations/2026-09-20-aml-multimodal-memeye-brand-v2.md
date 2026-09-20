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

### Apparatus failure during MM1, 2026-09-20

Frozen application commit `2cc83bae424494b93558777d918274721216a8e3` produced a complete valid
MM0 artifact in immutable result directory `2cc83bae-live1`. MM0 mean debiased exact match was
`0.4655172414`, any-clue Recall at 10 was `0.9655172414`, and any-clue Recall at 100 was `1.0`.
MM0 remains descriptive prior evidence and will not be copied or scored in a retry.

MM1 accepted all 72 rounds, passed idempotent replay, and completed 12 questions and 48 rotations.
The forty-ninth Search returned HTTP 200, after which the corresponding fixed Answer request
exceeded its 180-second client read timeout. The runner emitted an incomplete artifact, deleted all
MM1 data, and verified an empty Search. The combined v1 plus v2 spend ledger reached `$5.376192`.

The previously repaired 600-second hosted-memory timeout was therefore not implicated: all 48
recorded MM1 Search calls completed in 2.095 to 3.185 seconds. A focused Answer-client construction
test was observed red at the frozen 180-second value. The apparatus repair raises only the Answer
HTTP timeout to 600 seconds; model, prompt, temperature, output limit, token budget, payload,
retryable HTTP statuses, scoring, gates, and metrics remain unchanged. Any retry must use a new
application commit, worktree, table set, immutable v2 result path, and a spend ledger seeded with
the incurred `$5.376192`.

### Apparatus failure during the second MM1 retry, 2026-09-20

Frozen application commit `d633de57875a0beb07deb51d5190d004e86f71f0` completed a new valid MM0
artifact in immutable result directory `d633de57-live1`. MM0 mean debiased exact match was
`0.4396551724`, any-clue Recall at 10 was `0.9655172414`, and any-clue Recall at 100 was `1.0`.
This MM0 result is descriptive prior evidence and will not be copied or scored in a retry.

MM1 accepted all 72 rounds, passed idempotent replay, and completed one question and four
rotations. Six Search requests returned HTTP 200 before the next Answer request exceeded the new
600-second read timeout. The runner emitted an incomplete artifact, deleted all MM1 data, and
verified an empty Search. The recorded combined spend ledger reached `$6.372818`. Because the
timed out request may have completed at the provider after the client stopped waiting, the next
ledger adds the registered maximum one-request reservation of `$0.117824` and begins at
`$6.490642`.

The failure is therefore in the Answer transport, not in Add, Search, media reconstruction, or
cleanup. The apparatus repair keeps the model, prompt, temperature, output limit, token budget,
payload, arm order, scoring, gates, and metrics fixed. It extends the existing three-attempt Answer
policy to ambiguous client read timeouts. Every attempt reuses the exact canonical payload. Each
timeout immediately reserves the maximum request cost in the authoritative ledger before another
attempt begins, so a provider response that arrives after the client timeout cannot become hidden
spend. The Answer client's internal retry count is one, preventing nested retries from exceeding
three total provider requests.

Focused node
`tests/test_aml_multimodal_memeye.py::test_answer_timeout_retries_identical_payload_and_reserves_cost`
was observed red on `d633de57`: it caught the first synthetic read timeout and failed at the
intended assertion that the bounded retry policy handled it. The same node passed after the repair.
Any retry must use a new application commit, worktree, table set, immutable v2 result path, and a
spend ledger seeded with `$6.490642`.

### Apparatus failure during MM2, 2026-09-20

Frozen application commit `2fdd3005da55210920a286768046cf61ff8aaf62` completed valid MM0 and MM1
artifacts in immutable result directory `2fdd3005-live1`. MM0 mean debiased exact match was
`0.4482758621`; MM1 was `0.4568965517`. Both arms had any-clue Recall at 10 of `0.9655172414`
and Recall at 100 of `1.0`. MM1 recovered one ambiguous Answer timeout using the registered
identical-payload retry and reserved `$0.117824`. The combined ledger reached `$13.378924`.

MM2 accepted six rounds. Its seventh round contained the pinned source image
`McDonalds_1.png`, whose resolution is 5,096 by 3,300 pixels, or 16,816,800 pixels. Voyage rejected
the image on all three attempts because its documented per-image limit is 16 million pixels. The
runner emitted an incomplete artifact before Search or Answer, deleted all MM2 data, and verified
an empty Search. No new Answer-provider spend was incurred.

The provider limit is documented at
`https://docs.voyageai.com/reference/multimodal-embeddings-api`. The repair retains the exact
original image bytes in RE-call's media record and returns those exact bytes through Search. Only
the transient derived input sent to `voyage-multimodal-3.5` is proportionally resized when it
exceeds 16 million pixels. The derived embedding profile advances from
`voyage-multimodal-3.5-v1` to `voyage-multimodal-3.5-v2`, and each vector record discloses its image
transform count. Model, text, source media, preserved evidence, retrieval fusion, Answer payload,
scoring, gates, and metrics remain unchanged.

Focused node
`tests/test_aml_multimodal.py::test_voyage_input_fits_provider_pixels_without_changing_preserved_media`
was observed red on `2fdd3005`: the preserved image remained exact, while the derived Voyage input
still exceeded the test pixel ceiling. The same node passed after the repair.

The repair requires a new exact commit and a fresh three-arm run. The authoritative ledger remains
`$13.378924`, leaving `$11.621076` under the frozen `$25` ceiling. The measured cost of MM0 plus
MM1 under `2fdd3005` was `$6.888282`; a comparable MM2 would make a complete fresh run exceed the
remaining allowance. No retry may start until the user explicitly authorizes a new ceiling in a
newly frozen amendment.
