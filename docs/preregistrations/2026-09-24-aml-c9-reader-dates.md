# Pre-registration: does C9 lose AML Textual answers because the reader never sees a date?

**Date:** 2026-09-24   **Status:** predicted, not yet measured. Measurement waits for the user's
explicit go.

## The question

C9 stores and returns conversation windows as bare message content: `content_only_windows=True`
in the C9 `HostedVariant` (`recall_aml/variants.py`), renderer `message-content-only-v1`. No
timestamp and no role is in the text. A date can reach AML's Answer model only through the optional
`created_at` field of a Search item, and AML's API guide promises only that `content` is passed to
Answer.

With AML's own LoCoMo-Refined Answer and accuracy prompts over what C9 returns:

1. **How much accuracy does C9 lose when the reader sees `content` alone** instead of the
   `- [created_at] content` block the 2026-09-24 loss diagnosis rendered?
2. **Does the renderer C9 already carries, `timestamp-role-content-v1`, win that back** when the
   reader sees `content` alone?

## Why it is being asked

The official AML Textual Smoke on C9 (2026-09-24, 18:44 to 19:28 UTC, served `385c6074`) scored
**0 on B2** (causal chain, intermediate steps), **0 on D1** (value updates, current state), **0 on
F1** (summarization, long history) and **20 on C1** (dates, relative time). The C9 journal shows
134 Adds and 48 Searches from AML, every one HTTP 200, no failed request and no graph or atomic
fallback, so the zeros are answer quality and not transport.

Three facts make missing dates the leading explanation, and none of them measures it:

- C1, D1 and B2 all need chronology: a date, the latest of two values, the order of steps.
  Relevance-ordered, undated windows carry none of it. F1 fits less directly.
- Only CLBench, among AML's public pipelines at `1b8142bf`, renders `created_at`. Whether the
  LoCoMo, LongMemEval, BEAM and ScriptMem prompts see it is not disclosed.
- Only 474 of the 770 raw windows the Smoke stored (62%) carry an `event_time`, the source of
  `created_at`. The rest have no date anywhere.

And the one number that says C9 is fine on Textual, **75.3% on LoCoMo**
(`2026-09-24-aml-c9-locomo-loss-diagnosis.md`), was measured with `created_at` rendered into every
line. If the platform does not render it, that number overstates C9 exactly where the Smoke scored
zero.

The Smoke itself cannot answer this. It holds about 40 distinct questions over every leaf category,
so each category has roughly 1 to 3 questions, and AML asks participants not to mine evaluation
data. LoCoMo, where every message carries its session date, isolates the mechanism.

## Arms

Two collects on VPS3, three read views. The pairings are by question id.

| arm | collect | window renderer | what the reader sees |
| --- | --- | --- | --- |
| **A** | S, served C9 | `message-content-only-v1` | `- [created_at] content`, exactly as on 2026-09-24 |
| **B** | S, same items as A | `message-content-only-v1` | `- content` only: the platform view if `created_at` is never rendered |
| **C** | T, C9 with `content_only_windows=False` | `timestamp-role-content-v1` | `- content` only |

Collect T differs from S in that one flag and nothing else. The flag already exists in the
product; this experiment adds no retrieval code.

## What I predict

Accuracy is the AML accuracy prompt's CORRECT rate. LoCoMo category 2 is temporal (about 320 of 1,535).
Deltas are paired, in points, with a 10,000 resample 95% interval.

| quantity | prediction |
| --- | --- |
| A accuracy, all 1,535 | 72% to 78% (replicates 75.3%) |
| A accuracy, category 2 | 53% to 64% (was 59.1%) |
| B accuracy, category 2 | 5% to 30% |
| **B minus A, category 2** | **-30 to -55** |
| B minus A, category 4 (single hop) | -4 to +2 |
| B minus A, all | -6 to -13 |
| **C minus B, category 2** | **+20 to +50** |
| C minus A, all | -3 to +2 |
| C minus A, category 2 | -8 to +5 |
| collect T turn hit@10 minus collect S | -3.0 to +0.5 |
| mean Answer prompt characters, C over B | 1.05x to 1.30x |

Reasoning. LoCoMo temporal gold is almost always an absolute date or a date relative to one ("7 May
2023", "the week before 9 June 2023"), and the turns themselves rarely state one, so without a
timestamp the reader can only guess. That collapse is mechanical, which is why the B band is low
and wide despite my habit of over-predicting effects ([[i-over-predict-effect-magnitudes]]).
Single-hop answers rarely need a date, hence the small category 4 band. Timestamped windows give
the reader the same session date that `created_at` gave, but inside the text, so C should land near
A; it can lose a little because the timestamp and role prefix spend window words and add identical
date tokens to every window, which can dilute both BM25 and the embedding. That is the retrieval
band.

## Decision rule, fixed now

Applied in order, and the first that fires decides.

1. **Apparatus failure.** If any check below fails, report the numbers as unreliable and decide
   nothing.
2. **Hypothesis falsified.** If B minus A on category 2 is above -10 points, missing dates do not
   explain much on LoCoMo. Change nothing in C9 for this reason, and the next step is the Smoke's
   per-question results from the AML platform, if the user can see them.
3. **Fix supported.** If B minus A on category 2 is at most -10, AND C minus B on category 2 is at
   least +10 with its interval above 0, AND C minus A on all questions is at least -1.0 as a point
   estimate, AND collect T turn hit@10 is within 1.5 points of collect S: recommend to the user a
   C9 build with `content_only_windows=False` for the next Textual run. It is a recommendation
   only. It changes the official baseline, which needs the user's decision
   ([[official-textual-coding-config-graph-on]]), and one endpoint serves both tracks, so it also
   needs a Coding non-inferiority check under its own pre-registration before adoption.
4. **Dates matter, this renderer is not the fix.** If B minus A on category 2 is at most -10 and
   rule 3 does not fire, the next candidate is a leaner date header per window, under its own
   pre-registration.

Nothing here touches the official C9 instance on VPS2.

## What would falsify this

- Any quantity outside its band in the prediction table.
- In particular: B minus A on category 2 above -30 means I overrated how much the reader depends on
  the timestamp; C minus B below +20 means the renderer recovers less than the timestamp did.

## How it will be measured

- **Script:** `scripts/aml_locomo_loss_diagnosis.py`, extended in this commit with
  `collect --timestamped-windows` (the variant with only `content_only_windows` flipped, via
  `timestamped_windows`) and `answer --reader-view content` (`render_memories(..., dated=False)`).
  Everything else is the harness of the 2026-09-24 loss diagnosis, unchanged.
- **Code:** this branch, whose `recall_aml` is identical to the served `385c6074`
  (`git diff 385c6074 HEAD -- recall_aml` is empty).
- **Collect, on VPS3:** a fresh database of its own, the service built in process, variant
  `C9_routed_specialists_grounded_graph_atomic`, graph on, atomic stage active. One Add per LoCoMo
  session (272), then one Search per question at top_k 100 with the served router, items kept, then
  the users deleted.
  - S: `collect --run-id readerdatesS --preregistration docs/preregistrations/2026-09-24-aml-c9-reader-dates.md`
  - T: `collect --run-id readerdatesT --timestamped-windows --preregistration <same>`
  - S may reuse a copy of the embedding cache. T's window texts are new, so T embeds from scratch.
- **Dataset:** `locomo10.json`, SHA256 `79fa87e9…`, the same 1,535 questions in categories 1 to 4
  whose evidence resolves.
- **Answer:** A is `answer --reader-view dated` on S, B is `answer --reader-view content` on S, C
  is `answer --reader-view content` on T. The official prompts are imported at runtime from
  `github.com/AML-memory/agent-memory-leaderboard` at `1b8142bf`. `openai/gpt-4o-mini` through
  OpenRouter, temperature 0.
- **Judge:** the official accuracy prompt and parser, same model and settings, for each arm.
  Unparsed counts as WRONG and is reported.
- **Compare:** `compare` for B against A, C against B, and C against A, overall and by category.
- **Cost cap:** 25 USD of OpenRouter across both collects' Add-time compiles, three answer arms,
  three judge arms and the judge self-check. My estimate is about 17 USD. Voyage embedding for
  collect T is extra and small.
- **No classify stage.** Bucket shares are not part of this question.

## Apparatus checks

1. **Canary**, each collect: the verbatim text of one ingested turn as the query scores turn
   hit@10 = 1.
2. **Parity of S with 2026-09-24:** turn hit@10 within 1.0 point of 93.36 and hit@100 at least
   99.5%.
3. **Renderer guard:** `/version` reports `message-content-only-v1` for S and
   `timestamp-role-content-v1` for T, else the collect stops (enforced in the script).
4. **Views are what they claim:** B's and C's prompts contain no `- [` timestamp prefix, and A's
   does. Checked on 20 prompts per arm drawn by seed 0.
5. **Judge known answer:** at least 95% CORRECT when a question's own gold is judged.
6. **Budget:** the longest Answer prompt of each arm fits AML's 117,760 token input window, by
   character count.
7. The two new behaviours are unit tested and proved red by mutation
   (`tests/test_aml_locomo_loss_diagnosis.py`, docstring names each mutation):
   `test_the_content_view_never_shows_a_timestamp` and
   `test_timestamped_windows_changes_only_the_renderer_flag`.

## Confounds I can name now

- **The platform's rendering is unknown.** A is the best case (it renders `created_at`), B the
  worst (it does not). This run measures how much that unknown is worth, not which one is true.
- **LoCoMo is not the Smoke.** Every LoCoMo message carries a session date. In the Smoke, 38% of
  raw windows had none, and neither A nor C can help those.
- **D1 and F1 are thin in LoCoMo.** STALE was 2.1% of wrong answers there. LongMemEval's knowledge
  update category is the closer D1 analogue and is not measured here.
- **The speaker is inside the content in this harness** (`Speaker: text`), so the role that
  content-only windows also drop is not tested. If AML sends the speaker only as a role, C would
  gain more than it can show here.
- **Two collects, one Add-time compiler that is not deterministic.** S and T differ by compile
  randomness as well as by the flag. The 2026-09-24 replicate noise floor on category 1 was 1.42
  points, so C against A differences under about 1.5 points are noise.
- **Stand-in Answer and judge models.** gpt-4o-mini, as before; AML's live models are not
  disclosed. Reader runs also drift between sessions ([[llm-reader-runs-drift-between-sessions]]),
  which is why A is re-run rather than taken from 2026-09-24.
- **Timestamps in content are stored evidence, not instructions.** They are the conversation's own
  message times, which is within the Add contract and is what C6 and earlier variants served.

## What I already know

- `2026-09-24-aml-c9-locomo-loss-diagnosis.md`: 75.3% with dated rendering; TEMPORAL 28.8% of
  wrong answers, about half anchoring errors.
- `2026-09-24-c9-textual-reader-sees-no-dates` (memory): the Smoke facts above.
- `2026-09-20-aml-code4-exact-parity-official.md`: where `message-content-only-v1` came from, a
  Coding parity release, not a Textual decision.
