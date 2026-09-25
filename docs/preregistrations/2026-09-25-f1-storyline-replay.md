# Pre-registration: does an Add-time gpt-4o-mini storyline lift BEAM summarization (AML F1)?

**Date:** 2026-09-25   **Status:** predicted, not yet measured

## The question

AML's Textual smoke on C9 scored 0 on F1 (summarization and long-history synthesis), on one to
three questions. The public BEAM 100K probe (`2026-09-24-c9-beam-ability-probe.md`, branch
`claude/c9-beam-probe`) found the cause is the reader, not retrieval: summarization scored
**0.304** while the rubric evidence was in the returned 100 items for **0.950** of the points, and
time order did not help (-0.027). AML's reader is told to be "direct and concise" in 512 tokens and
leans on the top of the context. BEAM's summarization rubrics have 5 to 8 points, one per phase of
a long history, each with named specifics.

AML's live rules say "Search must not generate final answers or disguise answers as memory
records" (API guide, read 2026-09-25), so a summary written at Search for the question is out.
The rules expect gpt-4o-mini at Add. This test asks whether a synthesis built **at Add, before any
question exists**, and placed on top of the SAME stored retrieval **only for questions that ask for
a summary**, raises the summarization rubric score. It targets the SECOND Full run (user decision
2026-09-25); C9 is not changed.

## Design

Harness: `benchmarks/beam/f1_storyline_replay.py`, tests `tests/test_beam_f1_storyline_replay.py`
(8 tests, 6 mutations shown red, recorded in the test docstring). Answering and judging call the
probe's own functions (`benchmarks/beam/aml_c9_probe.py`, copied unchanged from
`claude/c9-beam-probe` at `74bfc681`): AML's BEAM answer prompt, batch rubric judge, Qwen3-14B with
thinking off for both, temperature 0.

Inputs (sha256 prefixes):

| file | sha256 | origin |
|---|---|---|
| `beam100k.jsonl` | `62a950489d597b68` | the probe's input, 20 conversations, 400 questions |
| `retrieval.jsonl` | `9b3336f52965d123` | the probe's stored C9 Search results, 400 x 100 items, C9 `385c6074` |
| `locomo10.json` | `79fa87e90f040813` | 1,986 LoCoMo questions, only to count gate false fires |

**Simulated Add stream.** Each BEAM session is cut the way AML's Textual adapter cuts it: a chunk
closes at 20 messages or 2,000 words, never crossing a session. Conversations run in parallel;
chunks within a user run in order, as C9's per-user lock serialises Adds. Each chunk is ONE
`openai/gpt-4o-mini` call (temperature 0, JSON mode) that receives the current storyline and the
chunk, and returns a digest of the chunk (80 words at most) and the updated storyline (600 words at
most, dated phases, oldest first, compress oldest first). The builder prompt is `BUILDER_SYSTEM`,
sha256 prefix `44d57274f2c68723`. A failed call keeps the previous storyline (fail closed).

**Gate.** `GATE`, a frozen regex for summary intent in English plus five Chinese words, sha256 of
the pattern prefix `db8d3caec553aa39`. No LLM gate in this stage: on BEAM a regex will match
"Can you provide a ... summary" trivially, so any LLM gate result here would say nothing about
AML's hidden phrasings.

**Arms**, all capped at 100 items (the platform's `top_k`):

| arm | gated questions | other questions |
|---|---|---|
| `r0` | stored items unchanged (concurrent replicate of the probe's `returned`) | same |
| `story` | final storyline at rank 1, then the stored items | r0's answer reused (byte-identical context) |
| `digests` | the user's digests, chronological, BM25 top 24 if more, then stored items | r0's answer reused |
| `story_digests` | storyline, then digests, then stored items | r0's answer reused |
| `story_all` | storyline at rank 1 on EVERY question, ungated (exploratory) | same |

Every BEAM question is asked after the whole conversation, so the storyline used is the one after
the user's last Add. `storycover` asks the probe's coverage judge whether the storyline ALONE holds
each summarization rubric point (the mechanism metric). Paired bootstrap, 10,000 resamples, seed
20260925. Model spend cap: USD 8 in total.

**Contamination.** While designing I read four BEAM **1M** summarization questions with their
rubrics (to learn the rubric shape). I have read no BEAM 100K summarization question, rubric or
answer. The prompt and the gate are frozen by this commit.

## Predictions

`[[i-over-predict-effect-magnitudes]]`: eleven of twelve past predictions were too high by 2 to 4
times, so the effect bands below are deliberately modest.

| quantity | band | point |
|---|---|---|
| `r0` summarization (replicate of 0.304) | 0.25 to 0.36 | 0.30 |
| `story` minus `r0`, summarization | +0.03 to +0.15 | +0.07 |
| `digests` minus `r0`, summarization | -0.02 to +0.10 | +0.03 |
| `story_digests` minus `r0`, summarization | +0.02 to +0.15 | +0.07 |
| storyline-alone coverage of summarization rubric points | 0.35 to 0.70 | 0.50 |
| `story_all` minus `r0`, mean over the 9 other types | -0.05 to +0.01 | -0.02 |
| `story_all` minus `r0`, multi_session_reasoning | -0.05 to +0.08 | +0.01 |
| `story_all` minus `r0`, event_ordering (rubric) | -0.03 to +0.10 | +0.03 |
| gate fires, BEAM summarization | 40 of 40 | |
| gate fires, the other 360 BEAM questions | 0 to 3 | 1 |
| gate fires, 1,986 LoCoMo questions | 0 to 10 | 2 |
| builder latency per Add, p50 | 5 to 15 s | 8 s |
| builder completion tokens per Add, p50 | 600 to 1,100 | 850 |
| builder spend, whole split | under USD 1.50 | USD 0.80 |

## Apparatus checks (a failed check voids the reading, not the run)

1. Every chunk built; at most 2% failed calls.
2. `r0` summarization lands within 0.07 of the probe's 0.304. Outside that, reader drift is too
   large for the deltas to be read.
3. In every gated arm, each ungated question's answer is `r0`'s, byte for byte.
4. No arm answers from more than 100 items.
5. Judge left at most 2 summarization questions unscored in any arm.

## Decision rule, fixed now

Stage 1 (building it into a C9 successor, tested end to end on VPS3) is licensed only if BOTH hold:

- `story` or `story_digests` raises summarization by **+0.05 or more, with the 95% CI lower bound
  above 0**;
- the gate fires on at most 1% of non-summary questions, on BEAM and on LoCoMo separately.

If both arms pass, Stage 1 builds the cheaper one unless the other leads by 0.03 or more. If
`story_all` is non-inferior on the 9 other types (CI lower bound above -0.02), that is recorded as
exploratory evidence for widening the gate, NOT as licence to drop it. Anything else closes this
design; the result goes to the user either way.

## What this cannot show

- The platform's real reader, judge and context format are not public; this uses the probe's
  assumptions (content only, returned order, Qwen3-14B).
- The AML chunking rule is taken from the API guide, not observed on AML Textual data.
- The storyline is not grounded to anchor ids here; Stage 1 must add grounding, and a
  hallucinated specific would be judged as a wrong answer, not flagged.
- The regex gate matching BEAM's phrasing says nothing about AML's hidden F1 phrasings.
- One public split, 40 summarization questions: the CI will be wide.
- Real Add latency inside C9 (the extra call runs beside the 17.9 s p50 compile) is estimated from
  the builder's own latency, not measured in the service.

## Result

(appended below after the run; nothing above is edited)

### Amendment 1, 2026-09-25, before any outcome was read

Nothing in the sections above is edited. Two apparatus failures stopped the first build; no arm
had been answered with a storyline and no judgement had been run when this was written.

1. **The OpenRouter account ran out of credit** (855 total, 853.22 used). 402 responses were
   recorded as fail-closed Adds, which misrepresents an apparatus failure as service behaviour.
   The build now stops on a 402 (`46f41eb7`), and the first 59 `r0` answers were discarded because
   the provider shrinks `max_tokens` when credit runs low.
2. **gpt-4o-mini ignores the 600-word storyline limit in the prompt.** Storylines reached 1,068
   words (p90 933), and past the 1,400-token output cap the JSON was cut off (`finish_reason:
   length`, reproduced on conversation 19 chunk 15); at temperature 0 the six retries fail
   identically, so 48 Adds were fail-closed. That is a defect in the design, not noise: a storyline
   that outgrows its cap stops updating.

**Change (the only one):** the builder's output cap is 2,600 tokens, and whenever the returned
storyline exceeds 700 words a second gpt-4o-mini call (`COMPRESS_SYSTEM`) rewrites it to at most
450 words, keeping every dated phase. This is the "compression" half of F1 enforced by a mechanism
rather than by a prompt instruction. Two tests cover it, both shown red by mutation.

**Consequences for reading the result.** The storyline is rebuilt from scratch for all 20
conversations with the amended builder; the first build is kept aside as
`memory.v1-abandoned.jsonl` and is not used by any arm. The builder latency, token and spend
predictions are read against the main call; compression calls are reported separately, and their
cost is added to the spend. The `r0` answers are independent of the builder and are kept. The
answers for `r0` start roughly 30 minutes before the storyline arms; that gap is recorded as a
drift risk (`[[llm-reader-runs-drift-between-sessions]]`).

### Amendment 2, 2026-09-25, before any storyline arm was answered

The exploratory `story_all` arm (storyline on every question, ungated) is **not run**. Two
reasons, both known before any storyline arm existed:

- Another session is testing the ungated placement directly on the same stored retrieval:
  `docs/preregistrations/2026-09-25-c9-beam-session-summaries.md` on `claude/c9-beam-probe`
  (per-session gpt-4o-mini summaries, undated, placed before the items on all 400 questions).
  Running a second ungated arm would duplicate it; its result is cited instead.
- The OpenRouter account shared with the official C9 has little credit left, and `story_all` is
  the largest remaining spend (400 long answers) with no decision riding on it.

The three `story_all` predictions in the table above are therefore left unscored, and the
widening-the-gate clause of the decision rule cannot fire. Nothing else changes.
