# Pre-registration: can C9 show the reader a date on Textual without losing Coding retrieval?

**Date:** 2026-09-25   **Status:** predicted, not yet measured. Measurement waits for the user's
explicit go.

## The question

Two results from 2026-09-24 pull against each other:

- `2026-09-24-aml-c9-reader-dates.md`: when AML's reader sees `content` alone, C9 loses 17.5
  LoCoMo temporal points, and timestamping every message in the window
  (`content_only_windows=False`) wins back +18.1.
- `2026-09-24-aml-c9-coding-window-check.md`: the same flag costs the Coding screen 0.09 MRR
  (0.8627 to 0.7728), 8 tasks worse and 1 better. One endpoint serves both tracks, so the flag
  cannot simply flip.

Two candidates, both behind default-off variant flags, so served C9 is unchanged
(`test_served_c9_keeps_both_formats_off`):

- **P, per-track windows** (`per_track_windows`). Each Add chooses its own window renderer:
  content-only when `recall_aml.window_format.looks_like_coding` calls it a coding trajectory,
  timestamp, role and content otherwise.
- **H, dated Search content** (`dated_search_content`). Nothing stored, embedded or ranked changes.
  At Search time each returned text item's `content` is prefixed with its own `created_at`, as
  `[2023-05-08 13:56 UTC] …` (`dated_items`).

Does either pass both screens?

## A limit stated up front

The user noted on 2026-09-25 that the only AML signal so far is the **Textual** Smoke; C9 has no
AML Coding result at all. Both Coding figures here come from our own synthetic Agent Memory Bench
screen, with per-event input that guesses AML's payload shape. That makes them a proxy. The rule
below therefore ends at a recommendation, and the AML Coding Smoke on the chosen build outranks
this screen whenever it exists.

## Why these two designs

- H cannot touch Coding retrieval: the flag acts after the ranking is final. Its Textual effect
  should match arm A of the reader-dates run, which showed the reader the same date through
  `created_at`. It needs no detector. It does put a date prefix in front of every Coding item the
  reader sees, which a retrieval screen cannot evaluate.
- P keeps Coding exactly as served, if the detector recognises the trajectory. But the detector is
  a fixed content rule (a code file name, or two kinds of code syntax, in at least 25% of an Add's
  messages), and it can be validated only on our own corpora. It never reads `request_id`,
  `user_id` or `session_id`, whose AML values carry the benchmark name; branching on those would be
  dataset-specific behaviour.

## How each arm is measured

- **P, offline** (`scripts/aml_c9_window_format_check.py`, no model and no database). Classify all
  272 LoCoMo Adds (as the reader-dates collects sent them) and all 196 Coding Adds (one message per
  event, as the Coding check sent them). Then, for each Add, compare the raw window texts that
  `build_chunks` writes with `per_track_windows` on against the renderer that Add's track was
  measured with: timestamp, role and content (arm C) for LoCoMo, content-only (K0) for Coding.
  Where every text is identical, P's windows ARE those arms' windows, and P inherits their
  measured results.
- **H on LoCoMo, a new answer arm on the existing retrieval.** `answer --reader-view
  product-dated` over `collected-S.json.gz` (the reader-dates served collect, still on VPS3). It
  applies the product's own `dated_items` to each item's stored `content` and `created_at`, so the
  reader sees exactly what C9 with `dated_search_content` returns, over identical retrieval. Then
  the official judge and paired comparisons against arms A and B of 2026-09-24.
- **B′, a replicate of arm B** (`--reader-view content` on the same collect), answered in the same
  session as H, because reader runs drift between sessions
  ([[llm-reader-runs-drift-between-sessions]]). H is compared with B′ as well as B.
- **H on Coding, K3.** A fourth Coding collect,
  `scripts/aml_c9_coding_window_check.py collect --arm K3 --dated-search-content`, in its own
  fresh database (`codingcheck_k3_20260925`), compared with K0 and K0b of 2026-09-24. It also
  records the share of returned items that carry the header, which is the live proof that the
  Search wiring fires.
- Code: this record's commit. `recall_aml` differs from the served `385c6074` only by the two
  default-off flags, the `window_format` module and a `search_content_profile` key in `/version`.

## What I predict

| quantity | prediction |
| --- | --- |
| LoCoMo Adds flagged as coding | 0 of 272 (band 0 to 3) |
| Coding Adds flagged as coding | 196 of 196 (band 185 to 196) |
| P windows identical to the measured arm | every correctly classified Add |
| **H minus B, LoCoMo temporal** | **+12 to +20** |
| H minus B′, LoCoMo temporal | +12 to +20 |
| H minus A, LoCoMo all | -1.5 to +1.5 |
| H minus A, LoCoMo temporal | -4 to +4 |
| B′ minus B, LoCoMo all | -1.5 to +1.5 |
| **K3 minus K0, Coding MRR** | **-0.005 to +0.005** |
| K3 recall@10 | 34 |
| K3 mean share of items carrying the header | 0.80 to 1.00 |

Reasoning. LoCoMo's turns carry no Python, JSON or file names, while almost every Coding
trajectory reads files, so both detector bands are near their ends; the Coding band is the wider
one because a short session with few tool events could fall under the 25% share. H gives the
reader the same session date arm A gave, so it should land near A; minute precision in a bracket
instead of an ISO timestamp is the only difference. K3 ranks exactly as K0 by construction, and K0b
reproduced K0 exactly on 2026-09-24, so its band is tight. Items without a `created_at`, such as
some compiled records, stay undated, hence the header share band.

## Decision rule, fixed now

N is the Coding noise floor, |K0b minus K0| on MRR (0 on 2026-09-24).

1. **Apparatus failure.** If any check below fails, report and decide nothing.
2. **P passes** if all 272 LoCoMo Adds are unflagged, all 196 Coding Adds are flagged, and every
   window text is identical to its measured arm. It then inherits arm C on LoCoMo (+18.12 temporal
   over B) and K0 on Coding (MRR 0.8627).
3. **H passes** if all of these hold:
   - H minus B on LoCoMo temporal is at least +10 with its interval above 0, and so is H minus B′;
   - H minus A on LoCoMo all is at least -1.0;
   - K3 is non-inferior to K0 and K0b: MRR at least -(0.02 + N), recall@10 and recall@100 at least
     the smaller of theirs, and **median** Search latency at most 1.25 times the larger. The
     median replaces p95: the Coding check showed that p95 over 34 queries is a single query.
4. **Recommendation to the user:**
   - If H passes, recommend H. It needs no detector, it leaves retrieval unchanged on both tracks,
     and it does not depend on how AML shapes a Coding payload.
   - If only P passes, recommend P, stating that the detector was validated only on our corpora.
   - If neither passes, report, and adopt nothing.

   Whatever is recommended is still the user's decision
   ([[official-textual-coding-config-graph-on]]). It should go through an AML Coding Smoke before
   any Full run, since that Smoke is the first real Coding evidence either way.

## What would falsify this

- Any quantity outside its band.
- In particular: a flagged LoCoMo Add or an unflagged Coding Add means the detector is less clean
  than I think. H minus B below +12 means a bracketed date helps the reader less than
  `created_at` did in arm A. K3 moving MRR at all means the flag leaks into ranking, which the
  design says it cannot.

## Apparatus checks

1. `/version` of K3 reports `search_content_profile` `created-at-header-v1` and the served window
   renderer (enforced by the collect).
2. K3 writes exactly 1,220 raw windows and has no Add or Search failure.
3. H's prompts carry the `[YYYY-MM-DD HH:MM UTC]` prefix and B′'s carry none, checked on 20 prompts
   each drawn by seed 0.
4. The judge answers its known-answer check correctly at least 95% of the time.
5. The new tests fail on their intended assertion under the mutation their docstrings name
   (`tests/test_aml_window_format.py`, 5 tests; `tests/test_aml_locomo_loss_diagnosis.py::
   test_the_product_dated_view_is_what_the_service_returns`), and the AML test set passes.

## Cost and where

VPS3, `/home/sentiment/reader-dates`. OpenRouter: H and B′ answers plus judges at about 4.5 USD
each, and K3's Add-time compile at about 0.5 USD; cap 15 USD. Voyage for K3 is small. The offline
P check costs nothing.

## Confounds I can name now

- **Coding is a proxy** (see the limit above), and H's date prefix in front of Coding items is not
  evaluated by a retrieval screen.
- **38% of the Textual Smoke's raw windows had no `event_time`.** Neither format can date those.
- **The detector threshold was set before seeing either corpus's classification**, but on a design
  that knows their general shape (conversation against tool trajectories). A real AML Textual
  sample about programming could be misread as a trajectory.
- **LoCoMo holds one date per session**, so a window's `created_at` (the Add's latest timestamp)
  equals every message's date. On AML Adds spanning several days, H shows only the latest.

## Amendment before arm H is measured (2026-09-25)

**Already measured, and unchanged by this amendment: arm P.** The offline check ran at `cb35b4ca`
and is reported with the result. Everything below concerns arm H only.

**What happened.** The first launch of arm H stopped after about three minutes, when the
OpenRouter account ran out of credit (HTTP 402). At that point 252 of 1,535 H answers and 242 of
1,535 B′ answers existed, all from gpt-4o-mini, and K3 had added 11 of 196 sessions; I stopped K3.
None of it has been judged or compared.

**The change, made by the user for cost.** The answer and judge models change from
`openai/gpt-4o-mini` to **`deepseek/deepseek-v4-flash-0731`** (the pinned snapshot, not the
floating alias), selected through `AML_DIAG_ANSWER_MODEL` and `AML_DIAG_JUDGE_MODEL` (`stage_model`
in `scripts/aml_locomo_loss_diagnosis.py`, test proved red by mutation). Every answer and judge
row now records its model. A one-call probe returned a correct, non-empty answer with a reported
cost.

**What follows from it.** Arms answered by different readers cannot be compared, so:

- The 494 gpt-4o-mini answers from the stopped launch are set aside, kept on VPS3 under
  `wf/stopped-gpt-4o-mini/`, and not used.
- Three LoCoMo arms are answered and judged fresh by DeepSeek Flash over the same retrieval,
  `collected-S.json.gz`: **A′** (`--reader-view dated`, as arm A), **B′** (`--reader-view
  content`, as arm B) and **H** (`--reader-view product-dated`).
- The decision rule's LoCoMo conditions read with the DeepSeek arms: "H minus B" and "H minus B′"
  both become **H minus B′**, and "H minus A" becomes **H minus A′**. The thresholds are
  unchanged: at least +10 on temporal with the interval above 0, and at least -1.0 overall.
- The rows "H minus B", "H minus A" and "B′ minus B" of the prediction table were written for a
  gpt-4o-mini reader. Their bands are left exactly as written; the result reports them against
  the DeepSeek arms and says so, since a different reader is a reason a band can miss.
- The judge known-answer check reruns with the DeepSeek judge and must still reach 95%.
- **K3 is unchanged.** Its Add-time compile is C9's served `openai/gpt-4o-mini`, because the arm
  must match production, and it restarts from an empty database.

Cost: about 3 USD for the three DeepSeek answer arms and their judges, and about 0.5 USD for K3's
compile, against a balance of 10.53 USD read before launch.

## Result (2026-09-25)

**Status:** measured

**Run facts.** VPS3, `/home/sentiment/reader-dates`.
- **Arm P**, offline, at `cb35b4ca`, 2 seconds, no model.
- **Arm H**, at `96155776`, 05:13 to 08:38 UTC. The answer and judge model was
  `deepseek/deepseek-v4-flash-0731` on every one of the 3 x 1,535 answer rows and judge rows
  (verified from the rows' `model` field). No row is duplicated.
- **K3**: 196 Adds and 34 Searches, 0 failures, 1 compiler fallback, 1,220 raw windows, 2,909 s,
  SHA256 `d50861f0…`.
- Answers, judge labels, the judge self-check, the comparisons, both reports and K3's per-task rows
  are in `docs/results/2026-09-25-aml-c9-window-format/`. The report is built by
  `scripts/aml_c9_window_format_report.py`.
- **OpenRouter spend** on the rows: A′ 1.77 USD, B′ 4.17, H 1.76, self-check about 0.01. That makes
  7.7 for the DeepSeek arms, against my estimate of about 3; B′'s undated prompts drew long hidden
  reasoning. Not recorded by the harness: K3's compile (about 0.5 at the 2026-09-23 rate) and the
  494 gpt-4o-mini answers of the stopped launch. The total is inside the 15 USD cap.
- **Interim pass.** Another session added `interim.sh` in the run directory at 06:38, at the user's
  request, and it judged A′ and H into the same files this run resumes from. Resume skips judged
  ids, so each id was judged exactly once (1,535 unique ids per arm, no duplicate rows). Its interim
  B′ files are separate and are not used here.

**Apparatus checks.**

| check | required | measured | pass |
| --- | --- | --- | --- |
| 1. K3 `/version` | `created-at-header-v1`, content-only windows | as required (enforced) | yes |
| 2. K3 windows and failures | 1,220, none | 1,220, 0 Add and 0 Search failures | yes |
| 3. views | H carries the `[YYYY-MM-DD HH:MM UTC]` prefix, B′ none (20 each, seed 0) | H 20/20, B′ 0/20, A′ 20/20 ISO | yes |
| 4. judge known answer | at least 95% | 101 of 103 (98.1%), with the DeepSeek judge | yes |
| 5. tests | red by mutation; AML set green | 7 of 7 red; 337 passed, 4 skipped | yes |

**Reported, not a check: UNPARSED judge labels**, counted as WRONG as the method says. A′ 106, B′
200, H 114. Nearly all are one formatting quirk: DeepSeek copies the doubled braces `{{ … }}` of
AML's judge prompt, which the official parser rejects, while the verdict inside is plain. The
exploratory recovery below measures what it moved.

**Arm P, offline.**

| corpus | Adds | flagged as coding | windows identical to the measured arm |
| --- | ---: | ---: | ---: |
| LoCoMo | 272 | 0 | 272 |
| Coding | 196 | **187** | 187 |

The nine Coding Adds it misread are distractors d050, d142 and d152, and six task sessions:
ts-append-only, ts-base36-id, ts-round-money, ts-stable-sort, ts-tz-utc and xs-evolve-lease p02.
Three of those tasks (ts-append-only, ts-base36-id, ts-stable-sort) lost rank under timestamped
windows on 2026-09-24.

**Arm H, LoCoMo, DeepSeek reader, pre-registered scoring.**

| arm | all | cat 1 (282) | cat 2 temporal (320) | cat 3 (92) | cat 4 (841) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A′, dated view | 70.68 | 49.29 | 60.31 | 58.70 | 83.12 |
| B′, content only | 55.24 | 43.97 | **7.81** | 51.09 | 77.53 |
| H, product-dated | 70.49 | 49.65 | 61.56 | 54.35 | 82.64 |

| comparison | all | temporal |
| --- | --- | --- |
| H minus B′ | +15.24 [+12.90, +17.65] | **+53.75** [+48.12, +59.38] |
| H minus A′ | **-0.20** [-1.82, +1.50] | +1.25 [-2.50, +5.00] |
| B′ minus A′ | -15.44 [-17.85, -13.09] | -52.50 [-58.12, -46.88] |

**Arm H, Coding.**

| | K0 | K0b | K3 |
| --- | ---: | ---: | ---: |
| MRR | 0.8627 | 0.8627 | 0.8627 |
| recall@10, recall@100 | 34, 34 | 34, 34 | 34, 34 |
| median Search ms | 606 | 702 | 561 |
| items carrying the header | 0 | 0 | **1.00** |

K3 minus K0 on MRR is **0.0000** [0, 0]. No task's first relevant rank moved.

**Predictions against measurements.**

| quantity | predicted | measured | in band |
| --- | --- | --- | --- |
| LoCoMo Adds flagged | 0 (0 to 3) | 0 | yes |
| Coding Adds flagged | 196 (185 to 196) | 187 | yes, but short of the pass rule's 196 |
| P windows identical where correctly classified | every one | 272 and 187 | yes |
| H minus B, temporal (read against B′) | +12 to +20 | +53.75 | no, above |
| H minus B′, temporal | +12 to +20 | +53.75 | no, above |
| H minus A, all (read against A′) | -1.5 to +1.5 | -0.20 | yes |
| H minus A, temporal (read against A′) | -4 to +4 | +1.25 | yes |
| B′ minus B, all | -1.5 to +1.5 | not comparable: B′ is DeepSeek, B gpt-4o-mini | n/a |
| K3 minus K0, MRR | -0.005 to +0.005 | 0.0000 | yes |
| K3 recall@10 | 34 | 34 | yes |
| K3 header share | 0.80 to 1.00 | 1.00 | yes |

**Gap.** The bands for H minus B were written for gpt-4o-mini, which kept 42% of temporal questions
without dates. DeepSeek Flash keeps **7.8%**. So the size of the loss that the date fixes depends
on AML's reader, which is undisclosed, far more than I assumed. The finding that transfers across
both readers is the mechanism: with the date in `content`, H matches the dated view within 0.2
points on both readers, and without it temporal accuracy collapses on both.

**Exploratory, post hoc: labels recovered.** Removing the doubled braces and reparsing
(`scripts/aml_c9_window_format_recover_labels.py`) leaves 1, 0 and 2 labels unparsed. Accuracy
becomes A′ 72.70, B′ 59.61 and H 73.22. H minus B′ is then +13.62 overall and +54.38 temporal; H
minus A′ is +0.52 overall and +2.19 temporal. Neither decision moves.

**Decision, by the rule fixed above.**
1. Apparatus: every check passes. Does not fire.
2. **P fails**: 187 of 196 Coding Adds were flagged, not 196.
3. **H passes every condition.**
   - H minus B′, temporal: +53.75, interval above 0, at least +10.
   - H minus A′, all: -0.20, at least -1.0.
   - K3 against K0 and K0b: MRR 0.0000, at least -(0.02 + 0). recall@10 34 and recall@100 34, at
     least 34 each. Median Search 561 ms, at most 1.25 x 702 = 878 ms.
4. **Recommendation to the user: H**, a C9 build with `dated_search_content=True`. Nothing stored,
   embedded or ranked changes on either track. It is the user's decision, and it should go through
   an **AML Coding Smoke** before any Full run. That Smoke is the first real Coding evidence either
   way, and it would also show how AML's Coding reader treats a date prefix, which this retrieval
   screen cannot.

## Deployment note (2026-09-25, after the result)

H was promoted before this result existed, on the interim look, at the user's request for the
official run deadline: PR #761 (`efb79146`, merged 07:12 UTC) set `dated_search_content=True` on
C9, and the official C9 on VPS2 has served `efb79146` since 07:13:22 UTC, with
`search_content_profile` `created-at-header-v1`, windows content-only, graph on and the atomic
stage active and fused. The full result above confirms the interim decision; nothing needs undoing.

Read from the C9 journal: AML's client (`[redacted-ip]`) ran a smoke on that build from 07:14 to
07:46 UTC, 134 Adds and 48 Searches, all HTTP 200, no `hosted_request_failed`. Its per-category
scores are on the AML platform, not here.

Live check at 08:50 UTC: one throwaway Add with a timestamp, then one Search, on the official
C9. The Search returned `[2023-05-08 13:56 UTC] Probe: …`, and the delete removed all 6 rows.
