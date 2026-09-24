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
