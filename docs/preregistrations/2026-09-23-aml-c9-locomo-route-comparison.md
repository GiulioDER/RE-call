# Pre-registration: does pinning C9's Search to Context4 beat its keyword router on conversational memory?

**Date:** 2026-09-23   **Status:** predicted, not yet measured

## The question

On LoCoMo, served through the AML hosted service with variant
`C9_routed_specialists_grounded_graph_atomic`, does forcing every Search onto the Context4
specialist route raise **turn-level evidence hit@10** over the served keyword router by at least
1.0 point, with a paired 95% bootstrap interval that excludes zero?

## Why it is being asked

The audit of official AML Multimodal run `teval_dcc1109c4331c3e3` found that
`recall_aml/specialists.py::route_query` sends a question to Context4 only when it contains an
English conversational keyword, and to Code4 by default. Offline, over 1,525 LoCoMo questions read
from `docs/results/2026-09-11-structural-edge-performance-locomo-full.json` and
`benchmarks/audit_data/locomo_errors.json`, it routed **1,079 to code (71%), 418 to context,
28 to multimodal**. The AML Textual Full run will be served by C9, and only two Full runs are
allowed per key and track, so the routing policy should be decided on evidence before that run
starts.

## What I predict

| quantity | prediction |
| --- | --- |
| router arm, share of questions routed `code` | 65% to 78% |
| router arm, share routed `context` | 20% to 33% |
| router arm, share routed `multimodal` | 1% to 4% |
| turn hit@10, all-Context4 minus router | **+1.0 to +3.0 points** |
| turn hit@10, all-Code4 minus router | -0.3 to -1.5 points |
| turn hit@100, all-Context4 minus router | 0.0 to +1.5 points |
| session hit@10, all-Context4 minus router | 0.0 to +2.0 points |

The reasoning. Context4 beat Voyage 4 by +6.25 points hit@5 on LoCoMo, embedding-only, with a
bootstrap interval of +3.65 to +8.92 (`docs/results/2026-09-13-voyage-context4-followup.md`, on
branch `codex/voyage-context4-production-20260913`). Code4 has never been measured against either
on conversation. C9 fuses canonical BM25 with the dense leg and adds a graph sidecar and atomic
views, which should damp any difference in the dense leg. Pinning moves about 71% of questions.
The ceiling is therefore roughly +4 points. My own record of earlier pre-registrations shows that
my interventions capture a quarter to a half of a ceiling, which gives +1 to +2, and I widen that to +3 for the
unmeasured Code4 gap.

## Decision rule, fixed now

- **Pin Context4 for the Textual run** if all-Context4 minus router on turn hit@10 is at least
  +1.0 point, with a 95% interval above zero, **and** turn hit@100 is not lower by more than
  0.5 point.
- **Otherwise keep the router.**
- The Coding track is out of scope. This measurement says nothing about it, and the router stays
  as it is there.

## What would falsify this

- A turn hit@10 difference below +1.0 point, or an interval that includes zero.
- A router-arm route distribution outside the predicted bands. That would mean the offline count
  does not describe the served router on this data.

## How it will be measured

- **Script:** `scripts/aml_locomo_route_compare.py`, committed together with this record.
- **Dataset:** LoCoMo `locomo10.json` (snap-research). The file's SHA256 is recorded in the output
  and compared with `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`, the copy
  frozen for the 2026-09-13 study.
- **Service:** built in process by `recall_aml.__main__.build_app` at this record's commit.
  Variant C9. A fresh database on VPS3 and a table of this run's own. Voyage Code4 and Context4
  embedders, gpt-4o-mini through OpenRouter for the Add-time compiler. It is driven over the
  real ASGI app through Starlette's `TestClient`.
- **Add:**
  - One Add per LoCoMo session, one user per conversation.
  - Each turn is one message: role `user`, content `"<speaker>: <text>"`, with
    ` [shared image: <blip_caption>]` appended when the turn carries one.
  - Every turn gets its session's timestamp in Unix milliseconds.
  - All Adds complete before any Search.
- **Arms:** each arm replaces `recall_aml.service.route_query`, the single call site, and is
  Search-only over the one ingested corpus.

  | arm | route |
  | --- | --- |
  | `router` | the served function, unchanged |
  | `context` | always `"context"` |
  | `code` | always `"code"` |

  For every question the three arms run back to back, in a fixed arm order.
- **Questions:** every LoCoMo question in categories 1 to 4 with at least one evidence id that
  matches `^D\d+:\d+$` and resolves to an ingested turn. n is recorded in the output.
- **Metrics, per arm, over the first d returned items:**
  - **turn hit@d**, the primary metric at d = 10: 1 when any gold evidence turn is present in
    any of the first d items. "Present" means the item content, whitespace-normalised,
    contains the turn's full normalised content. A turn longer than 12 words also counts when
    its first 12 or its last 12 words are present, because 160-word windows with stride 120 can
    split a long turn. Compiled graph records paraphrase and cannot match; they count at session
    level only.
  - **session hit@d:** 1 when any of the first d items carries a `session_id` of a session that
    holds gold evidence.
  - Reported at d in 5, 10, 20 and 100.
- **Statistics:**
  - The paired difference of each arm against `router`.
  - A 95% percentile bootstrap over questions: 10,000 resamples, seed 20260923.
  - Discordant counts (rescues and regressions).
- **Monitors:** the route the router arm actually took per question, recorded inside the
  replacement and compared with an offline call of `route_query`; the share of compiled records
  in each arm's top 10; and Add failures, compiler fallbacks and HTTP errors, each required to
  be zero.

## Apparatus checks, before the full run

1. Unit tests for evidence parsing, the presence matcher, timestamp parsing and the paired
   bootstrap. Each is proved red by a mutation of the function it guards.
2. A smoke run on one conversation and 20 questions. Every forced arm must record only its
   forced route, and a canary question built from the verbatim text of one ingested turn must
   score turn hit@10 = 1 in all three arms.

## What I already know

- Context4 against Voyage 4 on LoCoMo: +6.25 points hit@5, embedding-only (above).
- The 2026-09-20 study "Code 4 and Context 4 coding fusion" closed Context4 for coding: a direct
  replacement and two protected suffix policies failed their frozen gates. That was the Coding
  track, not conversation.
- The offline route counts above come from the same questions this run uses, so the mechanism
  prediction is not independent of them. It tests whether the served router behaves as the
  offline one does, not whether the count is new.

## Confounds I can name now

- **LoCoMo is not AML Textual.** The AML Textual sources are not known here. A result on LoCoMo
  is evidence for conversational memory in general, not a measurement of the official score.
- **The routes differ in more than the embedder.** The Context4 store also admits compiled
  records, `stable_window_order` applies only on the code route, and the atomic views are built
  per scope. The comparison is between routes as deployed, which is the decision being made,
  not between embedding models.
- **The turn-level match undercounts compiled records**, which appear more on the context route.
  That biases against the prediction. Session hit and the compiled share are reported so the
  size of the bias can be seen.
- **One run, one seed for the compiler.** gpt-4o-mini output at Add time is shared by all arms,
  so it cannot favour one of them, but it is one draw.
- **Provider nondeterminism** in Voyage embeddings is small and is shared across arms, because
  the corpus is embedded once.

## Amendment before measurement (2026-09-23, before any run)

The first committed version of the harness, `a3d56b35`, treated a single 5xx response as a
failure. For an Add, that would have silently dropped a whole session from the corpus shared by
all arms, because Voyage runs with provider retries disabled. The AML platform retries a 5xx with
the same request, and Add is idempotent by `request_id`. So the harness now retries a 5xx on Add
and on Search up to four attempts in total, with backoff, and reports the count as `retries`. The
predictions, the metrics and the decision rule above are unchanged. Nothing had been measured when
this was written.
