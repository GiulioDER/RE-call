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

## Result (2026-09-23, run `full1`, appended 2026-09-24)

**Status:** measured. **The central prediction is falsified in direction.** Pinning Context4 is
worse than the router, and the decision rule says **keep the router**.

Run facts:
- Harness `3369db1b` on VPS3, run `full1`, 4,525.7 s in total, 3,358.0 s of it Adds.
- Artifact `docs/results/2026-09-23-aml-c9-locomo-route-comparison.json`, SHA256
  `961d50ab1c46566c49b02eb1876441c567f4089a6ed53ca61539d1ddb7c07d93`.
- 272 Adds, 1,535 questions, all 1,535 scored in all three arms.
- 0 HTTP failures and 0 retries. Forced-route violations: 0 for Context4, 0 for Code4.
- The router arm matched the offline `route_query` on 1,535 of 1,535. The canary scored
  turn hit@10 = 1 in all three arms. The dataset matched the pinned SHA256.

| quantity | predicted | measured |
| --- | --- | --- |
| router arm routed `code` | 65% to 78% | 70.8% (1,087) |
| router arm routed `context` | 20% to 33% | 27.4% (420) |
| router arm routed `multimodal` | 1% to 4% | 1.8% (28) |
| turn hit@10, Context4 minus router | **+1.0 to +3.0** | **-1.30 [-2.35, -0.26]**, 24 rescues / 44 regressions |
| turn hit@10, Code4 minus router | -0.3 to -1.5 | **+0.72 [+0.20, +1.24]**, 14 / 3 |
| turn hit@100, Context4 minus router | 0.0 to +1.5 | -0.13 [-0.39, +0.13] |
| session hit@10, Context4 minus router | 0.0 to +2.0 | -1.17 [-2.15, -0.26] |

Absolute levels, turn hit@10: router 93.09%, Context4 91.79%, Code4 93.81%. Turn hit@100 is 99.87%
for the router and Code4, and 99.74% for Context4.

**Gap.** Both route predictions have the wrong sign. The mechanism predictions held: the served
router behaves exactly as the offline one. What failed is the premise, carried over from the
2026-09-13 embedding-only study, that Context4's advantage over Voyage 4 would survive inside C9.
It does not. Split by the route the router chose, turn hit@10 was:

| questions the router sent to | n | router | Context4 | Code4 |
| --- | --- | --- | --- | --- |
| code | 1,087 | 92.9 | 91.1 | 92.9 |
| context | 420 | 93.6 | 93.6 | **96.2** |

So the router's Context4 route costs about 2.6 points on exactly the questions it sends there.
The whole Code4 gain comes from those 420 questions.

**The confound named in advance does not explain it.**
- The Context4 route puts 2.18 non-raw items (compiled records and atomic views) into the top 10,
  against 0.57 for the router and 0.00 for Code4. Those items cannot match at turn level.
- But session hit@10, which does count them, moves the same way: -1.17 [-2.15, -0.26].
- The likelier mechanism is that the non-raw items displace raw evidence windows from the top
  10. That is an inference, not measured here.

**Monitor deviations, recorded rather than smoothed.**
- 3 of 272 Adds fell back in the compiler, where the pre-registration asked for zero. Their raw
  windows were stored and are shared by every arm; only those three sessions' compiled records
  were dropped.
- 130 Searches returned 80 items rather than 100: all of them `conv-30` on the Code4 route,
  whose Code4 store holds 80 windows in total. That is a small corpus, not a harness failure.

**Build.** Master `35ff7477` plus the harness. `git diff 936b7bda 35ff7477` leaves `recall_aml`
unchanged, and its `recall/` changes (the `iter_chunks_with_times` text flag, graph deletion,
four package exports) are code C9 never calls or imports. So this measures the baseline build as
far as C9 is concerned.

**Not decided by this record.** Code4 minus router is +0.72, below the +1.0 bar, and pinning
Code4 was not a pre-registered arm of the decision rule. Pinning Code4, or dropping the Context4
route from the Textual deployment, is therefore a new hypothesis, with this record as its motivation
rather than its evidence. LoCoMo is not AML Textual.
