# AML Coding multi-view memory plan

Status: M1 closed; M0 retained as the raw base; compiler v2 admission is running and the M2/M3
retrieval apparatus is draft-only as of 2026-09-19.

This document records the next RE-call experiment sequence for the Agent Memory Leaderboard
Coding track. It is not a preregistration, does not authorize an AML Full run, and does not change
any frozen protocol or measured result. Every experiment promoted from this plan needs its own
frozen identities, hypotheses, gates, and result record before execution.

## Decision

Keep raw engineering history as the validated baseline and build typed, source-grounded views
beside it. Retrieve from those views according to the task, while retaining raw evidence as a
rescue path.

The target configuration mirrors the useful separation in RE-call's local setup, where memory,
code, and documentation serve different evidence needs. AML provides one exact `user_id` scope to
Search, so these views must remain inside that scope. They are internal memory types, not separate
AML identities or cross-sample corpora.

The four planned views are:

1. **Raw trajectory:** ordered historical messages with exact code, paths, errors, commands, tests,
   and failed attempts intact.
2. **Code evidence:** deterministic identifiers, file paths, symbols, exceptions, configuration
   keys, tests, commands, and source-neighbour context.
3. **Repository knowledge:** architecture, module ownership, interfaces, constraints, conventions,
   and analogous implementations grounded in historical sessions.
4. **Engineering experience:** task shape, investigation path, root cause or constraint, failed
   approaches, successful repair, outcome, and validation grounded in historical sessions.

New Feature retrieval should favour repository knowledge, analogous implementations, interfaces,
and successful procedures. Bug Fix retrieval should favour exact diagnostics, raw episodes, failed
attempts, repairs, and validation. Search must still return ranked evidence rather than an answer.

## Why this is the next direction

The public AML article reports 62.00 percent Task Solve for MemoraX, or 93 of 150 tasks. The next
industry result is 52.00 percent, or 78 tasks, which is a fifteen task gap. The open-source leaders
are clustered at 52.67 percent, or 79 tasks. The exact winning server and AML configuration are not
public, so the gap cannot be attributed to one hidden algorithm.

The public signals that are relevant to a testable RE-call design are typed memory, distinct
repository and procedure roles, compact type-aware rendering, hybrid retrieval controls, and
automatic experience capture. Those signals support a multi-view hypothesis. They do not prove
that copying MemoraX's visible type names or context size will reproduce its score.

RE-call's own measurements narrow the hypothesis further:

1. The compiler replay retained `E0_raw`. Raw reached 11.76 percent complete coverage at 10 and
   0.2470 mean reciprocal rank. Compiled only reached zero complete coverage and 0.0321 mean
   reciprocal rank. Compiled plus raw recovered raw coverage but fell to 0.2100 mean reciprocal
   rank.
2. The compiler rejected 437 of 495 proposed E1 records and 417 of 467 proposed E2 records. It
   fell back for 168 and 171 of 196 sessions respectively. The current compiler is therefore not a
   clean test of procedure memory.
3. The coding retrieval ladder selected raw dense plus lexical retrieval. Its mean reciprocal rank
   was 0.3213339067, complete coverage at 10 was 0.1764705882, and complete coverage at 100 was
   0.9411764706.
4. Adding SPLADE reduced mean reciprocal rank and coverage at 10. The procedure arm was dominated
   by compiler fallback. Voyage reranking improved mean reciprocal rank over that weak procedure
   arm, but lost rank 100 coverage. The aggressive task pack cut returned characters by 93 percent,
   but reduced complete coverage to zero and increased p95 latency to 6613.73 milliseconds.
5. The graph path had no eligible relations. It cannot contribute until relation creation is
   measured and nonzero.

These results reject the current compiler, SPLADE configuration, and combined task pack. They do
not reject grounded procedure memory, task routing, or reranking when tested independently on a
strong raw candidate pool.

## Benchmark boundary

The plan must remain aligned with the real AML interface:

1. CAMBench Coding contains 150 held-out software tasks, 51 New Feature and 99 Bug Fix. Relevant
   and noisy conditions produce 300 scored attempts.
2. Task Solve is the primary outcome. Retrieval recall, coverage, latency, and context size are
   diagnostics.
3. RE-call controls Add and Search. AML controls Answer, evaluation, aggregation, and publication.
4. Add receives ordered messages plus `request_id`, `user_id`, and `session_id`. HTTP 200 means the
   data is durable and immediately searchable.
5. `user_id` is the sole Search isolation key. `session_id` is grouping metadata, not a Search
   filter.
6. Search receives the original question and exact `user_id`, then returns ranked evidence with
   stable identifiers and nonempty content. It must not return a final answer.
7. AML can admit 117,760 input tokens to Answer and truncates an over-budget result to a prefix in
   participant rank order. Early ordering and record length therefore matter.
8. The public guide says formal Top K is 100, but the live Coding contract must confirm Coding
   behaviour after the track opens.
9. The current open-source checklist says `gpt-4o-mini` during Add and Search. Whether Voyage
   embeddings and reranking are exempt, and whether RE-call's licence qualifies for the division,
   still require written confirmation.

## Experiment order by expected return

### 1. Code-aware raw retrieval — closed

ROI: very high.

Add deterministic boosts for exact paths, symbols, exceptions, test names, commands, and
configuration keys. Restore bounded source neighbours around a matching raw message. Preserve the
existing dense plus lexical pool and measure this as one independent change.

Expected benefit: better rank for exact implementation evidence without trusting generated
abstractions.

Main risk: identifier repetition can over-rank a noisy session. Bound every boost and require no
loss in rank 100 coverage.

Measured outcome: offline retrieval improved, but the executable screen was invalid and its 33
descriptively paired cells favored M0 by 14 solves to 13. M1-only cell wins were 2 versus 3 for
M0, and M1 Search activation was lower. Retain M0 and do not carry this exact configuration into
later arms. See [the decision record](results/2026-09-19-aml-code-aware-raw-decision.md).

### 2. Evidence-first compiler version 2

ROI: very high.

Replace model-generated character offsets with deterministic anchors:

1. Segment each session deterministically and assign stable evidence anchor identifiers.
2. Extract exact code tokens, paths, symbols, errors, tests, and commands locally.
3. Ask `gpt-4o-mini` to select anchor identifiers and create typed records.
4. Resolve selected anchors locally and attach verbatim evidence excerpts.
5. Reject unsupported fields individually. Do not discard an otherwise grounded session because
   one optional field fails.
6. Store raw history in every arm that contains compiled records.

Pilot gates:

1. At least one accepted typed record for 90 percent or more of eligible sessions.
2. Fewer than 10 percent full-session fallbacks.
3. Zero unsupported claims in an independent source audit.
4. No loss in rank 100 coverage against raw.

Expected benefit: reusable architectural and diagnostic memory without the contradictory offset
generation and exact-substring rejection that invalidated the first compiler.

Main risk: abstraction can still omit decisive code detail. Raw rescue and field-level grounding
remain mandatory.

### 3. Task-conditioned multi-view routing

ROI: high.

Classify the Search request as New Feature, Bug Fix, or unknown. Allocate candidate slots and final
rank weight by task type. Unknown tasks use a balanced profile. Routing may change evidence order,
but it may not remove the raw rescue quota.

Expected benefit: the evidence mix matches the different information needs of feature and repair
work while remaining inside the same AML user scope.

Main risk: task classification or a hard quota can suppress the only relevant view. Use soft
weights, explicit fallbacks, and per-stratum reporting.

### 4. Direct reranking ablation — closed on the current raw pool

ROI: medium to high.

Add Voyage reranking directly to raw or to the best independently validated candidate pool. Hold
ingest, candidate width, returned depth, and packing constant.

Expected benefit: the previous cumulative arm improved mean reciprocal rank by 0.0307897531, so a
clean test may recover that ordering gain without inheriting the weak compiler and SPLADE pool.

Main risk: reranking may discard tail coverage. Record calls, fallbacks, ordering changes, and
rank 100 losses.

Measured outcome: the clean Voyage `rerank-2.5` arm reduced present MRR from `0.322645` to
`0.286091`, reduced complete coverage at 100 from `0.941176` to `0.911765`, and raised p95 Search
latency from `368.293 ms` to `1107.133 ms`. Keep this as a locked prior and rerun only if corpus
construction changes materially.

### 5. Context budget sweep

ROI: medium, after selection quality improves.

Sweep token-counted budgets such as 16,000, 32,000, and 64,000 tokens while preserving evidence
diversity and raw rescue. Do not restore the old 7,000 character hard pack.

Expected benefit: reduce distraction and cost without repeating the catastrophic coverage loss of
the combined C4 pack.

Main risk: compactness can look operationally attractive while losing the evidence that determines
Task Solve.

### 6. Entailment at compiler admission

ROI: medium to low.

Use entailment to audit whether a generated field follows from its selected anchors. Do not first
deploy it as a Search-time filter over raw evidence.

Expected benefit: stronger precision in derived records with less risk of deleting imperfect but
useful raw history.

Main risk: false rejection increases fallback and recreates the first compiler failure.

### 7. Graph expansion

ROI: low until relations exist.

Park graph work until a candidate compiler produces nonzero, useful typed relations and an audit
shows queries where expansion changes the candidate set or rank.

## Independent matrix

The next screen must avoid the cumulative confounding in C0 through C4.

| Arm | One question it answers |
|---|---|
| M0 | Raw dense plus exact lexical baseline. |
| M1 | Does code-token boosting plus source-neighbour restoration improve M0? |
| M2 | Do grounded repository knowledge records improve M0? |
| M3 | Do grounded procedure, failure, repair, and validation records improve M0? |
| M4 | Do the validated views plus task-conditioned routing improve the selected base? |
| M5 | Closed on M0: direct reranking lost quality, coverage, and latency; reconsider only after a material candidate-pool change. |

Run the 34-task retrieval screen first. Promote no more than two configurations to a 12-task
executable screen with three seeds. Confirm the winner on all 34 executable tasks with three seeds.
Then test present, absent, adjacent, contradictory, and superseded or noisy controls in separate
local tenants. Only after those gates pass should the configuration advance to AML Smoke and Full.

Every stage reports overall results plus New Feature and Bug Fix strata. A retrieval improvement
cannot promote a configuration that loses executable Task Solve.

## Promotion and stop rules

A feature stays enabled only when all of these are true:

1. Logs prove the mechanism executed, including provider and model identity, calls, fallbacks, and
   whether candidates or order changed.
2. Retrieval coverage does not materially regress, and the larger executable confirmation has
   positive net Task Solve.
3. Relevant versus noisy degradation, false evidence, returned tokens, latency, and cost remain
   within the frozen experiment limits.
4. The result survives source and identity verification on immutable artifacts.

Park these lanes unless new evidence reopens them:

1. The current offset-generating compiler.
2. The measured SPLADE configuration.
3. The combined C4 task pack.
4. Graph expansion without relations.

Do not spend an AML Full run until the live track contract is archived, model and licence questions
are resolved, the submission build is frozen and compliant, and a local task-success confirmation
passes.

## Immediate execution sequence

1. Keep M0 frozen as the raw base; M1 is closed by the measured executable result.
2. Build compiler version 2 behind a separate flag and run its admission pilot before retrieval.
3. If and only if the compiler admission selector passes, freeze the prepared independent M2 and
   M3 comparisons against M0. The draft protocol is
   `preregistration/092-recall-grounded-multiview-retrieval.md` in Agent Memory Bench; its wrapper
   refuses execution while the document remains draft or the admission result does not authorize
   M2 and M3 retrieval.
4. Route only the views that pass independently, then test M4.
5. Keep the closed M5 reranking result as a locked prior unless the candidate pool changes materially.
6. Run bounded executable screens only after retrieval and activation gates pass.
7. Recheck the live AML Coding contract and organizer answers before freezing a submission.

## Evidence record

The measured compiler artifacts are on branch `codex/aml-experience-compiler` at commit
`35ca56e92566be1a23aa4deb457527a3fbd909d3`, under
`docs/results/aml-experience-compiler-v1/952bc048-vps2-retry2/`.

The validated coding matrix recovery is recorded by selection hash
`19e75d615fa991752d9234bf6ccffeec17ac3c4039df4fd3907f8f9a34ce74fc` in
`/home/sentiment/agent-memory-bench-coding-4826a7b0/results/aml-coding-memory-matrix-v1/714d4a81-a0abe03e-4826a7b0-retrieval-repair` on VPS2.

The M0-versus-M1 retrieval and executable evidence is recorded under
`results/aml-code-aware-raw-v1/`, with full checksum manifests and the narrative decision in
`docs/results/2026-09-19-aml-code-aware-raw-decision.md`. Confirmation and robustness were not
authorized.

Background and public sources:

1. [AML Coding Memory article](https://dev.to/aml-/from-remembering-code-to-solving-tasks-how-coding-memory-helps-agents-reuse-engineering-experie-25c6)
2. [AML API Guide](https://agentmemoryleaderboard.ai/api-guide)
3. [AML Rules](https://agentmemoryleaderboard.ai/rules)
4. [AML public repository](https://github.com/AML-memory/agent-memory-leaderboard)
5. [MemoraX public architecture](https://github.com/memorax-ai/memorax-code/blob/4b7fdcc8d413db9656f0c52fd3a1432043429f66/ARCHITECTURE.md)

Recheck the live official sources after the September 20 opening. The current site and live track
contract outrank the article, historical announcement, public repository, and this plan.
