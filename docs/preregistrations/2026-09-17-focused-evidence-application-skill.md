# Focused evidence application skill

Status: preregistered before implementation or measurement.

## Question

Can a short RE-call skill improve agent quality by separating discovery from application, then
re-reading only the governing source before work begins, without a mutation checkpoint?

This follows three measured results from the external agent-memory-bench harness:

1. `official-015` did not improve quality with a long authority and query-construction protocol.
2. `official-016` showed a large oracle ceiling when the exact current memory was supplied before
   work.
3. `official-017` reached the current authored source in 26 of 35 triggered treatment sessions,
   but succeeded in only 13 of those 26. Its deny-once checkpoint also reduced success sharply and
   raised cost.

Those results motivate a new mechanism rather than another graph-first or checkpoint variant. The
first call discovers candidate memory. A second, source-scoped evidence call removes distractors
and makes the selected source the sole memory context for application. The skill also corrects a
specific observed interpretation error: code that has not implemented a remembered decision yet
does not contradict that decision.

## Frozen treatment contract

The treatment's read path is exactly the text inside this block, without the fence and with one
terminal newline:

```text
## Use memory before acting

Memory is prior project state, not background reading. Before planning, editing, running a build,
or changing external state, call `recall_search` once. Search for the concrete artifact or
operation plus prior decisions, constraints, failures, and supersessions. This is how you discover
relevant history whose existence you could not have known in advance.

If the search abstains, is degraded, or returns no relevant `ok` hit, continue from current sources
and say memory did not establish a constraint. Do not turn a nearby hit into an answer.

If an `ok` hit could change what you will do, select the source that most directly contains the
current approved decision and call `recall_evidence` once with the same intent, that exact
`source`, and `max_items=2`. Use this focused bundle instead of mixing the other search hits into
the plan. Follow a declared successor before selecting the source.

Apply the focused evidence by separating four things: the requirement it establishes, historical
description, rejected alternatives, and the current source or test that will verify the change.
An explicit approved decision is normative. Code that has not implemented it yet is the work to
do, not a conflict. A conflict exists only when current code, tests, configuration, or a newer user
instruction explicitly requires incompatible behavior. In a real conflict, the current source or
newer instruction wins and the memory should be corrected.

Budget: at most two RE-call calls for this read path. Do not use reasoning, maintenance, mutation,
calibration, indexing, ingestion, or erasure tools unless the task explicitly requires that
separate operation.
```

The shipped `re-call` skill may contain lifecycle guidance after this exact read path and may link
to a reference that maps the full tool surface. The benchmark treatment is the frozen block only.
No full 22-tool catalogue enters the benchmark prompt.

## Product changes allowed before measurement

1. Put the frozen read path in both distributed `re-call` skills with byte-identical semantics.
2. Add a lazily loaded reference that maps all 22 tools by role. The default path must remain
   `recall_search`, then conditional source-scoped `recall_evidence`.
3. Teach the direct skill installer to preserve supporting skill files, because a copied
   `SKILL.md` with a missing reference is a broken install.
4. Add tests for bundle parity, the two-call default path, all 22 tool names, and recursive skill
   installation. These are apparatus checks, not quality measurements.

No retrieval ranking, corpus, trust threshold, reranker, graph expansion, task, checker, or model
change is authorized by this preregistration.

## Frozen benchmark design

Run id: `official-018-recall-focused-evidence-application-paired`.

Use the external `agent-memory-bench` harness only as a measurement instrument. Publish the
preregistration, implementation, verification receipt, aggregate results, and conclusion in the
RE-call repository. Do not publish this product-development result as a benchmark-repository
feature branch.

The run uses:

- the same model, corpus, `superseded` condition, task fixtures, checker, sandbox, trust settings,
  reranker-off setting, and 22-tool surface used by `official-017`;
- tasks `ts-base36-id`, `ts-bom-merge`, `ts-golden-regen`, `ts-ignore-gen`, `ts-legacy-hash`,
  `ts-mig-name`, `ts-natural-order`, `ts-schema-additive`, `ts-semver-pin`, and `ts-tz-utc`;
- seeds 0 through 4;
- a contemporaneous control using the exact `official-012` full-tools protocol and a treatment
  using the frozen block above;
- three participant workers;
- 50 paired cells and 100 participant sessions.

Before the grid, freeze and record the benchmark commit, RE-call commit, control and treatment
digests, corpus fingerprint, exact model identifier, tool inventory, worker count, and environment
receipt. A setup mismatch stops the run.

## Endpoints

Primary endpoint:

1. Paired checker success. Report both-success, treatment-only, control-only, both-fail, net wins,
   success-rate difference, and the exact paired sign or McNemar test.

Mechanism endpoints, in order:

1. Search before any non-memory tool and successful search exposure.
2. Governing-source exposure in the first search.
3. Conditional source-scoped `recall_evidence` use when the governing source was exposed.
4. Focused bundle identity: selected source equals the governing current source, item count is at
   most two, and every admitted item has trusted status.
5. Checker success conditional on governing-source exposure and on a valid focused bundle.
6. False-conflict interpretation, defined before the run as rejecting an explicit current
   decision merely because the repository has not implemented it yet.
7. Wrong-fact application and attributable damage.
8. Calls per session, forbidden tool use, tool errors, participant errors, timeouts, input and
   output tokens, model turns, memory latency, wall time, and estimated spend.

Historical runs are motivation only and are not pooled into the paired endpoint.

## Frozen predictions

1. At least 90 percent of admitted treatment sessions will call `recall_search` before any
   non-memory tool, and treatment exposure will not be more than three percentage points below
   control.
2. At least 70 percent of treatment sessions whose first search exposes the governing source will
   make one source-scoped `recall_evidence` call to that source.
3. At least 95 percent of admitted treatment sessions will use no more than two RE-call calls, and
   none will call a maintenance or mutation endpoint.
4. Treatment will produce at least three net paired wins over control. A smaller positive
   difference is not a demonstrated practical gain.
5. Checker success conditional on governing-source exposure will be at least eight percentage
   points higher in treatment, provided both arms have at least ten qualifying sessions.
6. False-conflict interpretation will be lower in treatment and occur in at most one admitted
   treatment cell.
7. Treatment wrong-fact application will not exceed control and will occur in at most one admitted
   cell.
8. Treatment mean input tokens will not exceed control by more than 15 percent, and mean wall time
   will not exceed control by more than 20 percent.

## Decision rule

The skill advances only if prediction 4 and prediction 7 pass, at least 40 paired cells are
admitted, and no setup or trust invariant fails. Predictions 1 through 3 must also pass for the
mechanism to be considered implemented as designed.

If source exposure is adequate but focused evidence use is low, the skill is not following its own
routing contract. If focused bundle use is high but conditional success does not improve, close
the prompt-only application lane and move the source-focus mechanism into a product-generated
context surface. If quality improves but cost exceeds either bound, simplify the treatment before
promotion. No post-run prompt edit, alternate endpoint, task removal, reranker activation, corpus
refresh, or threshold change may rescue a failed result.

<!-- results and append-only corrections go below this line; everything above is frozen -->

## Results appended after the frozen marker

Measured 2026-09-17 in run
`official-018-recall-focused-evidence-application-paired-retry`, using the frozen model, corpus,
superseded condition, three workers, and 22-tool surface. The benchmark harness required the
`bare` baseline, so the executed grid was 150 sessions rather than the 100 participant sessions
specified in the design. The admission gate admitted 40 of 50 paired cells.

The signed adjudication receipt verified successfully. The paired primary endpoint was:

| Arm | Successes | Admitted cells | Rate |
| --- | ---: | ---: | ---: |
| `recall_graph_fulltools_protocol` | 34 | 40 | 85.0% |
| `recall_graph_fulltools` | 28 | 40 | 70.0% |

There were 26 both-success cells, 2 treatment-only wins, 8 control-only wins, and 4 both-fail
cells. Net wins were `-6`, the success-rate difference was `-15.0` percentage points, and the
exact McNemar p-value was `0.109375`.

The treatment called `recall_search` before any non-memory tool in 11/40 admitted sessions and
used search in 16/40. It made five `recall_evidence` calls across three sessions, with two
non-abstaining successes, one abstention, and two tool errors. Focused-bundle identity and
governing-source exposure were not observable in the redacted final records. The treatment made
no mutation calls and stayed within two RE-call calls in 38/40 sessions.

Wrong-fact application was 1/40 for treatment and 0/40 for control. Mean treatment input tokens
were 63.68% below control and mean wall time was 1.41% below control.

Predictions 4 and 7 failed. Prediction 1 also failed because the treatment search-before-action
rate was 27.5%, far below the preregistered 90% threshold. Prediction 3 passed. Predictions 2,
5, and 6 were not observable from the redacted records. Prediction 8 passed. The decision rule
therefore returns `STOP`: do not promote this prompt-only variant. The complete aggregate and
publishable receipt artifacts are in
[`docs/results/2026-09-17-focused-evidence-application-official-018.md`](../results/2026-09-17-focused-evidence-application-official-018.md).
