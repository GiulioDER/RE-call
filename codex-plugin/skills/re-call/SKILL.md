---
name: re-call
description: Give an AI agent trusted persistent memory across sessions. Use before planning or answering when earlier runs may contain relevant decisions, failures, experiments, user corrections, unresolved work, or lessons that should influence the current task, and after learning a durable fact that future runs should inherit.
---

# RE-call

RE-call gives an agent continuity across sessions without treating persistence as truth. Use the
focused read path below before acting, then verify the selected evidence against current sources.
The complete tool map is in [references/tool-routing.md](references/tool-routing.md).

## Use memory before acting

Memory is prior project state, not background reading. Before planning, editing, running a build,
or changing external state, call `recall_search` once. Search for the concrete artifact or
operation plus prior decisions, constraints, failures, and supersessions. This is how you discover
relevant history whose existence you could not have known in advance.

If the search abstains, is degraded, or returns no relevant `ok` hit, continue from current sources
and say memory did not establish a constraint. Do not turn a nearby hit into an answer.

If an `ok` hit could change what you will do, select the source that most directly contains the
current approved decision and call `recall_evidence` once with the same intent, that exact `source`, and `max_items=2`. Use this focused bundle instead of mixing the other search hits into
the plan. Follow a declared successor before selecting the source.

Apply the focused evidence by separating four things: the requirement it establishes, historical
description, rejected alternatives, and the current source or test that will verify the change.
An explicit approved decision is normative. Code that has not implemented it yet is the work to do, not a conflict.
A conflict exists only when current code, tests, configuration, or a newer user
instruction explicitly requires incompatible behavior. In a real conflict, the current source or
newer instruction wins and the memory should be corrected.

Budget: at most two RE-call calls for this read path. Do not use reasoning, maintenance, mutation,
calibration, indexing, ingestion, or erasure tools unless the task explicitly requires that
separate operation.

## Verify and close the loop

Inspect the current source, tests, configuration, or external state named by the focused evidence.
Use `recall_current_facts` or `recall_current_state` only when the task needs a structured current
projection; they do not replace discovery. Use `recall_related` only after a trusted seed when a
structural neighbor is needed. Escalate to reasoning tools only when a plain search and focused
evidence cannot answer the question and the task calls for that analysis.

When a session learns one non-obvious, reusable fact, write one Markdown memo in the project's
configured memory directory and add its pointer to the memory index. Include the fact, why it is
true, how to apply it, and the evidence or failure that made it worth recording. Keep front matter,
use a new memo with `supersedes` when a fact changes, and never write secrets, credentials, tokens,
or customer data into memory.

After writing a memo, run the configured RE-call refresh. Verify retrieval with a distinctive phrase
from the memo; a file on disk or a row count alone does not prove that the next session can find it.
If refresh refuses, reports a stale corpus, or lacks a capability, report that state and do not
start a second indexing or embedding run while one is active.

## Boundaries

RE-call retrieval is advisory context. Current source, tests, explicit user direction, and verified
external state remain authoritative for what exists now. A memory abstention is not proof that no
memory exists, and an unavailable trust gate is not empty memory. Never use a nearby hit to fill an
unanswered question. Use the explicit erasure mechanism with the exact recorded source only when the
user requests forgetting data.
