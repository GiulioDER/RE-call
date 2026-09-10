---
name: re-call
description: Use when working with RE-call project memory, prior decisions, retrieval evidence, or the freshness of documentation and code corpora.
---

# RE-call

RE-call is a project-memory layer that searches durable notes and can abstain when the corpus does
not support an answer. Use it to recover relevant project context before acting, and to preserve
new facts so a later session can find them.

## Search before acting

Use `recall_search` before repeating prior work, proposing an approach, changing configuration,
running a build or deployment, or revisiting a decision. Search for the operation and its likely
hazard, not only the task name. For example, search for a deployment rollback or a stale index,
not just “deploy the service”.

Use the corpus that matches the question when separate RE-call tenants are configured:

- project memory for prior decisions, feedback, and postmortems
- documentation for documented behavior and usage
- code for implementation details and symbols

For an exact current implementation, inspect the source after retrieval. Memory is context, not a
replacement for the repository.

## Trust the result correctly

Use `recall_evidence` when an answer depends on recalled facts. Treat only an `ok` verdict as live
evidence. Follow `superseded_by` when a fact was replaced, and treat `expired`, `closed`, or
abstained results as a reason to verify rather than as permission to guess. Check `gap_warning`:
it means the corpus may not cover the question. Use `recall_stats` when freshness, generation,
or coverage matters, and do not mix evidence from different generations without saying so.

## Write durable memory

When a session learns one non-obvious, reusable fact, write one Markdown memo in the project's
configured memory directory and add its pointer to the project's memory index. Include:

- the fact
- why it is true
- how to apply it
- what evidence or failure made it worth recording

Keep the existing front-matter contract and use a new memo with `supersedes` when a fact changes.
Do not silently edit away the old fact. Do not write secrets, credentials, tokens, or customer data
into memory.

## Close the loop

After writing a memo, run the configured RE-call refresh for the project. The refresh should update
the memory corpus and, when enabled by the installation, the documentation and code corpora through
their configured generation pipelines. Verify with a search for a distinctive phrase from the memo;
a file on disk or a row count alone does not prove the next session can retrieve it.

If the configured refresh reports a refusal, stale corpus, or missing capability, report that state
and its evidence. Do not start a second indexing or embedding run while one is active.

## Boundaries

RE-call retrieval is advisory context. Current source, tests, explicit user direction, and verified
external state remain authoritative. Never use a nearby memory hit to fill an unanswered question,
and use the explicit erasure mechanism with the exact recorded source only when the user requests
forgetting data.
