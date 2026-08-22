---
description: Search this project's own memory before doing something it may already have opinions about. Use before running a build/test/deploy command, before choosing a library or approach, before changing configuration or CI, and before re-opening a decision that may already have been settled. Also use when a task resembles something the project has plausibly hit before.
---

# Check memory before acting

This project has a RE-call memory store: notes, decisions and postmortems written by earlier
sessions, searchable through the `recall_search` tool. It answers with a verdict per hit and
refuses when nothing clears its threshold, so a confident answer from it means something.

## Why this skill exists, and what it is correcting

Measured over 54 paired sessions, this memory layer eliminated a class of known hazard when the
relevant memo reached the agent, and reached it in only about a third of sessions. The layer was
not the problem. **The queries were.** The agent searched for what it was doing rather than for
what could go wrong, and a memo written about the failure did not match a query written about the
task.

One recorded example, in full, because the shape is the whole lesson:

| The agent searched for | The memo that would have saved it was written as |
|---|---|
| "version bump script versioning conventions" | "why does a file edited by a python script show as modified with no content change" |

Both are about the same file. Neither query retrieves the memo. Every session then wrote CRLF into
a tree configured for LF, which is exactly what the memo warns about.

⛔ **You cannot look up a hazard you do not know exists.** That is not a reason to skip the search.
It is the reason to search for the *hazard* rather than the *task*.

## How to search

Before acting, ask what could go wrong with the thing you are about to do, and search for **that**,
in the vocabulary of a symptom rather than of an intention.

- Not "how do I run the test suite here" but "test suite hangs", "tests pass locally fail in CI",
  "database tests silently skipped".
- Not "add a dependency" but "dependency breaks the build", "lockfile conflict", "version pin".
- Not "deploy the service" but "deploy failed", "migration locked the table", "rollback".

Two or three short searches beat one long one. Symptom words, error text and file names retrieve
well; a sentence describing your plan does not.

## When to do this

- Before a command that changes state: a build, a migration, a deploy, a release, a force push.
- Before choosing a library, a pattern or an approach, where the project may have tried one and
  rejected it.
- Before editing configuration, CI, or anything under a path you have not touched this session.
- When something looks like a decision that may already have been made. Re-opening a settled
  question is the failure this store was built to prevent.

## Reading the answer

`recall_search` returns a verdict per hit rather than a ranked list to interpret.

- **`ok`**: a live claim. Use it.
- **`superseded`**: a claim that was retracted, carrying a pointer to what replaced it. Follow the
  pointer. **Do not act on a superseded hit**, and do not treat it as a disagreement to resolve
  yourself: the corpus is telling you the newer document won.
- **`low_confidence`** and an **abstention**: the store is saying it does not know. Treat that as
  no answer, not as a weak yes. It is not evidence that no memo exists, only that this query did
  not find one, so try the hazard's vocabulary before concluding the corpus is silent.
- **`DEGRADED:INDEX_NOT_READY`** on every result means the corpus has no calibration fitted to it,
  so its threshold is a placeholder rather than a measurement. Weigh its verdicts accordingly and
  mention it if a decision turns on one.

## What not to do

- Do not paste secrets, credentials or customer data into a search.
- Do not treat memory as authoritative over the code in front of you. A memo records what was true
  when it was written; the repository records what is true now. **When they disagree, the code
  wins and the memo is worth correcting.**
- Do not search once, find nothing, and conclude the project has no opinion. That is the exact
  failure measured above.
