---
description: Ask this project's memory what it already knows about the work you are about to start, before the first file edit rather than after something breaks.
argument-hint: [what you are about to work on]
---

# Open a session against project memory

The `SessionStart` hook has already told you a corpus exists and how large it is. It could not
search it, because that event carries no user prompt and a similarity search with nothing to be
similar to returns whatever happens to be nearest. This command is the version that has a query.

Run it **before the first file edit or state-changing command**, not after something breaks. A
search after the failure can only explain it.

## 1. Establish what this session is about

If `$ARGUMENTS` is non-empty, that is the work. Otherwise infer it, cheaply, and say what you
inferred so the user can correct you in one line:

- `git status --short` and `git log --oneline -5` for what is in flight
- the branch name, which usually names the intent

Do not ask the user a question you can answer from the repository.

## 2. Decompose it into operations, then search for the operations

⛔ **Do not search for the goal.** This is the failure the whole corpus exists to prevent, and it
is measured: agents search for the task they are doing rather than for the failure they are about
to cause, and a memo written about the failure does not match a query written about the task. One
recorded pair, in full, because the shape is the lesson:

| The agent searched for | The memo that would have saved it was titled |
|---|---|
| "version bump script versioning conventions" | "why does a file edited by a python script show as modified with no content change" |

Both are about the same file. Neither query retrieves the memo.

So list the **operations** this work will actually perform (the tools, file types, commands and
services it touches), and search for each operation paired with what it does when it goes wrong.

Two or three short `recall_search` calls with different words, never one long one. Symptom words,
error text and file names retrieve; a sentence describing your plan does not.

- Not "add caching to the API" but "redis connection pool exhausted", "cache invalidation stale".
- Not "run the test suite" but "tests hang", "database tests silently skipped", "flaky in CI".
- Not "deploy" but "migration locked the table", "rollback", "deploy succeeded but health check
  failed".

## 3. Report what came back, and be exact about the verdicts

`recall_search` returns a verdict per hit rather than a ranked list you interpret:

- **`ok`**: a live claim. Say it plainly and cite the memo.
- **`superseded`**: a retraction. Follow the pointer and report the successor, never the
  retracted claim. This is not a disagreement for you to resolve: the corpus is saying the newer
  document won.
- **`low_confidence`**, or `abstained: true`: the store is saying it does not know. That is **not**
  evidence that no memo exists, only that this query did not find one. Reword once into the
  hazard's vocabulary before concluding the project has no opinion.
- **`DEGRADED:INDEX_NOT_READY`** on every result: the corpus has no calibration fitted to it, so
  its threshold is a placeholder rather than a measurement. Weigh the verdicts accordingly, and
  say so if a decision turns on one.

Then give the user a short brief: standing decisions that constrain this work, hazards with a
recorded cost, and anything that looks like a settled question they may be about to re-open.
**Re-opening a settled question is the failure this store was built to prevent.**

If every search abstained, say that in one line and start the work. An empty corpus is a normal
state, not a blocker, and this command must never become a gate that stops someone working.

## What this command does not do

It does not make memory authoritative over the code in front of you. A memo records what was true
when it was written and the repository records what is true now. **When they disagree the code
wins, and the memo is worth correcting**. Note it, and `/recall:session-close` will write the
correction as a new memo rather than an edit.
