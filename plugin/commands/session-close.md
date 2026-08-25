---
description: Write down what this session learned that the repository does not already record, in the format the trust layer reads, and make it searchable before the context is discarded.
argument-hint: [optional: a specific thing to record]
---

# Close a session into project memory

The `SessionEnd` and `PreCompact` hooks index whatever is in the memory store. **Nothing makes
anything be there.** That is this command's job, and it is the half of the loop that decides
whether memory compounds or decays.

Run it before ending a session, before a long compaction, and after anything that cost real time
to work out.

## 1. Decide what is actually worth keeping

One question separates a memo worth writing from noise:

> Could a competent engineer with repository access derive this by reading the code and running
> the commands?

If **yes**, do not write it. The repository already records it, and a memo that duplicates the
code is a memo that will contradict it later.

If **no**, it is a candidate. In practice that means:

- A decision and the **reason** behind it, especially a considered option that was rejected.
- A hazard that cost something, with **what it cost**. The cost is what lets the memo be retired.
- A measured result, with the **date** it was measured and the **command** that re-measures it. A
  number with neither is folklore that will outlive its subject.
- A constraint from outside the repository: an environment, a service, a person's requirement.

⚠️ **A negative claim about a codebase has a shorter half-life than a positive one**. "X is read
nowhere" is exactly the kind of fact somebody then fixes. If you write one, date it and cite a
symbol rather than a bare absence.

If nothing clears that bar, say so and stop. **A session that learned nothing durable is normal**,
and writing a memo to have written one poisons the corpus for every later search.

## 2. Check whether it retracts something already stored

Before writing, search for what you are about to claim, in the vocabulary of the claim itself.

If an existing memo says something this session proved wrong:

⛔ **Do not edit it, and do not delete it.** Write a **new** file that names the old one:

```markdown
---
name: <slug-of-the-new-fact>
description: <one line, used to judge relevance at query time>
valid_from: <today, YYYY-MM-DD>
supersedes: <old-file>.md
---
```

That is what lets the store serve the correction *and* still explain a decision taken under the old
fact. Overwriting destroys the only record that the old fact was ever believed, which is usually
the more interesting half.

## 3. Write one file per fact

One fact, one file, under the project's memory directory. Never a digest of several facts: a
retrieval returns files, so a memo carrying four facts is three irrelevant ones attached to every
hit that finds it.

```markdown
---
name: <short-kebab-case-slug>
description: <one line, used to judge relevance at query time>
valid_from: <today, YYYY-MM-DD>
metadata:
  type: user | feedback | project | reference
---

<the fact, and what it cost to learn>

Related: [[another-memo-name]]
```

Three things that decide whether this memo is ever found again:

- **`valid_from` is the day you learned the fact.** A malformed date fails the index rather than
  passing silently, which is deliberate.
- **Leave `valid_until` out** unless you know a real end date. An absent value reads as unknown; a
  guessed one reads as judged, and the guess is worse.
- **Title it in the vocabulary of the failure, not of the task.** Whoever needs this memo will be
  searching for the operation that went wrong, not for the goal you had. "A file edited by a python
  script shows as modified with no content change" is findable. "Version bump conventions" is not.

Link liberally with `[[name]]`. A link to a memo that does not exist yet is not an error; it marks
a gap worth filling.

## 4. Index it, and add the index line

Add one line per new memo to the memory index file (`- [Title](file.md), one-line hook`), then
call `recall_index` on the memory directory so the new file **and** the updated index both become
searchable.

Do not skip this on the assumption that the hook will handle it. `SessionEnd` runs asynchronously
and cannot block termination, so it is a best effort rather than a promise, and a memo written at
10:00 in a session that ends at 17:00 is otherwise unsearchable for seven hours, including by the
turn immediately after the compaction that just discarded its context.

## 5. Report

Tell the user, briefly: what was written, what was superseded and by what, what was considered and
deliberately not written, and anything you could not index. If `recall_index` failed, say so with
its message rather than reporting the session closed. An unindexed memo is invisible to the search
the next session runs, so it gets rediscovered rather than recalled.
