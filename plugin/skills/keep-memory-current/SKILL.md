---
description: Load this project's memory when you start, and write back what you learned before you stop. Use at the beginning of a work session, when finishing a task or about to end a session, after debugging something non-obvious, after a decision that took argument to reach, and whenever you catch yourself thinking "I should remember this". Also use when a memory turns out to be wrong.
---

# Keep memory current

A memory store is only worth what the next session can find in it. Two moments decide that, and
they are the two this skill is about: what you load when you start, and what you write when you
stop.

The second one decays first and costs the most. A session that solves something hard and ends
without recording it has not saved anyone the hour; it has spent the hour on that project's behalf
and thrown away the receipt.

## When you start

**Read the hub, then search before you act.** The hub is the index file the project loads into
context (commonly `MEMORY.md`); it carries current state and pointers, not the content. Reading it
tells you what exists. It does not tell you what applies.

⛔ **An index line is a pointer, not the memo.** Being able to quote the one-line hook is not
having read the entry, and compression drops SCOPE first — which is the part that decides whether
the memo applies to what you are about to do.

Before acting on anything the project may already have an opinion about, search. The companion
skill `check-memory-before-acting` covers how to phrase that search, and the short version is:
search for the **hazard**, not the task.

## When you stop

### One fact per memo, with the cost that bought it

A memo that records only the conclusion cannot be retired later, because nobody can tell whether
the reason still holds. Write:

- **the fact**, stated so it can be checked
- **why** it is true, in enough detail to argue with
- **how to apply it**, concretely
- **what it cost to learn** — the hour, the outage, the wrong release

The cost is not decoration. It is what lets a future reader judge how hard to defend the memo when
new evidence arrives, and it is what makes a stale memo removable instead of load-bearing.

### Link liberally, including to things that do not exist

A `[[name]]` pointing at an unwritten memo is not a broken link. It marks a gap worth filling, and
it is the cheapest form of a to-do the store supports.

### Delete what turns out to be wrong

**A stale memory is worse than an absent one**, because it suppresses the retry that would have
corrected it. When you find one, correct it in place and say that it was corrected — the fact that
this particular thing rots is usually more useful than the fixed value.

Prefer correcting to deleting when the error is informative. "This was true until X changed" tells
the next reader which direction to distrust.

### Re-index, or the memo is invisible

**An unindexed memo is not in memory.** It is a file. The next session's search will not return it,
so the thing you just learned will be rediscovered rather than recalled — at full price.

If the project has a searchable memory corpus, re-index it now, as part of stopping, not as a
separate chore you will get to.

### Verify with a search, never with a row count

A row count says something was written. Only a search says the next session will find it.

Search for a distinctive phrase from a memo you just wrote. If it comes back, the loop is closed.
If it does not, you have files on disk and nothing in memory, which looks identical from the
outside and is the failure this whole step exists to prevent.

## Three failure modes worth knowing before you meet them

**Renaming a project directory can fork its memory store.** Where the store is keyed by the
project's path, a rename leaves the old store intact and invisible: no error, no warning, and the
index you load is the new empty one. Move the store at rename time. One store forked this way sat
unreachable for ten days while the same bug was worked from the other half.

**A flat index stops working long before it stops growing.** Past roughly fifty entries, move to a
hub with sub-indexes by type, and keep the top file for state and pointers only. A list nobody
finishes reading is a list that is not consulted.

**The count in the index will be wrong.** Any file that states how many memos exist will drift,
because concurrent sessions write memos without seeing each other. Do not trust it and do not
spend effort maintaining it. Assert **coverage** instead — that every memo is reachable from some
index and appears in exactly one — which is the property that actually matters and can be checked
in seconds.

## What good looks like

You should be able to answer these at the end of a session:

- What did I learn that was not obvious at the start?
- Is it written where the next session will search, rather than where I happened to be working?
- Did I check, by searching, that it comes back?
- Did I correct anything I found to be wrong on the way?

If the answer to the first is "nothing", that is a legitimate answer. Write nothing rather than
padding the store; a memo recording that an ordinary task went as expected costs future readers
attention and returns nothing.
