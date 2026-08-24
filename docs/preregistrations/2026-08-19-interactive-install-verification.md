# Pre-registration: does the wizard's install reach a fresh interactive session?

**Date:** 2026-08-19   **Status:** predicted, not yet measured

## The question

After `recall setup` completes on a machine that has never had recall, does a **fresh interactive**
Claude Code session opened in that project list the `recall` MCP tools, and does the `SessionStart`
digest appear in its context? Yes or no, per session, over n sessions.

This is the only question the installer cannot answer about itself. Registration writes
configuration; whether the client acts on it is observable only from inside the next session.

## What I predict

**Primary: 3 of 3 fresh interactive sessions list the `recall` tools.** Confidence roughly 75%.

The 25% is not evenly spread. It is concentrated in one thing: whether local scope really is
outside the approval gate. That reading is documentary, derived from the gate being described as
covering "project-scoped servers from `.mcp.json` files" and from the approval keys being named
`enabledMcpjsonServers`. Nobody has watched a local-scope entry load. If it turns out local scope
is gated too, the expected result is 0 of 3, not a partial one, because every session would hit the
same unanswered prompt.

**Secondary: 3 of 3 sessions show the digest**, conditional on the primary passing and the corpus
being non-empty. The digest is emitted by a hook, not by the MCP server, so it does not depend on
the tools loading. If the tools load and the digest is absent, the hooks did not install or did not
fire, which is a different defect and the split tells them apart.

**Tertiary: 1 of 1 compaction indexes `memory/`.** Lower confidence, roughly 60%, and the
uncertainty is about triggering a compaction on demand at all rather than about the hook.

**Magnitudes worth stating because they are cheap to check and easy to wave through:**

- Session start overhead attributable to the hook: **under 300 ms**, against 66 ms measured here for
  the hook process in isolation. Anything above a second means the hook is doing work it was
  designed not to do, most likely resolving an embedder.
- The digest names a chunk count **greater than zero** on a seeded project, and the seeded count
  should equal what `recall setup` printed. A digest saying zero chunks after a successful seed
  means the cached count and the corpus disagree.

## What would falsify this

Any of the following, each of which falsifies a different claim:

- **Primary:** one or more fresh interactive sessions with no `mcp__recall__*` tool of any kind.
  A single failure falsifies "this works", because the claim is that installation is sufficient.
- **Secondary:** tools present, no `additionalContext` from the hook.
- **Tertiary:** a compaction occurs and `memory/` is not re-indexed afterwards.
- **The overhead claim:** measured session-start delay above one second attributable to the hook.

A partial result (some sessions yes, some no) falsifies the prediction more informatively than a
clean zero, because it implies a race or a per-session condition nobody has modelled.

## How it will be measured

**n = 3 fresh interactive sessions**, plus the two apparatus checks below. Metric, by name:
**fraction of fresh interactive sessions started in the registered project (denominator = 3
sessions started) whose own tool inventory contains at least one `mcp__recall__*` tool.**

### Preconditions, all of which must hold before session 1

1. A Windows account **without administrator rights**. This is not incidental: `initdb` refuses to
   run elevated, and a per-machine installer running as Administrator cannot create a cluster.
2. No prior recall state. Confirm all four:

   ```bash
   python -c "import json,os;d=json.load(open(os.path.expanduser('~/.claude.json')));p=d.get('projects',{});print('user scope:',list(d.get('mcpServers',{})));print('local entries:',[k for k,v in p.items() if 'mcpServers' in v])"
   ```

   plus `~/.claude/recall-hook.json` absent, no `hooks` block naming `recall_hooks` in
   `~/.claude/settings.json`, and no `.mcp.json` in the project.
3. The database reachable, and `recall setup` run to completion **with seeding accepted**, so the
   corpus is non-empty. Record the chunk count it prints.

### The runs

Each session is opened **interactively**, by launching Claude Code in the project directory the
way a user would. Not `claude -p`, not the SDK, not a cloud session. Then, in the session:

1. Ask it to list the MCP servers it can call tools on, names only.
2. Record whether any `mcp__recall__*` tool is present.
3. Record whether the session's context contains the digest line naming a chunk count.
4. Close the session. Do not resume it: a resume is a different condition and conflating the two is
   how the previous attempt at this question went wrong.

For the tertiary claim, in one additional session: fill the context until a compaction occurs, or
trigger one, then confirm `memory/` was re-indexed by checking that the cached count in
`~/.claude/recall-hook.json` changed after adding a memo.

### Apparatus checks, because predicting the outcome does not reveal a broken harness

**Check 1, the instrument rule, adopted from the correction in
`2026-08-16-sessionstart-hook-mcp-ordering.md`: count a session only if its inventory lists at
least one MCP tool of ANY kind.** A session that died before assembling an inventory has no
inventory, and scoring it as "no recall tools" is scoring a null as data. That mistake produced two
withdrawn rows in that document and is the single most likely way this run also produces nothing.

**Check 2, a known-answer case.** Before the three runs, open an interactive session in a project
with **no** recall registration and confirm the tools are absent. If they are present there, the
registration is not what put them in the treatment sessions and the whole measurement is void.

**Check 3, the authentication trap.** `claude -p` returns 401 on the machine this was designed on,
and it **exits 0**, so a harness reading exit codes scores that as success. This procedure avoids
`claude -p` entirely, which removes the trap rather than working around it, but if any step is
automated later, the exit code must not be the signal.

## What I already know

- **Approval is not necessary**, on one positive observation: a session in a never-approved project
  received a full recall tool set. Recorded in `2026-08-16-sessionstart-hook-mcp-ordering.md` and
  narrowed by the later correction there.
- **The companion claim that `resume` versus fresh `startup` separates the outcome was withdrawn**,
  because both supporting rows were sessions that died on a 401 with empty tool deltas. So there is
  currently **no measured result about fresh startups at all**, which is exactly the gap this
  document is for.
- **Headless, SDK and cloud sessions load project-scoped servers without prompting**; the refusal is
  interactive-only. This is why the measurement must be interactive: a headless run cannot
  reproduce the gate in either direction and would produce a confident irrelevance.
- **The client does not normalise project keys.** Measured: 313 keys, 7 directories carrying two
  spellings each. `_project_keys` writes every spelling that resolves to the project, so a wrong-key
  miss should not be the failure mode here, but it is the first thing to check if the tools are
  absent.
- **Hook cost in isolation: 66 ms** over a bare interpreter, against 1128 ms had the hooks lived in
  the `recall` package. This is the basis for the sub-300 ms prediction.
- **Local scope skipping the approval gate has never been observed**, only inferred.

## Confounds I can name now

1. **Session type.** The single largest one. An SDK or headless session loads project-scoped servers
   unasked, so running this any way other than interactively answers a different question. The
   procedure says interactive four times for this reason.
2. **A machine that is not actually clean.** A leftover local entry, a `recall-hook.json`, or a
   stale `.mcp.json` would make the install look successful when it changed nothing. Precondition 2
   exists to exclude this and should be re-checked, not assumed, because a previous run on the
   original machine was polluted by an artifact written hours earlier by a different process.
3. **An empty corpus.** The digest is deliberately silent when the cached count is zero, so a
   session with no digest and no seed is consistent with the hook working perfectly. Seeding must be
   accepted, and the count recorded, or the secondary claim is untestable.
4. **The database not running.** By design the digest still appears from the cached count, so this
   does not falsify the secondary claim, but the count will be stale. Worth recording rather than
   controlling.
5. **Workspace trust.** Since v2.1.196 a repository cannot approve its own servers until the
   workspace is trusted. Local scope should be outside this, but if the trust dialog appears and is
   dismissed, record it: that is a plausible mechanism for a partial result.
6. **The tertiary claim's trigger.** Forcing a compaction is not reliably controllable, so a null
   there is more likely to mean "no compaction happened" than "the hook failed". The check on the
   cached count distinguishes them.
7. **Sample size.** n = 3 detects a total failure and a coin-flip, and does not detect a 1-in-10
   condition. That is the accepted limit, not an oversight.

## Why this cannot be automated, stated so nobody tries

The measurement's whole content is what an interactive client does. Every automation route
available here is either the wrong session type (`claude -p`, SDK) or unable to observe the
inventory (the hook log records the treatment condition and never the outcome, which is why 18 of
19 historical treatment opportunities on the original machine are unrecoverable). A human opening a
session and reading what is in front of them is the instrument.

## Recorded preconditions (2026-08-19, before any treatment run)

**Status: still predicted, not yet measured.** Nothing above this line is edited. This section
records the state of the machine the procedure was written on, taken read-only by the wizard
session before anyone runs the check, because approval and registration are timestamped nowhere and
a snapshot taken afterwards proves nothing.

**The machine was not clean, and none of the three findings were expected.**

1. **A local-scope `recall` server was already registered**, and one of this branch's own tests put
   it there:

   ```
   projects carrying LOCAL scope servers: 1
     ...\pytest-of-gde00\pytest-12211\test_every_cli_call_is_made_fr0\proj -> ['recall']
   ```

   `test_every_cli_call_is_made_from_the_project_root` stopped taking the faked CLI arm when the
   primary path became the direct merge, fell through to `user_config_file()`, and wrote to the
   real home because it pinned no config directory. Removed, with a backup, and the module now
   carries an autouse fixture pinning `CLAUDE_CONFIG_DIR` for every test in it.

2. **Hook state was already installed**: `~/.claude/recall-hook.json` present, and a `recall_hooks`
   block already in `~/.claude/settings.json`.

3. **Three projects carried recorded `.mcp.json` approvals** for `recall` and `recall-memory`, from
   earlier sessions running `session_mcp_approve.py`.

### What this changes about the procedure

Nothing in the method, and everything about where it may be run. An interactive session opened on
this machine today could list recall tools for at least three reasons that have nothing to do with
`register_local_scope`, and a naive run would have scored one of them as a pass. Precondition 2 of
"How it will be measured" is therefore not a formality: it is the step that decides whether the
result means anything.

**The known-answer control needs a project absent from all three lists above**, not merely a
project nobody has registered by hand.

### One thing the snapshot cost, which is the argument for taking it

Finding 1 was a defect in the code under test's own test suite, discovered by a step whose stated
purpose was to characterise the environment. It would otherwise have been discovered as a passing
measurement.

## Scheduling constraint added 2026-08-19

**This check must not run until the mechanism it measures is the one that will ship.** Registration
currently exists twice: `recall.claude_code` on the branch this document is on, and
`recall.wizard.wiring.register_local_scope` on the wizard's, which is the one that survives. The
second replaces the first on merge. Measuring the temporary implementation and then deleting it
produces a result about code nobody runs.

Order: land the wizard's mechanism, swap this branch's registration to call it, then run the check
against the merged code. Agreed with the wizard session, which owns the mechanism.

### State after the cleanup (2026-08-19), so this is not read as a clean machine

Only finding 1 above was removed. Re-measured after the removal, and independently re-measured by
the wizard session, which took it as its user's configuration rather than accepting a report:

```
projects tracked                : 314   (was 315; exactly one key removed)
LOCAL scope servers             : 0     (was 1)
USER scope (top-level)          : 0     (unchanged)
recorded .mcp.json approvals    : 3     (unchanged, the same three)
~/.claude/recall-hook.json      : present
recall_hooks in settings.json   : present
```

**"No prior recall state" is still false on this machine**, and the past tense above should not be
read as saying otherwise. Findings 2 and 3 remain and are nobody's to remove: the hook state is a
working installation, and the three approvals belong to earlier sessions.

The consequence for precondition 2 is unchanged and worth restating in the present tense: the
known-answer control needs a project absent from **all** of the lists above, and this machine cannot
supply one without removing state that is legitimately in use. That is a reason to run the check
elsewhere, not a reason to relax the precondition.
