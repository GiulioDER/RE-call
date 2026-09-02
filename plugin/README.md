# RE-call for Claude Code

Memory that abstains instead of guessing. This plugin gives Claude Code a searchable store of your
project's own notes and decisions, where a retracted claim comes back marked `superseded` and a
question the corpus cannot answer is refused rather than answered from the nearest neighbour.

## Install

**Install the package first.** This plugin is a client, not an engine: its MCP server and its four
hooks are the console scripts `recall-mcp` and `recall-hooks`, invoked by bare name. A plugin
manifest is written once and shipped to every machine, so it cannot name your interpreter, and
without those scripts on `PATH` the server simply fails to spawn. Claude Code reports that as
absent tools, which is also what a missing config, an unreachable database and pending migrations
look like.

```bash
pip install "recall-rag[fastembed]"
recall quickstart
```

Then, inside Claude Code:

```
/plugin marketplace add GiulioDER/RE-call
/plugin install recall@re-call
```

If the tools do not appear, do not guess which of the five causes it is:

```bash
recall doctor
```

It checks the scripts on `PATH`, the database, the schema, whether the table and tenant you gave
the plugin actually hold anything, the calibration and the registration, prints the command that
repairs whatever it found, and writes nothing.

Claude Code then asks for four things: the **DSN** of the database to read, the **table** and the
**tenant** inside it, and the **trust mode**. The DSN is stored in your OS keychain rather than in
`settings.json`, because it carries a password.

**All four have to match whichever command built the store**, and the exact values are
printed at the end of that command. Get the table or the tenant wrong and the server starts
cleanly, answers, and finds nothing: there is no error to read, because an empty answer from the
wrong table looks exactly like an empty corpus.

## What the database has to look like

`recall quickstart` above starts a throwaway PostgreSQL, indexes a sample corpus, answers three
questions and prints the four values to paste back, verbatim. Trust mode is `development` there
because a sample corpus has no calibration fitted to it and a strict server correctly refuses to
answer from one.

For your own notes, run `recall setup` instead. It indexes what you point it at, fits a threshold
to it, and writes to table `chunks` and tenant `default`, which is the trust mode `strict` case.

| | `recall quickstart` | `recall setup` |
|---|---|---|
| Table | `quickstart_chunks` | `chunks` |
| Tenant | `quickstart` | `default` |
| Trust mode | `development` | `strict` |
| Corpus | 22 sample documents | yours |
| Calibrated | no | yes, if you accept the prompt |

The quickstart uses a table of its own so that 22 documents of fiction about a fictional service
can never be retrieved beside your real memory from the same database.

## What the plugin adds

**An MCP server** exposing `recall_search` and the rest of the tool surface, so Claude can query
memory as a tool.

**Five hooks**, which are no-ops until `recall setup` has run, and fail open in every case:

| Event | What it does |
|---|---|
| `SessionStart` | Injects a short digest of project memory before the first turn |
| `UserPromptSubmit` | Searches the project's memo files with your prompt and names prior records that bear on it |
| `PreToolUse` | Searches memory with the text Claude is about to write, on every write |
| `PreCompact` | Saves memory before a compaction discards the detail behind it |
| `SessionEnd` | Indexes the session so the next one can find it |

The two retrieval hooks answer different questions and are separately switchable in
`~/.claude/recall-hook.json` (`write_time.enabled`, `prompt_time.enabled`). `PreToolUse` uses the
draft text, which is what reaches a hazard; `UserPromptSubmit` uses your words, which is what
reaches a decision the project already made.

**A skill**, `check-memory-before-acting`, which teaches Claude *when* to search and, more
importantly, *how*. That second half is not decoration: measured over 54 paired sessions, the
memory layer eliminated a class of known hazard when the relevant memo reached the agent, and
reached it in only about a third of sessions. The layer was not the problem; the queries were.
Agents search for the task they are doing rather than the failure they are about to cause, and a
memo written about the failure does not match. The skill exists to correct that, and its
instruction has now been measured on the same benchmark
([record](https://github.com/GiulioDER/RE-call/blob/master/docs/preregistrations/2026-08-22-hazard-query-instruction.md)):
every session searched instead of half, and the governing memo reached the agent in 0.674 of
sessions against 0.319 (p = 0.0006), at roughly double the token overhead. What it did not fix is
query vocabulary itself, which is retrieval-side work, and the record says so.

## What it measurably does, and honest limits

The measured claim first: **adding this memory to a good `CLAUDE.md` reliably stops a known
hazard being repeated.** Over 40 pairs, additive memory gave a hazard rate of 0.000 against
0.525 and roughly doubled answer correctness
([record](https://github.com/GiulioDER/RE-call/blob/master/docs/preregistrations/2026-08-21-claude-md-plus-recall-additive.md)).
That is the reason to install it. The limits:

- **Replacing `CLAUDE.md` with memory does not work.** The measured win above is additive only;
  substituting memory for the hand-written file gave neither effect.
- **It does not make ordinary work come out better.** A second measurement, on tasks scored by
  running the agent's output against an oracle rather than by a judge, found **no significant
  uplift**: 54 admitted pairs over 8 tasks, delta +0.154 with the cluster interval crossing zero,
  sign test p=1.0. It also costs about 56,000 extra input tokens and 25 seconds per session.
  ([record](https://github.com/GiulioDER/RE-call/blob/master/docs/preregistrations/2026-08-21-task-success-executable-endpoint.md))
- **It needs PostgreSQL with pgvector.** There is no embedded mode.
- **`recall quickstart`'s corpus is uncalibrated**, so every result carries
  `DEGRADED:INDEX_NOT_READY`. That is the store saying its threshold was never fitted to this
  corpus, not an error. `recall setup` fits a real one.

Those results are not in tension, and together they are the honest pitch: **this layer reliably
stops a known hazard being repeated, and does not reliably make ordinary work come out
better.** Both are published, including the one that did not work.

## Links

- [Setup guide](https://giulioder.github.io/RE-call/)
- [Repository](https://github.com/GiulioDER/RE-call)
- [Using RE-call with Claude](https://github.com/GiulioDER/RE-call/blob/master/docs/USING_WITH_CLAUDE.md)
