# RE-call for OpenCode

An OpenCode 2 plugin (tested on 2.2.7) that gives the model RE-call's memory: RE-call's six assistant
tools under their own names, the other agent-usable tools a Lite store serves, and a memory block before
each turn with what RE-call found for it. It starts RE-call's own `recall-mcp` as a child process over
stdio and reimplements no retrieval.

The plain MCP setup on the [OpenCode integration page](https://re-call.dev/integrations/opencode.html)
needs none of this: it adds `recall-mcp` to `opencode.json` as an MCP server. Use the plugin when you
also want the per-turn memory block and write tools limited to the session you drive.

`port/` holds scripts that copy Claude Code commands and agents into an OpenCode config directory.

## Setup

1. **RE-call with its MCP server** in a venv of its own:

   ```bash
   python -m venv .venv-recall
   .venv-recall/Scripts/pip install "recall-rag[fastembed,mcp]"     # bin/pip on macOS and Linux
   ```

2. **A Lite store**, created once (the plugin does not run setup). RE-call certifies trusted answers only
   when the memo folder holds at least 20 distinct chunks; on an empty folder setup binds no calibration
   and every search fails with "no calibration artifact is bound to this tenant and generation". Run it in
   the store's own folder, since it writes a `.env` in its working directory:

   ```bash
   recall --embedder fastembed setup --lite --memory <memo folder> --store <store>/memory.db --no-register
   ```

3. **The plugin**, installed by OpenCode from this repository. Pin a full commit hash of `master`:

   ```bash
   opencode plugin add 'github:GiulioDER/RE-call#<commit>::path:integrations/opencode'
   ```

   OpenCode clones the repository, installs this folder's runtime dependencies (not its test tools)
   into its own cache, and adds the spec to the global `opencode.json`. The first install took 3 min 37 s
   and about 430 MB of cache (2026-11-20, OpenCode 2.2.7 on Windows); later starts reuse it. When no
   OpenCode background service is running, `plugin add` starts one, which loads the plugin at once: with
   a store in place its `recall-mcp` child, and the embedding model in it, then stays up as long as that
   service (seen on Linux, 2026-11-20).

4. **The plugin entry** in `opencode.json`. With no options the plugin uses the defaults below. To pass
   options, replace the string entry `plugin add` wrote with an object carrying the same spec:

   ```json
   {
     "plugins": [
       {
         "package": "github:GiulioDER/RE-call#<commit>::path:integrations/opencode",
         "options": {
           "recall_command": "<.venv-recall>/Scripts/recall-mcp.exe",
           "dsn": "sqlite:///<store>/memory.db",
           "memo_folder": "<memo folder>"
         }
       }
     ]
   }
   ```

   From a clone instead: run `bun install --production` in `integrations/opencode/` and give
   `"package"` the path `<clone>/integrations/opencode/src`. OpenCode 2.2.7 loads a configured local
   plugin only from a directory.

   | Option | Default | Meaning |
   |---|---|---|
   | `recall_command` | `recall-mcp` | the server, an absolute path or a name found on `PATH`'s absolute entries (never the working directory) |
   | `dsn` | `sqlite:///<shared dir>/memory.db` | the store; a relative path resolves against the data dir |
   | `memo_folder` | `<shared dir>/memos` | the memo root; the model's own memos go in its `opencode/` subfolder |
   | `k` | `5` | memories per prefetch, clamped to 1..20 |
   | `prefetch` | `true` | the per-turn block |
   | `write_tools` | `true` | the write tools, on a Lite store only |
   | `extra_tools` | `true` | the tools beyond the six |

   The shared dir is `<XDG_CONFIG_HOME or ~/.config>/re-call`, one per user and the same for every agent;
   XDG_CONFIG_HOME counts only when it is an absolute path. With no `dsn`, the dsn and memo defaults point
   there, and with no store anywhere the plugin logs the setup command that creates one.

   The data dir is `<XDG_CONFIG_HOME or ~/.config>/opencode/re-call`, OpenCode's own. It is the server's
   working directory, and the server's stderr goes to `server.log` there, so two agents never write one
   log. At a start, a `server.log` of 8 MiB or more becomes `server.log.1` (replacing an older one) and a
   new log begins.

   A store made by an earlier version of this plugin sits in the data dir. With no `dsn` and no shared
   store, the plugin keeps using it and logs one line naming both paths. To move it, close every agent,
   then move `memory.db`, `memory.db-wal`, `memory.db-shm` and the `memos` folder from the data dir to the
   shared dir.

5. **A model whose context OpenCode knows.** OpenCode 2.2.7 takes an Ollama model's context length from
   its native length in `/api/show`, not the Modelfile's `num_ctx`, so a model served with a smaller
   window needs `limit.context` set to it in its provider entry.

With no store at the configured path the plugin registers nothing and logs one line with the setup
command; OpenCode then runs as if it were absent. Every other expected failure (an unknown option or a
wrong type, a `recall_command` that is not there, a server that does not start) is logged the same way,
as a line starting `re-call: inactive:`, and never makes the plugin's setup throw: OpenCode 2.2.7 skips a
plugin whose setup throws and runs on, so a throw would look like a session without memory.

## What leaves the machine

The plugin makes no network request. It starts `recall-mcp` as a child over stdio with only the
`RECALL_*` settings and the OS, cache and proxy variables (`src/child.ts`), so no provider key reaches
it. With the fastembed embedder RE-call embeds locally; fastembed downloads its model on first use.

## Tools

All are registered with `codemode: false`, so the model sees them by name beside OpenCode's own tools,
and each call forwards to the server tool of the same name. The model receives the server's answer as
Python's `json.dumps` writes it; `null` and `""` arguments are dropped, and `recall_search` gets `k` when
the model gives none.

- **The six assistant tools**, with the descriptions and schemas RE-call pins in
  `tests/fixtures/assistant_surface_contract.json`: `recall_search`, `recall_evidence`,
  `recall_current_facts`, `recall_apply_fact`, `recall_remember`, `recall_forget_memo`.
- **The extras**, with the server's own definitions from `tools/list` (reasons in `src/extra_tools.json`):
  `recall_related`, `recall_records`, `recall_current_state`, `recall_resolve_reference`,
  `recall_query_construction_challenge`, `recall_report_stale`, `recall_report_use`, `recall_withdraw`.
- **Write tools** (`recall_apply_fact`, `recall_remember`, `recall_forget_memo`, `recall_report_stale`,
  `recall_report_use`, `recall_withdraw`) are offered on a Lite store only, and only to the run the user
  drives: a root session (one with no parent) whose agent's mode is `primary` or `all`. The plugin removes
  them from every other request, including a subagent's session and a `subagent`/`subtask` command's.
- Operator tools (index, ingest, calibration, statistics, reasoning inspection, `recall_forget`) are never
  offered.

## Prefetch

- With `prefetch` on, the system prompt gets RE-call's sentence: "Long-term memory is RE-call. Before
  each turn it lists memories that passed a calibrated trust check, flags superseded ones, and says when
  it abstained."
- On each non-trivial user turn the plugin calls `recall_search` with the turn and `k`, renders the
  answer, wraps it in a `<memory-context>` block, and appends it to the user message after a blank line.
  It does this in OpenCode's `prompt` hook, so the block is part of the stored user message and later
  requests replay it.
- A trivial turn (greetings, acknowledgements, slash commands) gets no search and no block.
- A search that fails or takes longer than 6 s gives the block RE-call's "could not answer" line, and the
  turn goes on.
- The title request OpenCode sends for an untitled session gets the user text with the block removed.
- The block's sanitizer repeats until nothing changes, so a memo holding a nested fence tag
  (`</memory-<memory-context>context>`) cannot close the block early.

## Porting a Claude Code setup

```bash
bun port/install.ts --real-home [--recall-command <path to recall-mcp>]
```

This writes into `<XDG_CONFIG_HOME or ~/.config>/opencode`: each `~/.claude/commands/*.md` with Claude
tool names mapped to OpenCode's, each `~/.claude/agents/*.md` as a subagent whose `tools:` list becomes
permission rules, and an `opencode.json` naming `~/.claude/CLAUDE.md`, `CLAUDE.md` and `CLAUDE.local.md`
as instructions and this plugin as its memory. A step that needs a Claude-only feature (parallel
subagent calls, `TodoWrite`, `subagent_type`, `CLAUDE_*` variables) is kept and listed as unportable. It
refuses to overwrite: an existing `opencode.json`, `opencode.jsonc`, `command(s)/`, `agent(s)/` or
`mode(s)/` there stops it before anything is written. Skills need nothing, since OpenCode reads
`~/.claude/skills` and `~/.agents/skills` itself.

## Tests

```bash
bun install --frozen-lockfile   # includes the OpenCode CLI 2.2.7 the port tests run
bun run typecheck
bun run test:all                # unit, port and contract; `bun run` puts the pinned CLI on PATH
bun run mutations               # red proof: every behaviour test fails against a mutation of its line
```

On Windows, rows PLUGIN-DISPOSE and PLUGIN-SHARED-RELEASE are reported as survivors although their test
fails at the intended assertion: the child the mutation leaks keeps the test scratch folder open, so the
preload's cleanup also fails (`EBUSY`), and the runner counts any failure outside the row's assertion
as no proof.

- **Unit:** the MCP client and child process against `tests/fake_mcp_server.py`, a scripted stdio server;
  the renderer against fixtures in `tests/fixtures/render`; the tool surface against RE-call's pinned
  contract; the config.
- **Contract:** the plugin in a real in-process OpenCode 2.2.7 host, a scripted model from `harness/`, and
  the fake server: the tools in the request, the system sentence, the block byte for byte and in later
  requests, trivial turns, the model's arguments reaching the server, every expected setup failure, a dead
  server, the 6 s deadline, dispose, the title request, concurrent sessions, and subagents without write
  tools.
- **Port:** each script against the fixture Claude home in `tests/fixtures/claude`, and OpenCode loading
  what it writes, in an isolated home outside the real one.
- **Live surface** (opt in): the tool list of a real `recall-mcp` on a throwaway Lite store, started
  exactly as the plugin starts it. The hashing embedder loads no model:

  ```bash
  RECALL_OPENCODE_SURFACE=1 RECALL_OPENCODE_EMBEDDER=hashing RECALL_OPENCODE_BIN=<dir holding recall and recall-mcp> \
    bun test tests/contract/surface_live.test.ts
  ```

No test reaches a model or a credentialed service: `tests/preload.ts` strips every provider key and
gives the test process its own OpenCode data, state, cache and config folders.
