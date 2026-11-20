# re-call: RE-call memory for Hermes Agent

A Hermes memory provider backed by [RE-call](https://github.com/GiulioDER/RE-call), agent memory on a
local SQLite file. Before each turn Hermes receives the memories that passed RE-call's calibrated
trust check, with a verdict and a memo id for each. Memories that a later one superseded are
flagged as retired, and when nothing clears the threshold the block says that RE-call abstained,
so the model knows it should not answer about the present from memory.

```text
## RE-call memory
These memories passed RE-call's calibrated trust check for this message. Name the memo id when you rely on one.
- [ok] memo note-0002 (valid from 2026-10-01T00:00:00+00:00): My employer is Borealis Labs in Oslo.
Retired memories (superseded or expired): do not state these as current.
- [superseded, replaced by hermes/note-0002.md] memo note-0001 (valid from 2026-01-05T00:00:00+00:00): My employer is Acme Robotics in Turin.
```

That block is verbatim from the end-to-end test against a real RE-call 0.15.0 Lite store. When the
same store is asked something it holds nothing about, the block is the abstention line instead:

```text
## RE-call memory
RE-call abstained: nothing in memory passed its calibrated trust check for this message. Do not answer questions about the present from memory.
```

## What it sends where

- **Nothing leaves your machine** with the default settings. The provider starts RE-call's own MCP
  server (`recall-mcp`) as a child process and talks to it over a pipe. RE-call embeds with
  fastembed on your CPU; its first start downloads that model from HuggingFace.
- The child receives the `RECALL_*` settings it needs, a short list of operating system variables
  (`PATH`, `HOME`, temporary and cache folders, locale), the model cache variables, and the proxy
  and CA bundle variables so a first model download works behind a proxy. Hermes's own environment,
  including its provider API keys, is not passed to it. Its working directory is the provider's data
  folder, so it reads no project `.env`. The server command is looked up only in the absolute
  folders on `PATH`, never in the directory Hermes was started in.
- What is stored: the facts the model saves with `recall_remember`, and a copy of each entry
  Hermes's built-in memory tool adds, replaces or removes (turn this off with
  `mirror_builtin: false`). Conversation turns are **not** captured; RE-call keeps one fact per memo.
- Where: the memory itself (`memory.db` and the `memos/` folder) is in the store your agents
  share, `<XDG_CONFIG_HOME, else ~/.config>/re-call/` by default. What belongs to one Hermes
  profile stays in `$HERMES_HOME/re-call/`: `config.json`, `mirror.json`, `embeddings.sqlite` (the
  embedding cache of your memo text) and `server.log`. The one thing kept elsewhere is the embedding
  model itself, which fastembed caches where `recall setup` put it (RE-call's platform cache folder,
  or `FASTEMBED_CACHE_PATH`); it holds no memo text.
- No telemetry, no credentials, no shell commands. The one background process is the RE-call
  server, which stops when Hermes shuts the provider down.

## Install

1. Install RE-call 0.15.0 or later in its **own** environment (not Hermes's):

   ```bash
   pipx install "recall-rag[fastembed,mcp]>=0.15.0"
   ```

2. Create the shared store, once per user (with `XDG_CONFIG_HOME` set, use it in place of `~/.config`):

   ```bash
   mkdir -p ~/.config/re-call/memos
   recall --embedder fastembed setup --lite --memory ~/.config/re-call/memos --store ~/.config/re-call/memory.db --no-register
   ```

   RE-call calibrates the store on questions it generates from your memos. A very small store can
   stay uncalibrated; strict search then refuses, and the provider shows that reason in one line
   instead of recalling. Add some memos (Markdown files, one fact each) to the folder and the next
   start refits.

3. Install the provider and select it:

   ```bash
   hermes plugins install GiulioDER/RE-call/integrations/hermes
   hermes memory setup        # choose re-call
   ```

## Configuration

`hermes memory setup` writes `$HERMES_HOME/re-call/config.json`:

| Key | Default | Meaning |
|---|---|---|
| `recall_command` | `recall-mcp` | The server command, or an interpreter with arguments such as `/path/to/python -m recall_mcp.server` |
| `dsn` | the shared Lite store (see below) | A `sqlite:///` path, or a PostgreSQL DSN (search only; keep the password in `~/.pgpass`) |
| `memo_folder` | `memos/` beside the default store; `$HERMES_HOME/re-call/memos` when `dsn` names another store | The folder the store indexes; the provider's own memos go in its `hermes/` subfolder. Set with `dsn` blank, the store is `$HERMES_HOME/re-call/memory.db` |
| `k` | `5` | Memories retrieved per turn |
| `mirror_builtin` | `true` | Mirror Hermes's built-in memory writes into RE-call |

### One memory for every agent

The owner's decision of 2026-11-13 is "One shared Lite store": every agent that uses RE-call Lite
on this machine uses the same per-user store by default. With `dsn` and `memo_folder` blank, the
provider picks the store this way:

1. `<XDG_CONFIG_HOME, else ~/.config>/re-call/memory.db`, when it exists.
2. Otherwise this profile's old store, `$HERMES_HOME/re-call/memory.db`, when it exists. It is kept,
   not stranded, and the provider logs one line naming both paths.
3. Otherwise the shared path, and the provider is unavailable until you create the store there; its
   message gives the setup command.

**This changes what a profile means.** Before, each Hermes profile had its own memory. Now two
profiles share one memory unless one of them sets `dsn`. Each profile still has its own
`config.json`, mirror journal and log, and its mirrored memos go into the same `hermes/` subfolder
of the shared memo folder.

The shared store and `<shared>/memos`, the folder every agent's server indexes, always go
together. A profile with its own `memo_folder` and no `dsn` therefore keeps its own store,
`$HERMES_HOME/re-call/memory.db`. A `dsn` naming the shared store gets `<shared>/memos`, a
`memo_folder` naming `<shared>/memos` gets the shared store, and pairing either with anything else
is refused with a reason.

To move an old profile store into the shared directory, close every agent that uses either store,
then move `memory.db` with its `memory.db-wal` and `memory.db-shm` sidecars (when present) and the
`memos/` folder from `$HERMES_HOME/re-call/` to the shared directory. The profile's `mirror.json`
follows the moved store on the next start if its memo ids still name its entries' memos there:
none may hold other text, and at least one must be found. A mapped memo that cannot be read then
(not moved yet, or held open) is kept unconfirmed, and used only once its file is read holding the
entry's text. The same check runs when the store file at the journal's own path has changed (its
recorded volume and file number differ). A journal that
fails it stops mirroring and logs the reconciliation warning below. If the shared store already exists, the old one is not
read any more (the provider logs that once); set `dsn` to `sqlite:///<path to the old memory.db>` to
keep a profile on it.

## Tools

RE-call's six assistant tools, with RE-call's own names, descriptions and schemas:
`recall_search`, `recall_evidence`, `recall_current_facts`, `recall_apply_fact`, `recall_remember`
and `recall_forget_memo`.

The three write tools (`recall_apply_fact`, `recall_remember`, `recall_forget_memo`) are offered only
to the primary agent on a Lite store; subagents and cron runs get the three read tools. A PostgreSQL
store gets `recall_search` and `recall_evidence`, the tools its server registers.

The write tools honour Hermes's `memory.write_approval`. With it on, Hermes asks you inline before each
write, where someone can answer; a write you deny is not saved. Where nobody can answer (a single
query, cron, an unattended platform, a background review), Hermes would stage the write for later
review, but its pending store holds writes for its own memory only, so the provider refuses the write
and tells the model why. If the check itself fails, the write is refused while approval is on, or
while its state cannot be read. Mirrored writes are the built-in memory tool's own, so they have
already passed the same gate.

## Behaviour worth knowing

- Hermes waits at startup until the server is ready (the first start loads the embedding model), up
  to 120 seconds, so the first turn already has memory. Recall is then synchronous each turn, and a
  whole prefetch stays within 7 seconds, under Hermes's 8. Trivial prompts (`ok`, `thanks`, slash
  commands) skip recall.
- A crashed server is restarted on a later turn, at most once a minute; a turn that arrives while
  it restarts waits for it, up to 6 seconds, and gets no memory if it is not back by then.
- A built-in memory write taken while the server is not running was never sent, so it waits in
  `mirror.json` and is applied, in order, by the next mirrored write once the server is back, or
  when the next session starts.
- Built-in `replace` and `remove` are mirrored only when Hermes says which entry changed
  (`previous_content`); otherwise they are skipped rather than guessed. A replace becomes a new memo
  that supersedes the old one, so the old text stays in RE-call as a flagged, retired memory.
- Each target holds one memo per stripped entry text. Duplicate adds do nothing; removing the entry
  forgets its memo.
- Mirror state is bound to the store and assistant memo folder. A changed destination or an older
  journal without that identity stops mirroring and logs a reconciliation warning.
- A write whose outcome is unknown stays in `mirror.json` and blocks later mirrored writes. It is
  never automatically submitted again: RE-call's request IDs expire by age or count. Search and
  explicit tools still work. Before resuming mirroring, stop every Hermes instance on the profile,
  reconcile the journal with its original store, and preserve a backup. Only then archive `mirror.json`
  to start a new mapping; existing memories are not automatically reimported or deleted.

## Development

```bash
python -m pip install -e . -r requirements-dev.txt
python -m pytest                 # unit tests: fake Hermes, scripted server over a real pipe
python tests/mutations.py        # red proof: each mutation must fail its test's assertion
HERMES_SRC=/path/to/hermes-agent python -m pytest tests/test_contract.py
RECALL_HERMES_E2E=1 RECALL_E2E_BIN=/path/to/recall/bin python -m pytest tests/test_e2e_lite.py -s
```

The contract tests need a Hermes checkout with its dependencies importable (Hermes v0.23.0 pins them
for Python 3.14; `uv sync --frozen --no-install-project` in the checkout provides them). The
end-to-end test runs fastembed and creates a store in pytest's temporary directory.

This provider lives in the RE-call repository at `integrations/hermes/`, and runs its tests from
this folder. License: MIT, for this folder; the rest of the RE-call repository is AGPL-3.0 or the
commercial licence.
