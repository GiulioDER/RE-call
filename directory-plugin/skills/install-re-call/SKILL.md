---
name: install-re-call
description: "Walk the user through installing RE-call, a local persistent memory for Claude Code: the `recall-rag` Python package, a SQLite store built by `recall setup --lite`, and optionally the full RE-call plugin from its own marketplace. Use when the user agrees to add memory across sessions, asks how to install RE-call or recall-rag, or asks to set up persistent memory for a project."
---

# Install RE-call

The user installs; you guide. Show each command before it runs, say what it changes, and run it
only after the user says yes. A command typed into Claude Code itself (the `/plugin` lines) is the
user's to type.

## Tell the user what this installs and what reaches the network

- **The `recall-rag` package from PyPI**, into a Python environment the user chooses.
- **A store in the project**, `.recall/memory.db`, holding memo text and vectors.
- **For Claude Code**, setup registers a `recall-memory` MCP server for this project, adds hooks
  to `~/.claude/settings.json`, and copies five skills into `~/.claude/skills/`.
- **On first use**, the local embedder downloads the `BAAI/bge-small-en-v1.5` model from Hugging
  Face. No memo text is sent there.
- **No memo text leaves the machine** unless `RECALL_EMBEDDER` names a hosted embedder (Voyage,
  OpenAI), which then receives the text of every memo and search under the user's own key.

The full privacy statement is in the plugin README's
[Privacy section](https://github.com/GiulioDER/RE-call/blob/master/plugin/README.md#privacy-what-it-sends-and-where).

## 1. Install the package

RE-call needs Python 3.11.5 or later (`python --version`). Create a dedicated environment at a
path that will stay put, because setup registers this interpreter with Claude Code.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install "recall-rag[fastembed,mcp]>=0.16.0"
```

On macOS or Linux:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install "recall-rag[fastembed,mcp]>=0.16.0"
```

## 2. Build the project's store

With that environment active, from the project root, pick the folder of Markdown memos to index
(create `notes` if the project has none) and run:

```bash
recall setup --help
recall setup --lite --memory ./notes
```

The help must list `--lite`; if it does not, the shell is using an older installation, so activate
the environment from step 1. To keep embedding local whatever the environment says, name the
embedder before the command: `recall --embedder fastembed setup --lite --memory ./notes`.

Setup indexes the folder, attempts calibration and prints the store path, the calibration result,
the registration result and the doctor command. Read those lines back to the user. A small corpus
can index and stay uncertified; that is the store saying its threshold is not fitted yet, not an
error. Do not turn off strict trust to make a search answer.

Add `.recall/` to the project's `.gitignore`: the database holds memo text.

## 3. Restart Claude Code and check

Ask the user to restart Claude Code in the same project. The `recall_search` tool should then be
available. If it is not, run:

```bash
recall doctor
```

It checks the scripts on `PATH`, the store, the schema, the calibration and the registration,
prints the command that repairs whatever it found, and writes nothing. Do not guess the cause:
a missing registration, an unreachable store and pending migrations all look like absent tools.

## 4. Optional: the full RE-call plugin

Lite setup already gives Claude Code the server, the hooks and the skills, so the full plugin is
optional. Offer it when the user wants Claude Code's plugin manager to own and update them. It is
installed from RE-call's own marketplace, inside Claude Code:

```
/plugin marketplace add GiulioDER/RE-call
/plugin install recall@re-call
```

Installed before step 2, it changes what setup does: setup then configures the plugin's hooks
and skills instead of copying its own. When the plugin asks for its **Database DSN**, give it the
`sqlite:///` address setup printed, with table `chunks`, tenant `default` and trust mode `strict`.
The plugin's own
[README](https://github.com/GiulioDER/RE-call/blob/master/plugin/README.md) covers its settings.

## After installing

Memory is only worth what the next session can find. Once `recall_search` works, the RE-call
skills that setup installed (`re-call`, `check-memory-before-acting`, `keep-memory-current`) take
over: search before acting, and write one memo per durable fact before stopping. This setup
plugin has done its job and the user may uninstall it.
