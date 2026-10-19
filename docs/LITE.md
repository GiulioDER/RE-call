# RE-call Lite: install and maintain

Lite keeps one project's memory in a local SQLite file. It uses the shared trust layer, local
embeddings, automatic memo refresh, and automatic calibration. Choose PostgreSQL when you need
immutable generations, multi-tenant serving, authentication, or a database shared across machines.

## Install a build that contains Lite

Lite is available in the repository source. The published `0.14.0` PyPI wheel predates Lite;
installing or upgrading that wheel alone will not add `recall setup --lite`. Until a release includes
Lite, use the source path below. It requires Python and Git, with no Docker or PostgreSQL server.

Release status checked on 2026-10-19 against [PyPI package metadata](https://pypi.org/pypi/recall-rag/json)
and the published wheel, which contains no `recall/lite/` package. To recheck, download the current
wheel with `python -m pip download --no-deps recall-rag` and inspect it with
`python -m zipfile -l <wheel-file>`; check `recall setup --help` after installing a newer release.

Create a dedicated environment and install the package from a clone:

```bash
git clone https://github.com/GiulioDER/RE-call.git re-call
cd re-call
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install ".[fastembed,mcp]"
```

On macOS or Linux:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install ".[fastembed,mcp]"
```

Keep this environment at its installed path: setup registers its Python interpreter with the MCP
client. The first local embedding run downloads model files; subsequent runs reuse the model cache.

## Set up your project

With that environment active, change to the project whose memory you want to use. Create a
`notes` folder or choose an existing folder of Markdown memos, then run:

```bash
recall setup --help
recall setup --lite --memory ./notes
```

The help should list `--lite`. If it does not, this shell is using an older installation. Activate
the environment above before continuing.

Setup indexes the chosen folder into `.recall/memory.db`, attempts calibration, and registers
`recall-memory` with Claude Code for this project. It prints the store path, calibration result,
registration result, and configured doctor and dashboard commands. Check those results and restart
Claude Code in the same project. A registration conflict is reported and left for you to resolve.

Add `.recall/` to the project's `.gitignore`: the database contains memo text and vectors, and its
SQLite sidecars and refresh state belong outside version control too. Keep your original notes in
their own backup or version control policy.

Setup leaves strict trust enabled. A small corpus can index successfully while remaining
uncertified. Add substantive memos and refresh; inspect the reported calibration reason rather than
turning off trust to make a search answer. Automatic calibration does not guarantee that every
corpus can certify. See [CALIBRATION.md](CALIBRATION.md) for its coverage and limits.

For a custom database location, use `--store /absolute/path/memory.db` on setup and refresh. Keep
the file on a local disk. Multiple clients on the same machine can share it, with writes queued;
network shares and simultaneous writers on different machines are unsupported.

## Connect another client

`recall setup --lite` registers Claude Code only. It does not install Codex hooks or the Claude
plugin. For CLI use or manual client wiring, run it with `--no-register`.

For Codex, Claude Desktop, or another stdio MCP client, configure the interpreter from the Lite
environment to launch `-m recall_mcp.server`. Use the absolute paths and environment printed by
setup. The essential settings are:

| Setting | Value |
|---|---|
| `RECALL_SERVING_DSN` | The absolute `sqlite:///` DSN printed by setup |
| `RECALL_INDEX_ROOT` | The absolute folder of memos |
| `RECALL_EMBEDDER` | The same embedder used at setup, `fastembed` in this guide |
| `RECALL_TENANT` | `default` |
| `RECALL_INDEX_MODE` | `legacy` |
| `RECALL_ENV` | `development` |
| `RECALL_TRUST_MODE` | `strict` |
| `RECALL_TRANSPORT` | `stdio` |

Here `RECALL_ENV=development` selects the local operating environment; the separate
`RECALL_TRUST_MODE=strict` still enforces certification. A POSIX absolute path has a DSN such as
`sqlite:////home/me/project/.recall/memory.db`; a Windows path uses
`sqlite:///C:/projects/myapp/.recall/memory.db`. Use setup's OS environment passthrough when a
client replaces its child process environment. See [the MCP guide](USING_WITH_CLAUDE.md) for client
configuration locations and [the Codex guide](CODEX_RECALL_INTEGRATION.md) for the distinct
PostgreSQL installer and hooks.

## Keep memory current

Edit or add memos in the configured folder. With `RECALL_INDEX_ROOT` configured, the MCP server
refreshes on startup and checks for changes before searches. It re-indexes changed content and
refits calibration when needed. There is no separate scheduler to install for this path.

For an explicit check or refresh, run these from your project with the same environment active:

```bash
recall lite refresh --store .recall/memory.db --memory ./notes --check
recall lite refresh --store .recall/memory.db --memory ./notes
```

Keep `--memory` explicit, especially before the first MCP refresh: setup writes the memo root into
client configuration, not into your shell. Use the same root and embedder as setup. Once a refresh
has recorded its roots, the command can reuse them, but explicit paths make scheduled jobs easier
to inspect.

With `--check`, exit code `0` means current and `1` means changed, never refreshed, or missing input.
A writing refresh returns `0` for refreshed, unchanged, or coalesced work, `1` for failure, and `3`
when the write queue is busy. A successful refresh is not itself proof of certification; read its
calibration result.

Automatic refresh enforces the indexing file and byte budgets. If it cannot refresh, search
responses carry `refresh_warning`, `recall_stats` reports the last outcome, and doctor reports the
problem. A manual refresh warns and proceeds over budget, so inspect the chosen folder before
running it. A large unexpected deletion is guarded; restore the missing folder before considering
`--allow-prune`. The switches and queue limits are listed in [ENVIRONMENT.md](ENVIRONMENT.md).

Use the doctor and read-only dashboard commands printed by setup. A plain `recall doctor` in a new
shell can inspect PostgreSQL instead, because the CLI does not read the MCP client's environment.
Apply setup's printed environment to CLI searches as well:

```bash
recall search "what did I decide about caching?"
```

## Update the software

Back up first, stop every MCP client and dashboard using this environment, and activate the same
environment used for installation. From the clean source clone:

```bash
git pull --ff-only
python -m pip install --upgrade ".[fastembed,mcp]"
```

Read [CHANGELOG.md](../CHANGELOG.md) before the update. Source installs follow the repository's
current code; they do not imply that the same code is published on PyPI. Record the installed
commit with `git rev-parse HEAD` when you need a reproducible deployment.

Return to your project, run setup again with the same `--memory` and `--store` choices, inspect its
report, and restart clients. Reinstalling the package is necessary because this guide uses a regular
install rather than an editable one. Re-run the printed doctor command and the freshness check.
Keep the same embedder; to try a different model, build a separate store instead of reusing a
calibrated file with incompatible vectors.

## Back up and restore

Back up the original memo folder and the SQLite store. The store also contains fact ledger data
that re-indexing memos alone does not recreate.

For a simple offline backup, stop all clients, refresh jobs, and dashboards using the store, then
copy the entire `.recall/` directory and the memo folder to your backup destination. Include any
`-wal` and `-shm` sidecars that remain; do not copy only `memory.db` from a running store. Keep the
backup private, since it contains the indexed text. For an online database snapshot, use SQLite's
backup API rather than an ordinary live file copy.

Restore with all clients stopped. Restore the store and notes together, rerun setup with their
actual paths to check calibration and client registration, then run the printed doctor command and
an explicit refresh check before restarting clients. If the environment or project moved, the old
absolute interpreter, database, and memo paths in client configuration must be updated.

## Common problems

| Symptom | Next step |
|---|---|
| `--lite` is unrecognized | Activate the source installation's environment and inspect `recall setup --help`. |
| Doctor tries PostgreSQL | Apply the environment printed by Lite setup; client configuration is not shell configuration. |
| No tools after setup | Inspect registration results, restart the client in the configured project, and check its interpreter path. |
| Store is indexed but searches refuse | Inspect calibration status; add useful memos and refresh while keeping strict trust enabled. |
| Changed memo is missing | Check `refresh_warning` and run refresh with explicit store and memo paths. |
| Writer is busy | Let the current writer finish and retry; do not remove live lane files. |
| Need shared serving or generations | Use the [PostgreSQL installation](INSTALLATION.md#postgresql-database); copying the SQLite file is not a PostgreSQL migration. |
