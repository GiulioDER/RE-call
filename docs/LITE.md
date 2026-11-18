# RE-call Lite: install and maintain

Lite keeps one project's memory in a local SQLite file. It uses the shared trust layer, local
embeddings by default (fastembed), automatic memo refresh, and automatic calibration. Choose PostgreSQL when you need
immutable generations, multi-tenant serving, authentication, or a database shared across machines.

## Install a build that contains Lite

Lite ships in the PyPI wheel from `0.15.0`. Earlier wheels do not contain it, so upgrading an
older environment is enough only when it ends on `0.15.0` or later. It needs Python, with no
Docker, Git or PostgreSQL server.

Create a dedicated environment and install the package.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install "recall-rag[fastembed,mcp]>=0.15.0"
```

On macOS or Linux:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install "recall-rag[fastembed,mcp]>=0.15.0"
```

To run unreleased changes instead, install from a clone of the repository with the same extras:
`git clone https://github.com/GiulioDER/RE-call.git re-call`, then
`python -m pip install "./re-call[fastembed,mcp]"` in the environment above.

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

Setup embeds with `RECALL_EMBEDDER` when that variable is set, in the shell or in the project's
`.env`, and with fastembed otherwise. A hosted embedder (Voyage, OpenAI) sends the text of every
memo to that provider. To keep embedding local whatever the environment says, name the embedder
before the command: `recall --embedder fastembed setup --lite --memory ./notes`.

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

After memos change, Lite re-verifies the calibration it already has rather than refitting at once.
Up to a quarter of the sources changed since the fit, the carried calibration needs no refit;
up to three quarters, it keeps serving while it still re-verifies, with a refit due; past that,
strict search refuses until the store is refitted. A PostgreSQL store stops carrying at a quarter.

For a custom database location, use `--store /absolute/path/memory.db` on setup and refresh. Keep
the file on a local disk. Multiple clients on the same machine can share it, with writes queued;
network shares and simultaneous writers on different machines are unsupported.

## Connect another client

`recall setup --lite` registers Claude Code only. It does not install Codex hooks or the Claude
plugin. For CLI use or manual client wiring, run it with `--no-register`.

For Codex, Claude Desktop, or another stdio MCP client, configure the interpreter from the Lite
environment to launch `-m recall_mcp.server`. Setup prints the DSN and the memo folder but not the
interpreter; with the Lite environment active, `python -c "import sys; print(sys.executable)"`
prints its absolute path. The essential settings are:

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
`sqlite:///C:/projects/myapp/.recall/memory.db`. Also set `PYTHONSAFEPATH=1`, so a project that
holds its own `recall/` package cannot shadow the installed one. When a client replaces its child
process environment instead of extending it, pass through `APPDATA`, `LOCALAPPDATA`, `SystemRoot`,
`TEMP`, `TMP`, `USERPROFILE`, `HOMEDRIVE`, `HOMEPATH`, `HOME`, `XDG_CACHE_HOME` and
`XDG_DATA_HOME` with their current values; setup copies these into the server it registers itself,
but prints none of them. See [the MCP guide](USING_WITH_CLAUDE.md) for client
configuration locations and [the Codex guide](CODEX_RECALL_INTEGRATION.md) for the distinct
PostgreSQL installer and hooks.

## Keep memory current

Edit or add memos in the configured folder. With `RECALL_INDEX_ROOT` configured, the MCP server
refreshes on startup and checks for changes before searches. The Claude plugin's settings cannot
set `RECALL_INDEX_ROOT`, so unless that variable is already in the environment Claude Code starts
from, the plugin's server does not refresh by itself: with it, run the refresh below yourself or on
a schedule, or use the server setup registers. It re-indexes changed content and
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

The store records the embedder setup used (or, for a store `recall index` built, the one it used),
and `recall lite refresh`, `recall index`, `recall search`, `recall forget` and `recall withdraw`
against the store use it when you pass no `--embedder`. A `RECALL_EMBEDDER` in your shell that
names a different embedder is refused by refresh, index and search, because a variable left over
from another project would otherwise re-embed the store silently or rank its memos with the wrong
model: unset it, or for a refresh confirm the switch with `recall lite refresh --force`. Forget and
withdraw embed nothing, so they ignore such a variable and use the recorded embedder. Typing
`--embedder` with a different value for a refresh or an index switches the store to it, with a
warning, and the store's calibration then waits for a refit fitted with the new model. A switch
re-embeds only what the run covers (the memo folders for a refresh, the given path for `recall
index`); sources outside it keep the old vectors. Setup also records `RECALL_EMBED_PROFILE`, which
outranks the embedder name, and a different one in your shell is refused the same way. The MCP
server keeps the embedder setup configured until you run setup again with the new one; until then
it does not refresh a store that records another embedder, profile or model than its own, and its
answers say so. To change models cleanly, build a separate store, as described under updates below.

`recall forget` and `recall withdraw` also erase the memo's vectors from the shared embedding cache
(`RECALL_EMBED_CACHE`, on by default under the platform cache directory), under every embedder that
embedded its text, so they do not stay there until eviction. They erase the cache file the
forgetting process would use; a cache another setting of the variable pointed at is not reached.
Entries written by 0.15.0 and earlier record no digest of their text, so a forget cannot find them
until the next index or search that embeds the same text through the cache records one. While any
remain, a forget's message says so with a WARNING: delete the cache file once to clear them, which
costs only a re-embed. A forget whose closing checkpoint another reader of the cache blocked also
reports a WARNING, since the erased vectors may then stay in the file and its write-ahead log. With
the cache off or absent, a forget creates none.

With `--check`, exit code `0` means current and `1` means changed, never refreshed, or missing input. It compares the memo files, the store and the policy; it does not detect a change to the facts a refresh would state (an upgraded stating rule with no memo edited), which only a refresh records.
A writing refresh returns `0` for refreshed, unchanged, or coalesced work, `1` for failure, and `3`
when the write queue is busy. A successful refresh is not itself proof of certification; read its
calibration result.

Automatic refresh enforces the indexing file and byte budgets. If it cannot refresh, search
responses carry `refresh_warning`, `recall_stats` reports the last outcome, and doctor reports the
problem. A manual refresh warns and proceeds over budget, so inspect the chosen folder before
running it. A large unexpected deletion is guarded; restore the missing folder before considering
`--allow-prune`. The switches and queue limits are listed in [ENVIRONMENT.md](ENVIRONMENT.md).

Use the doctor and dashboard commands printed by setup. The dashboard reads the store read-only;
accepting a review or saving an edit there writes the memo file. A plain `recall doctor` in a new
shell can inspect PostgreSQL instead, because the CLI does not read the MCP client's environment.
Apply setup's printed environment to CLI searches as well:

```bash
recall search "what did I decide about caching?"
```

## See how memos link

Lite keeps the same semantic graph as the PostgreSQL install, inside the store file. It is built
only from what memos state: Markdown and wiki links, `supersedes`, `depends_on`, `recall_graph`
front matter and headings. No model is involved, so building it embeds nothing and calls no API.
Every corpus write clears the graph in the same transaction, so a forgotten memo never stays
in it. A refresh builds it again, `recall graph rebuild` builds it on demand, and the dashboard
builds one in memory, without writing it, when none is stored.

With setup's printed environment applied, so that `RECALL_DSN` names the store:

```bash
recall graph status
recall graph rebuild
```

`status` reports whether the graph is built for the store's current corpus and how many entities,
mentions and relations it holds; `rebuild` builds it now. A lite store has one corpus, so
`--generation` is optional, and when given it must be the `generation_id` that `status` prints.

In the dashboard, the memory graph's corpus switch lists the store's corpus beside the memo files.
That view draws every indexed file as a node, joined by supersession, authored references and
dependencies, Python imports, and uncommon names two files share; opening a node shows its chunks,
the entities they mention and the relations it takes part in.

The graph is for navigation. Search does not read it: graph expansion is a feature of the
PostgreSQL generation route, where `recall_reasoning_query` expands one hop by default (see
[REASONING_GRAPH.md](REASONING_GRAPH.md)); on a lite store it expands nothing. `recall_related`
answers the same structural relations (`source`, `ordinal`, `supersession`, `caused_by`) on both
backends.

## Atomic rescue

Atomic rescue finds a memo whose answer is one sentence inside a long chunk. Each chunk is split
into overlapping 24 word views (stride 12), the views are embedded, and a search moves at most one
parent chunk, the one whose best view matches the query most closely outside the dense top five, to
dense rank six before fusion. Fusion can move it into the final top five. That chunk is scored
by its own cosine, never the view's, so a rescue changes rank only:
it cannot turn an abstention into an answer. It ships from 0.16.0; 0.15.0 does not have it.

`recall setup --lite` turns it on. It registers two variables for the MCP server,
`RECALL_ATOMIC_RESCUE_MODE=active` and `RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT`, a folder beside the
store (`.recall/memory.db.atomic`), and builds the first artifact before it returns; its `atomic`
line says what it built. That build is most of setup's time on a large memory folder: measured on
a loaded host with the default local model, 1,776 chunks took about 35 minutes to index and about
two hours to build (2026-11-15). On a small project the difference is small. Its two packages, `numpy` and `threadpoolctl`, come with the `fastembed`
and `desktop` extras. In an environment without them, setup leaves the rescue off and says so; install the
`atomic` extra and run setup again to turn it on. A mode set in the shell that runs setup is registered as it is, so to keep the rescue
off, run setup with `RECALL_ATOMIC_RESCUE_MODE=off`. The CLI does not read the server's settings:
to use the rescue from `recall search`, set the same two variables in your shell.

```bash
export RECALL_ATOMIC_RESCUE_MODE=active
export RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT=/absolute/path/to/project/.recall/memory.db.atomic
```

The root must be absolute, and a relative one is refused: the MCP server and the CLI each run in
their own working directory, so a relative root would name two different folders.

`RECALL_ATOMIC_RESCUE_PLACEMENT=fused` places the rescued chunk at rank six in the retriever's
fused ranking, before trust evaluation. The default `dense` inserts it before fusion.

Several stores (two projects, or two MCP servers) may share one root. Each store keeps its
artifacts in a folder of its own under it, `store-<digest>`, named by the store's tenant and file
path, and a build removes old artifacts from that folder only. Moving a store file starts a new
folder; delete the old one by hand. The stores share `micro-views.sqlite`.

Before per-store folders, a build put its artifacts at the top of the root, as `lite-*` folders,
and removed every other such folder, including another store's. A search still serves the one for
the memory and calibration in force, after the same lineage check, so a root built that way keeps
working without a rebuild. The next build writes the store's own folder, and its `atomic` line says
how many `lite-*` folders remain at the top. No build removes them, because nothing records which
store built them: delete them by hand once every store using the root has built once.

With the rescue on, every write that keeps a calibration in force (a refresh, `recall index`,
`recall_index`, a background refit landing) builds an artifact for the memory and calibration then
in force, under the root, and the refresh report prints an `atomic` line. On a store that already
exists, the next refresh run with the variables set builds the artifact even when no memo changed,
and the MCP server starts that refresh at its next check. The refresh also checks the current
root and calibration when the memos stay the same. A certified store with five or fewer chunks
has no sixth candidate and serves ordinary trusted results without an artifact. Nothing is built
until the store has a certified calibration, and
until then every search with the rescue on refuses, in development trust mode too, where search
would otherwise answer uncalibrated. The views of each chunk text are kept in
`micro-views.sqlite` under the root, so a later build embeds only the views of new or edited
chunks. The first build is the expensive one: a chunk of about 120 words has about ten
views, so it embeds roughly ten times as many short texts as the first index did. That file grows
until a forget erases from it; delete it to reclaim space, at the cost of one full build.

The rescue applies only to a search with no folder, source or other scope and no source policy
(`RECALL_SOURCE_POLICY_FILE`): any other search neither uses the artifact nor needs it, so with a
source policy in force every build is paid for and no search uses it.

A search the rescue applies to refuses while the artifact for the memory as it is now is missing:
a search that lands between a write and the end of its build fails with an artifact error and
succeeds once the build finishes. `recall forget` and `recall withdraw` build nothing, and neither
does a `recall index` or `recall lite refresh` run without the two variables. The MCP server
notices such a change at its next check (at most every `RECALL_LITE_REFRESH_INTERVAL`, 30 s) and
refreshes, which builds the artifact again, mostly from views it already holds; until that build
ends such a search fails on the missing artifact.
A background refit that stores a new calibration and then fails to build only logs the failure, so
searches that need the artifact fail until a write or `recall lite refresh` builds it again. A store's older artifacts are removed after each build; on Windows one a running
server still has open stays until a later build.

`recall forget` and `recall withdraw` also erase what the rescue keeps of the memo: its views in
`micro-views.sqlite`, with SQLite `secure_delete` and a truncating checkpoint, and every artifact
in the store's own folder but one built for the memory as it is after the forget. A legacy
`lite-*` folder directly under the shared root has no recorded owner. If it names the memo or
its source list is unreadable, forget reports incomplete cleanup and leaves the folder for
ownership review.
Views are shared by text, so another store holding the same text re-embeds it at its next build.
They do so even with the rescue since turned off, and report an erasure that failed (on Windows,
an artifact a running server still has open). With `RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT` unset,
forget checks the default setup root beside the store, `<store-path>.atomic`. A custom root still
requires that variable in the forgetting process, or manual cleanup of that root.

## Update the software

Back up first, stop every MCP client and dashboard using this environment, and activate the same
environment used for installation:

```bash
python -m pip install --upgrade "recall-rag[fastembed,mcp]"
```

For a source install, run `git pull --ff-only` in the clean clone and then
`python -m pip install --upgrade "./re-call[fastembed,mcp]"`; record the installed commit with
`git rev-parse HEAD` when you need a reproducible deployment. Read [CHANGELOG.md](../CHANGELOG.md)
before the update.

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

Restore with all clients stopped. A store is bound to the absolute path of the memo folder it was
built from: every memo is recorded under its absolute path. Restore the notes to that same path,
restore the store, rerun setup with those paths to check calibration and client registration, then
run the printed doctor command and an explicit refresh check before restarting clients.

If the memo folder must live at a new path, do not rerun setup on the restored store: it would index
every memo again under its new path while the rows under the old path stay and keep being returned,
so each memo would be found twice. Build a new store from the notes instead, with
`recall setup --lite --memory <new folder> --store <new file>`. Fact ledger data lives only in the
old store, so keep that file. If the environment moved, also update the absolute interpreter,
database, and memo paths in client configuration.

## Common problems

| Symptom | Next step |
|---|---|
| `--lite` is unrecognized | The environment holds a release older than `0.15.0`, or another environment is active. Activate the one you installed and check `python -m pip show recall-rag` and `recall setup --help`. |
| Doctor tries PostgreSQL | Apply the environment printed by Lite setup; client configuration is not shell configuration. |
| No tools after setup | Inspect registration results, restart the client in the configured project, and check its interpreter path. |
| Store is indexed but searches refuse | Inspect calibration status; add useful memos and refresh while keeping strict trust enabled. After a forget, this is expected: forgetting a memo, or a refresh that prunes one, deletes the calibration test questions built from it, and the next refresh refits the store ([CALIBRATION.md](CALIBRATION.md)). |
| Changed memo is missing | Check `refresh_warning` and run refresh with explicit store and memo paths. |
| Writer is busy | Let the current writer finish and retry; do not remove live lane files. |
| Need shared serving or generations | Use the [PostgreSQL installation](INSTALLATION.md#postgresql-database); copying the SQLite file is not a PostgreSQL migration. |
