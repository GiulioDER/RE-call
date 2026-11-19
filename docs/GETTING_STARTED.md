# Getting started

Install RE-call, connect it to your agent, or try the PostgreSQL demo. The short version is in the
[README](../README.md#quickstart).

## Start with Lite

For one project's memory on your machine, start with **RE-call Lite**. It stores memory in
`.recall/memory.db`, refreshes changed memos automatically, and needs no Docker or database server.

**Install 0.15.0 or later:** Lite ships in the PyPI wheel from `0.15.0`; earlier wheels do not
contain it. Follow the [Lite installation guide](LITE.md#install-a-build-that-contains-lite)
to create an environment and install the package, then run this from your own project with an
existing folder of notes:

```bash
recall setup --lite --memory ./notes
```

Setup indexes the notes, attempts automatic calibration, and registers Claude Code for this project.
Inspect the calibration and registration results, use the printed doctor command, and restart the
client. Strict trust stays enabled; a corpus that cannot certify will refuse trusted answers.
For other MCP clients, use [manual Lite wiring](LITE.md#connect-another-client).

| Choose | Best fit | Maintenance |
|---|---|---|
| **Lite** | One project on one machine, using a local SQLite file | Automatic memo refresh and calibration; back up notes and the store. |
| **PostgreSQL** | Shared serving, immutable generations, multiple tenants, or authentication | Manage the database, schema migrations, generations, calibration, and backups. |

### Try the PostgreSQL demo

Prerequisites: Python 3.11 or newer, Docker, and a Docker installation able to run PostgreSQL with
pgvector.

```bash
pip install "recall-rag[fastembed]"
recall quickstart
```

The demo starts a throwaway database, indexes a small corpus included in the package, and runs
three searches. It includes a normal answer, a stale claim that is returned as superseded, and a
question that is refused. The demo uses development trust and changes no personal files.

Remove the demo database when finished:

```bash
recall quickstart --remove
```

Already have PostgreSQL with pgvector? Use `recall quickstart --existing-dsn <dsn>` instead. The
demo is intentionally separate from a real install and is not calibrated for your data.

## Install and integrate

### Local, with no database server and no Docker

Use the [Lite installation](LITE.md#install-a-build-that-contains-lite), then run
from the project whose notes you want to index:

```bash
recall setup --lite --memory ./notes
```

Setup creates `.recall/memory.db`, attempts automatic calibration, and registers Claude Code for
this project. Use the printed doctor and dashboard commands (the dashboard reads the store
read-only; accepting a review or saving an edit there writes the memo file), and add `.recall/` to
`.gitignore`. For CLI or manual client wiring, add `--no-register`.

The MCP server refreshes changed memos automatically. To check or refresh explicitly:

```bash
recall lite refresh --store .recall/memory.db --memory ./notes --check
recall lite refresh --store .recall/memory.db --memory ./notes
```

Use the same memo root and embedder as setup. The CLI does not read the MCP client's environment;
apply setup's printed settings before CLI searches or doctor. Strict trust remains enabled, and
successful indexing alone does not guarantee certification.

The [Lite guide](LITE.md) covers other clients, source updates, backup and restore, refresh
budgets, and queued writes. Keep the file on a local disk. Lite uses exact cosine search and
SQLite keyword retrieval; results can differ from PostgreSQL. Generations, multi-tenant serving,
and authentication require PostgreSQL. See [the API](API.md) for trust and retrieval limits.

### With PostgreSQL

For your own corpus, provide PostgreSQL with pgvector and run the guided setup wizard after
installing `recall-rag[fastembed]`:

```bash
recall setup
```

It applies the schema, asks for the embedder and retrieval options, indexes the corpus, and
registers the selected agent integration. It does not calibrate: run `recall wizard`, which builds,
calibrates and publishes a generation, with a labeled query file that refers to the corpus you are
installing. Calibration fitted
to the bundled demo is only an example, not a certification for your data. The schema uses an
ordered SQL migration path and pre-tenancy tables are migrated in place.

For Docker, an existing database, headless provisioning, manual calibration, and troubleshooting,
see [docs/INSTALLATION.md](INSTALLATION.md) and [docs/WIZARD.md](WIZARD.md).

### Choose an integration

| Use case | Install | Next step |
|---|---|---|
| Lite CLI or MCP | [Install Lite](LITE.md#install-a-build-that-contains-lite) (in the PyPI wheel from 0.15.0) | Run `recall setup --lite --memory ./notes`; use `--no-register` for CLI or manual client wiring. |
| PostgreSQL CLI and Python | `pip install "recall-rag[fastembed]"` | Run `recall setup`, then use `recall search` or the [Python API](API.md). |
| PostgreSQL MCP clients | `pip install "recall-rag[fastembed,mcp]"` | Run setup and follow [the MCP guide](USING_WITH_CLAUDE.md). Host specific steps are below. |
| Claude Agent SDK | `pip install "recall-rag[agent,fastembed]"` | Use the in process integration in [USING_WITH_AGENT_SDK.md](USING_WITH_AGENT_SDK.md). |
| LangChain or LlamaIndex | Install the matching extra | Use the adapters described in [API.md](API.md). |
| Windows desktop UI | `pip install "recall-rag[desktop]"` | Run `recall-install`; the current release does not ship a standalone Windows binary. See [the wizard guide](WIZARD.md). |

#### Claude Code

Inside Claude Code, install the plugin after installing the Python package:

```text
/plugin marketplace add GiulioDER/RE-call
/plugin install recall@re-call
```

The plugin supplies the MCP server, memory search skill, and lifecycle hooks. Lite setup already
registers a project server; the plugin is optional. If you add it, configure its Database DSN with
the SQLite DSN printed by setup and avoid enabling duplicate servers for the same memory. The
plugin's server does not refresh memos by itself; run `recall lite refresh` yourself or keep the
server setup registers.
For PostgreSQL, run `recall setup` against the project and database Claude should use.
Details are in [plugin/README.md](../plugin/README.md).

#### Codex

For Lite, use [manual MCP wiring](LITE.md#connect-another-client): `recall setup --lite` only
registers Claude Code. For PostgreSQL, run `recall setup` from the project. When Codex is detected,
setup installs the Codex MCP server,
plugin bundle, memory and session skills, and hooks into the user configuration. Restart Codex afterward. The
Codex and Claude Code integrations share the same memo format and trust layer. See
[docs/CODEX_RECALL_INTEGRATION.md](CODEX_RECALL_INTEGRATION.md).

#### Claude Agent SDK

The SDK integration runs the same tools in process and does not start an MCP server:

```python
from recall_agent import RecallAgentMemory

with RecallAgentMemory.from_env() as memory:
    options = memory.options()
```

See [docs/USING_WITH_AGENT_SDK.md](USING_WITH_AGENT_SDK.md) for the complete example and
write-tool boundaries.

#### LangChain and LlamaIndex

Both adapters use the same trusted retrieval path. If trust abstains, they return no document by
default, and returned documents retain verdict, confidence, cosine, and supersession metadata.
See [docs/API.md](API.md) for the supported classes and methods.

### If an install is not working

```bash
recall doctor
```

For Lite, first apply the environment printed by setup so doctor inspects the SQLite store.
For PostgreSQL, the doctor checks the interpreter, console scripts, embedder, Docker, database,
pgvector, schema,
configured table and tenant, calibration, and agent registration. It changes nothing and prints the
repair command for each problem.
