# Install RE-call

This page contains the longer installation path that does not belong in the root README. Use it
when you are installing RE-call for your own corpus, provisioning it without prompts, or debugging
an existing setup.

## Choose a path

| Goal | Command or guide |
|---|---|
| See RE-call work | `pip install "recall-rag[fastembed]"` then `recall quickstart` |
| Install for your own corpus | `pip install "recall-rag[fastembed]"` then `recall setup` |
| Provision from a reviewed JSON file | `recall wizard --headless --config wizard.json` |
| Configure Claude Code or Codex | Run `recall setup`, then see [the Claude guide](USING_WITH_CLAUDE.md) or [the Codex guide](CODEX_RECALL_INTEGRATION.md) |
| Configure Claude Desktop or another MCP client | See [USING_WITH_CLAUDE.md](USING_WITH_CLAUDE.md) |
| Diagnose without changing anything | `recall doctor` |

The distribution name is `recall-rag`; the command and Python package name are `recall`. Do not
install the unrelated PyPI package named `recall`.

## Prerequisites

- Python 3.11 or newer
- PostgreSQL with the pgvector extension
- Docker, if you want RE-call to start PostgreSQL for you

The default `fastembed` extra provides a local embedding backend. Hosted alternatives are available
through the `voyage` and `openai` extras. Optional retrieval and integration extras are listed in
the [dependency guide](DEPENDENCIES.md).

## Database

RE-call keeps the corpus in your PostgreSQL database. If you need a disposable local database, save
this as `docker-compose.yml`:

```yaml
services:
  db:
    image: pgvector/pgvector:pg18
    environment:
      POSTGRES_USER: recall
      POSTGRES_PASSWORD: recall
      POSTGRES_DB: recall
    volumes:
      - recall_pgdata:/var/lib/postgresql
    ports:
      - "127.0.0.1:5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U recall"]
      interval: 2s
      timeout: 3s
      retries: 30

volumes:
  recall_pgdata:
```

Start it with:

```bash
docker compose up -d --wait
```

If PostgreSQL with pgvector already exists, provide its DSN to `recall setup` when prompted. The
wizard applies the schema itself after the embedder is chosen, because the vector dimension belongs
to the selected embedder.

## Guided setup

```bash
pip install "recall-rag[fastembed]"
recall setup
```

The wizard can:

1. select the embedder, reranker, entailment backend, and retrieval profile
2. apply pending PostgreSQL migrations
3. build an indexed corpus and its immutable generation
4. fit and publish calibration when you provide labeled queries
5. register MCP servers and offer memory scaffolding for the selected agent

When asked whether to calibrate, use queries labeled against the same corpus being installed. A
calibration made from the bundled sample corpus demonstrates the mechanism but is not valid for
your own corpus.

The interactive setup is the recommended install. The scriptable equivalent is documented in
[WIZARD.md](WIZARD.md), including the JSON configuration, resume behavior, GUI front end, and
two-role database deployments.

## Headless setup

Create a configuration file with absolute corpus paths:

```json
{
  "dsn": "postgresql://recall_server:PASSWORD@127.0.0.1:5432/recall",
  "migration_dsn": "postgresql://recall_migrator:PASSWORD@127.0.0.1:5432/recall",
  "serving_role": "recall_server",
  "embedder": "fastembed",
  "corpus_version": "2026-09-05",
  "docs_root": "C:/projects/myapp/docs",
  "code_root": "C:/projects/myapp/src",
  "memory_root": "C:/projects/myapp/memory",
  "project_root": "C:/projects/myapp",
  "project": "myapp"
}
```

Then run:

```bash
recall wizard --headless --config wizard.json
```

The wizard builds, validates, calibrates, publishes, and promotes in that order. It records state
next to the configuration and can resume a completed install. Use `--fresh` to rebuild after corpus
content changes.

## Manual schema and corpus path

Use the manual path when the serving role cannot create tables. Apply migrations with the owner DSN,
then index and search with the serving configuration:

```bash
recall --migration-dsn postgresql://recall:recall@localhost:5432/recall schema --dim 384 apply
RECALL_TRUST_MODE=development recall --table recall_notes index ./notes
RECALL_TRUST_MODE=development recall --table recall_notes search "what did we decide about caching?"
recall lint ./notes
recall check ./notes/new-memo.md --strict
```

Development mode is for local evaluation of an uncalibrated corpus. For production, query an active
generation with a published calibration. See [OPERATING_MODES.md](OPERATING_MODES.md) and
[CALIBRATION.md](CALIBRATION.md).

Use separate credentials when possible:

- `RECALL_SERVING_DSN` for application and MCP traffic
- `RECALL_MIGRATION_DSN` for schema changes
- `RECALL_TENANT` and `RECALL_TABLE` for the corpus scope in the legacy local path

Never commit passwords or API keys to MCP configuration, `.env` files, or the repository. See
[MIGRATIONS.md](MIGRATIONS.md), [AUTH.md](AUTH.md), and [ENVIRONMENT.md](ENVIRONMENT.md).

## Integration setup

The package installs the core CLI and Python API. Add only the integration you need:

| Integration | Extra and follow-up |
|---|---|
| MCP | `pip install "recall-rag[fastembed,mcp]"`; use [USING_WITH_CLAUDE.md](USING_WITH_CLAUDE.md) |
| Claude Code | Install the MCP package, install the [plugin](../plugin/README.md), and run `recall setup` |
| Claude Desktop | Install the MCP package and add the server block from [USING_WITH_CLAUDE.md](USING_WITH_CLAUDE.md) |
| Codex | Install the base package and run `recall setup`; see [CODEX_RECALL_INTEGRATION.md](CODEX_RECALL_INTEGRATION.md) |
| Claude Agent SDK | `pip install "recall-rag[agent,fastembed]"`; see [USING_WITH_AGENT_SDK.md](USING_WITH_AGENT_SDK.md) |
| LangChain | `pip install "recall-rag[langchain,fastembed]"` |
| LlamaIndex | `pip install "recall-rag[llamaindex,fastembed]" llama-index-core` |
| Windows desktop UI | `pip install "recall-rag[desktop]"`, then run `recall-install` |

All integrations share the same trust and provenance behavior. They differ only in how the result
is delivered to the host application.

## Troubleshooting

```bash
recall doctor
```

The doctor is read only. It checks the interpreter, console scripts on `PATH`, embedder backend,
Docker, database, pgvector, schema, configured table and tenant, calibration, and client
registration. It prints a repair command for each problem it finds.

If an agent starts successfully but finds no memory, first verify `RECALL_TABLE` and `RECALL_TENANT`.
An incorrect table or tenant is intentionally isolated and can look exactly like an empty corpus.
