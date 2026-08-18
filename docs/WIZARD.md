# The installation wizard

`recall wizard` takes a JSON config and drives every corpus from a directory to a calibrated,
promoted generation, then writes the MCP servers that serve them. It exists because the calibrated
path has enough load-bearing ordering in it that assembling it by hand is a research project:
build → validate → **calibrate → publish** → promote, with promotion last because it is irreversible
and because promoting first gives a fresh corpus fingerprint that makes the calibration stale.

```bash
python -m recall.cli wizard --headless --config wizard.json
```

`--headless` is required rather than implied. The interactive and GUI front ends do not exist yet, and
a bare `recall wizard` that silently built, calibrated and promoted would be the wrong surprise for
an installer.

## The config

```json
{
  "dsn": "postgresql://recall_server:PASSWORD@127.0.0.1:5432/recall",
  "migration_dsn": "postgresql://recall_migrator:PASSWORD@127.0.0.1:5432/recall",
  "serving_role": "recall_server",
  "embedder": "fastembed",
  "corpus_version": "2026-08-18",
  "docs_root": "C:/projects/myapp/docs",
  "code_root": "C:/projects/myapp/src",
  "memory_root": "C:/projects/myapp/memory",
  "project_root": "C:/projects/myapp",
  "project": "myapp"
}
```

| Key | Required | What it decides |
|---|---|---|
| `dsn` | yes | the serving connection: where chunks are written and read |
| `migration_dsn` | yes | the DDL owner, which applies migrations and issues GRANTs |
| `embedder` | yes | the vector dimension, which is welded to the table |
| `corpus_version` | yes | stamped on every chunk; the convention is an ISO date |
| `docs_root`, `code_root`, `memory_root` | yes | the three corpora. **Absolute paths only** |
| `project_root` | no | where `.mcp.json`, `.env` and `CLAUDE.md` are written |
| `project` | no | a label stamped on every chunk, so a hit can say where it came from |
| `serving_role` | conditionally | see the two-role case below |

Every root must be absolute. A relative root resolves against the wizard's own working directory, so
the commit stamped on each chunk would be the wizard's rather than the corpus's.

**On Windows, write paths with doubled backslashes or forward slashes.** A single backslash is a JSON
escape, and the refusal names the problem when it happens.

## The two-role case, which is the one that fails quietly

If `dsn` and `migration_dsn` authenticate as **different roles**, `serving_role` is required and the
wizard refuses without it. No migration emits a GRANT: the role name is a deployment decision the
packaged SQL cannot know, so the serving role would own no privileges on the tables just created and
every query would fail with `permission denied` *after* the install reported success.

This cannot fail on a laptop or in CI, where both DSNs are usually the same superuser. It fails only
on the deployment the two keys exist for. See `docs/MIGRATIONS.md` for the role model.

## What the three corpora are, and why they differ

| Tenant | Calibrated | Served as | Why |
|---|---|---|---|
| `docs` | yes | `RECALL_ENV=production`, strict trust | prose separates well, so it can certify |
| `code` | yes | `RECALL_ENV=production`, strict trust | same, measured: separability 0.999 on this repo |
| `memory` | **no** | development trust, no `RECALL_ENV` | writable after install, and a fresh directory cannot meet the certification floor |

`memory` is deliberately never calibrated. Certification needs at least 20 answerable and 20
unanswerable questions, a fresh memory directory holds one file, and a calibrated memory tenant would
answer `CALIBRATION_MISSING` forever. It is also the one corpus that must accept writes after install,
which production mode refuses. It is indexed into the legacy `chunks` table instead.

## Reading the report

The exit code is the only thing CI reads, so the three outcomes are not collapsed.

- **`certified and promoted`** — the generation is live and its calibration is published.
- **`DEGRADED, not promoted`** — it built and validated but did not certify, so it was deliberately
  left unpromoted. Whether this is an install depends on what came before it:
  - if the tenant already had a generation, that one keeps serving and the exit code is **0**;
  - if this was a **first** install, the tenant answers nothing and the exit code is **1**. Those two
    states used to render identically, which is why they are now spelled out.
- **`REFUSED`** — a configuration problem, raised before anything was built.
- **`FAILED`** — a crash partway through, which may have left a promoted generation behind.
- **`NO SERVER`** — a tenant deliberately given no MCP server because it cannot answer.

A degraded generation still holds a full copy of the corpus. `recall generation abandon <id>` releases
it for `recall generation gc`; the report names the command with the id filled in.

## Resuming

A corpus takes minutes to build. The run records what it finished, keyed on a digest of the config
fields that would invalidate it, and reuses a **promoted** corpus rather than rebuilding it.

```bash
python -m recall.cli wizard --headless --config wizard.json            # resumes by default
python -m recall.cli wizard --headless --config wizard.json --fresh    # rebuild everything
python -m recall.cli wizard --headless --config wizard.json --no-state # never read or write state
```

The state file defaults to the config path with a `.state.json` suffix, and `--state PATH` moves it.

Three things worth knowing:

- **A degraded corpus is always retried.** It was left unpromoted precisely so a re-run would retry
  it; reusing it would make the re-run a no-op for the one corpus you are re-running to fix.
- **The digest covers the CONFIGURATION, not the content.** Editing a document and re-running reuses
  the old generation. `--fresh` is the answer. Hashing every corpus before deciding to skip it would
  cost a large fraction of what the skip saves.
- **A state file that cannot be read or written never fails the install.** Losing it costs a rebuild;
  refusing would cost the completed corpus.

## What it writes

With `project_root` set:

- **`.mcp.json`** — one server per servable tenant. Merged into any existing `mcpServers`, so servers
  the wizard knows nothing about survive.
- **`.env`** — the serving DSN and embedder, for the CLI. **Not** for the MCP servers: a stdio server
  launched with an explicit `env` block inherits nothing, so each server carries its own variables.
- **`CLAUDE.md`** and **`memory/MEMORY.md`** — block-scoped edits between markers, so your own file
  content is preserved.

Without `project_root` nothing is written and the report says so. The wizard will not guess a
location to put an MCP configuration in.

**Docker autostart is not configured.** Making a container start on login is a change to the machine
rather than the project; on Windows it is a Docker Desktop setting.

## What is not built yet

- The interactive and GUI front ends. `--headless` is required for that reason.
- An end-to-end smoke search per server after wiring. The configuration is written from what
  actually happened, and a tenant that cannot answer gets no server, but the wizard does not yet
  issue a query to prove each server answers.
- The Windows installer (`.exe`), winget prerequisites and reboot-resume.
