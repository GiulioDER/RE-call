# The installation wizard

`recall wizard` takes a JSON config and drives every corpus from a directory to a calibrated,
promoted generation, then writes the MCP servers that serve them. It exists because the calibrated
path has enough load-bearing ordering in it that assembling it by hand is a research project:
build → validate → **calibrate → publish** → promote, with promotion last because it is irreversible
and because promoting first gives a fresh corpus fingerprint that makes the calibration stale.

```bash
python -m recall.cli wizard --headless --config wizard.json
```

`--headless` is required rather than implied: a bare `recall wizard --config <file>` that
silently built, calibrated and promoted would be the wrong surprise for an installer.

🔁 **Corrected: there are now three front ends, and this page used to say there were none.** They
are three RENDERERS of one question plan (`recall/wizard/questions.py`) driving one engine, not
three installers:

| | |
|---|---|
| `recall wizard` | asks in the terminal |
| `recall wizard --gui`, or `recall-install` | asks in a window |
| `recall wizard --headless --config <file>` | asks nothing |

All three produce the same config document and run it from disk, so an install made in a window is
one you can re-run in CI.

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
| `project_root` | no | the `cwd` each MCP server runs from, and where `.env` and `CLAUDE.md` are written |
| `project` | no | a label stamped on every chunk, so a hit can say where it came from |
| `serving_role` | conditionally | see the two-role case below |

Every root must be absolute. A relative root resolves against the wizard's own working directory, so
the commit stamped on each chunk would be the wizard's rather than the corpus's.

**`project_root` is the one root the wizard creates, and only its last directory.** A project
directory that does not exist yet is an ordinary first install, so it is made before `.env` and
`CLAUDE.md` are written. A `project_root` whose *parent* is also missing is refused by name before
anything is built, because that is a mistyped path rather than a place you meant to put a project,
and so is one that exists and is a file. Both refusals happen at config-reading time on purpose:
the wiring is the last step of the run, so a bad value discovered there costs the whole install
after every corpus has been built, calibrated and promoted.

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

- **MCP servers in `~/.claude.json`** — one per servable tenant, registered at **local scope**, under
  this project's own entry in Claude Code's configuration file. Merged into any existing
  `mcpServers`, backed up first and written atomically, so servers the wizard knows nothing about
  survive.

  ⚠️ **Local scope, and deliberately neither of the other two.** Project scope means a `.mcp.json`
  in the tree, which Claude Code gates behind an approval prompt: until it is answered the tools are
  silently absent, with no error naming the cause. Measured on one machine, 2 of 310 tracked
  projects had any approval recorded. **User scope** skips that prompt but is the only scope that
  loads in *every* project on the machine, and each of these servers carries a `RECALL_TENANT` for
  this project's corpus, so they would answer confidently about the wrong repository everywhere
  else. Local scope skips the prompt and loads in one project, which is how the blocks are built:
  one `project_root`, one DSN, one tenant each.

  Three consequences worth stating.

  **The entry is keyed by PATH**, so moving or renaming the project orphans it silently. The report
  names the key it registered under. Re-run the wizard after a move. Claude Code does not normalise
  that key either, and one project can carry several spellings of it; every existing key that
  resolves to the project directory is written, so a native launch and a Git Bash launch both find
  the servers.

  **A server name already registered by something else is refused, not repointed**: names are
  `{project}-{kind}`, so a second install under the same `project` would silently aim the first
  one's servers at a different corpus. Use a distinct `project`.

  **The wizard will not create `~/.claude.json` if it is absent**, because inventing another
  application's configuration file is not an installer's business; start Claude Code once and
  re-run. In the refusal cases the report says so and the install is reported incomplete, since
  servers no client can see are the whole failure this step exists to prevent.

  Restart Claude Code afterwards. It reads this file at startup.
- **`.env`** — the serving DSN and embedder, for the CLI. **Not** for the MCP servers: a stdio server
  launched with an explicit `env` block inherits nothing, so each server carries its own variables.
- **`CLAUDE.md`** and **`memory/MEMORY.md`** — block-scoped edits between markers, so your own file
  content is preserved.

Without `project_root` nothing is written or registered and the report says so. The wizard will not
guess which project these servers belong to.

**Docker autostart is not configured.** Making a container start on login is a change to the machine
rather than the project; on Windows it is a Docker Desktop setting.

## Choosing a database

Two shapes of install, and the wizard asks which one you want:

| | What it does | When |
|---|---|---|
| **Docker** | provisions a PostgreSQL container at a location you choose | you have no PostgreSQL, or want this install isolated from one you do have |
| **Existing** | uses a PostgreSQL you already run | you already have one with pgvector, and would rather not run Docker at all |

With an existing database, nothing about Docker is touched: `provision_stack` returns before any
compose file is written.

**An existing database is checked before it is accepted**, in one read-only connection, because
every way it can be wrong otherwise fails minutes later and names something else. Reachable;
pgvector present, or available and merely not created; a role that may create objects; and an
existing `chunks.embedding` whose dimension matches the embedder you chose. That last one is the
expensive one: a mismatch does not fail during setup at all, it fails on the first insert, well into
a build, with a driver error naming neither side of the disagreement.

### A database behind SSH

Open a tunnel first, and give the wizard the local end:

```bash
ssh -L 5433:localhost:5432 user@your-host
```

Then the connection string is an ordinary local one, `postgresql://user:password@127.0.0.1:5433/recall`,
and everything above applies unchanged.

**The wizard deliberately does not manage the tunnel.** Doing so would mean owning key handling,
host-key verification, reconnection and process lifetime, which is a large surface and a class of
failure that is hard to report clearly to somebody installing their first index. One documented
command does the same job and fails in ways its own documentation already covers.

## Running it

```bash
recall wizard                                  # asks, writes a config, then runs it
recall wizard --headless --config wizard.json  # runs a saved config, asks nothing
```

The interactive flow **writes the config file and then runs that file**, rather than installing
from the answers directly. So there is one engine rather than two that drift, and you keep an
artefact you can re-run, hand to somebody else, or commit to CI.

It refuses, before asking anything, in a session with no terminal. Piping into it would otherwise
either hang on a line that never arrives or read EOF and accept every default, and both of those
look like a successful install from the outside.

- **A smoke search per server.** After wiring, one real query goes through each configured server,
  drawn from that tenant's own indexed text and using the store its own `RECALL_ENV` selects. The
  report shows hits, trust state and any failure code, so "a config was written" and "the install
  answers" are separate lines rather than one assumption.

## Uninstalling

```bash
recall uninstall --data-root C:/Users/me/.recall            # prints the plan, then asks
recall uninstall --data-root C:/Users/me/.recall --dry-run  # prints the plan and stops
recall-uninstall --data-root C:/Users/me/.recall            # the same thing in a window
```

⛔ **It never removes a folder, and that is the whole design.** The installer SUGGESTS the corpus
roots underneath the data folder, so on a default install your notes, your source and your agent
memory sit inside the directory being uninstalled. Removing the index is recoverable by
re-indexing; removing what was indexed is not. So it removes the specific files the installer
wrote, by name, and prints the corpus roots under **This will KEEP** so you can see they survived.

What it removes: the stack's containers (found by the compose project label recorded in the stack
file, never by a name pattern, so a second install on the same machine is untouched), the files the
installer wrote into the data folder, and the MCP registrations whose `cwd` marks them as written by
this install. An entry you wrote by hand under a name the wizard also uses is left alone.

`--purge-data` additionally removes the database volume holding the built indexes. Off by default:
they are reproducible by re-indexing and expensive to rebuild, so whoever reinstalls next week and
whoever is reclaiming disk want opposite things.

## The Windows executable

`packaging/recall-install.spec` builds a frozen bundle of the graphical installer, for somebody who
has no Python and is not going to get one.

```bash
pip install pyinstaller
cd packaging && python -m PyInstaller recall-install.spec --noconfirm --distpath ../dist
```

Two choices in that spec are not preferences:

- **`onedir`, not `onefile`.** A onefile build unpacks its whole payload to a temporary directory on
  every launch, and this bundle carries PySide6 and the ONNX runtime. On an installer, whose entire
  job is to reassure somebody that something is happening, that is a long unexplained pause before
  the window appears.
- **`collect_submodules("recall")`.** This codebase imports lazily nearly everywhere, so
  PyInstaller's static analysis cannot see most of what an install actually needs. A bundle built
  from the visible imports alone starts fine and dies with `ModuleNotFoundError` at the moment the
  user presses Install.

## Code signing, which is what makes the binary runnable at all

⛔ **This is not cosmetic, and "SmartScreen warning" understates it.** Measured 2026-08-21 on a
Windows 11 machine with Smart App Control enforced (`VerifiedAndReputablePolicyState = 1`), the
freshly built unsigned executable was refused outright:

    An Application Control policy has blocked this file

There is no "run anyway". Smart App Control is on by default for machines that qualify, and it can
only be enabled on a clean install — so a user who turns it off to run recall **cannot turn it back
on without resetting Windows**. Do not suggest that as a workaround.

The fix is settled by Microsoft's own documentation
([Smart App Control overview](https://learn.microsoft.com/en-us/windows/apps/develop/smart-app-control/overview)):

> If the app intelligence service is unable to make a prediction, then Smart App Control will still
> allow an app to run if it is signed with a certificate issued by a certificate authority (CA)
> within the Trusted Root Program.

**A signature is sufficient. Reputation is not additionally required**, which matters more than it
sounds: it means the first release runs, rather than being blocked until enough people have
downloaded it to build reputation it cannot build while blocked. An **OV** certificate does this;
EV buys nothing extra here.

### Getting a certificate

The private key for any publicly trusted code signing certificate must live in hardware or a cloud
HSM, so none of these routes end with a file you keep.

| Route | Cost | The catch |
|---|---|---|
| [SignPath Foundation](https://signpath.org/) | free for open source | needs a public repository, a recognized licence and an existing release; every release is manually approved |
| [Azure Artifact Signing](https://azure.microsoft.com/en-us/products/artifact-signing) (was Trusted Signing) | $9.99/month basic | **individual sign-up is US and Canada only**; the organisation route wants verifiable business history |
| A CA directly (DigiCert, Sectigo, SSL.com) | a few hundred a year | you buy and manage the token or cloud HSM yourself |

recall qualifies for the SignPath Foundation route on paper: the repository is public, the licence
is Apache-2.0, and there are releases on PyPI and GitHub. That is an application somebody has to
make and be approved for; it is not something the build can arrange for itself.

### What is already wired

`.github/workflows/windows-installer.yml` signs through SignPath when two settings exist, and does
the right thing when they do not:

- repository **variable** `SIGNPATH_ORGANIZATION_ID`
- repository **secret** `SIGNPATH_API_TOKEN`

Without them a pull request build still runs and still self-tests, unsigned — that is a check, not
a download. **A tag build without them fails**, deliberately: shipping an unsigned release is worse
than shipping none, because it works for whoever built it and is blocked for the audience it exists
to serve, with nothing in the release saying which one you got.

After signing, `signtool verify /pa` runs against the artifact. The signing step's exit code only
says the service accepted the request; `signtool` asks Windows whether the file now carries a chain
it will honour, which is the question the user's machine will ask.

## What is not built yet

- winget prerequisites and reboot-resume. The bundle assumes Docker Desktop is already installed;
  it does not install it, and it does not survive the reboot Docker Desktop asks for.
- A certificate. Everything above is wired and inert until somebody obtains one; see the table.
