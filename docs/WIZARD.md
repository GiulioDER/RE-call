# The installation wizard

This page describes the PostgreSQL installer. For local SQLite memory without a database server,
use [RE-call Lite](LITE.md). `recall setup --lite` is a separate noninteractive path that does not
run this wizard; it wires Claude Code and Codex the same way this wizard does (see
[What it writes](#what-it-writes)).

> **When to prefer this page:** use `recall wizard` for a calibrated, promoted generation. It can
> ask questions interactively or run a saved config with `--headless --config` for CI and
> provisioning. `recall setup` configures the installation; it does not calibrate, and it does
> not build or promote a generation. For the manual command-by-command path, see
> [FIRST_CALIBRATION.md](FIRST_CALIBRATION.md).

`recall wizard` takes a JSON config and drives every corpus from a directory to a calibrated,
promoted generation, then writes the MCP servers that serve them. It exists because the calibrated
path has enough load-bearing ordering in it that assembling it by hand is a research project:
build → validate → **calibrate → publish** → rescue → promote, with promotion last because it is
irreversible and because promoting first gives a fresh corpus fingerprint that makes the calibration
stale. `rescue` builds the generation's atomic rescue artifact ([below](#atomic-rescue)).

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
| `dsn` | one of `dsn` or `data_root` | the serving connection to an existing database |
| `data_root` | one of `dsn` or `data_root` | absolute directory for a managed Docker stack; its database uses a named volume |
| `migration_dsn` | no | the DDL owner; defaults to `dsn` for an existing database and cannot be set with `data_root` |
| `embedder` | yes | the vector dimension, which is welded to the table |
| `corpus_version` | yes | stamped on every chunk; the convention is an ISO date |
| `docs_root`, `code_root`, `memory_root` | yes | the three corpora. **Absolute paths only** |
| `project_root` | no | the `cwd` each MCP server runs from, and where `.env` and `CLAUDE.md` are written |
| `project` | no | the tenant namespace and chunk provenance label; defaults to `default` |
| `serving_role` | conditionally | see the two-role case below |

Every root must be absolute. A relative root resolves against the wizard's own working directory, so
the commit stamped on each chunk would be the wizard's rather than the corpus's.

Set exactly one of `dsn` and `data_root`. For a managed Docker install, omit both DSN keys:

```json
{
  "data_root": "C:/projects/myapp/recall-stack",
  "embedder": "fastembed",
  "corpus_version": "2026-08-18",
  "docs_root": "C:/projects/myapp/docs",
  "code_root": "C:/projects/myapp/src",
  "memory_root": "C:/projects/myapp/memory",
  "project_root": "C:/projects/myapp",
  "project": "myapp"
}
```

The managed image supports `hashing` and `fastembed` (including `fastembed:<model>`). Other
embedders require an existing database via `dsn` and a host environment with their dependencies
and credentials. A managed install refuses those backends before creating the stack or building
corpora. Project names use ASCII letters, digits, hyphens, underscores and dots; they cannot end
in `-docs`, `-code` or `-memory`.

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
| `{project}-docs` | yes | `RECALL_ENV=production`, strict trust | prose separates well, so it can certify |
| `{project}-code` | yes | `RECALL_ENV=production`, strict trust | same, measured: separability 0.999 on this repo |
| `{project}-memory` | **no** | development trust, no `RECALL_ENV` | writable after install, and a fresh directory cannot meet the certification floor |

For the example above these are `myapp-docs`, `myapp-code` and `myapp-memory`. Omitting
`project` produces `default-docs`, `default-code` and `default-memory`.

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
  re-run. In the refusal cases the report says so, and the install is reported incomplete unless
  Codex got the servers (below), since servers no client can see are the whole failure this step
  exists to prevent.

  Restart Claude Code afterwards. It reads this file at startup.
- **For each client found on the machine, its hooks and skills**, reported line by line:
  - Claude Code: the hooks in `~/.claude/settings.json` (or only the hook config, when the RE-call
    plugin is installed for this project) and the five skills in `~/.claude/skills/`.
  - Codex: the same servers in this project's `.codex/config.toml` (which Codex reads only for a
    trusted project, and which is added to the project's `.gitignore` inside a git work tree),
    the hooks in `CODEX_HOME/hooks.json`, and the skills in `~/.agents/skills/`.

  The hook config is bound to `project_root` and to the memory tenant, with prompt-time search
  reading `memory_root`. One that already serves another project is left alone, one that names no
  project is taken over and the report says so, a switch you
  turned off in it stays off, and a skill of the same name RE-call did not install is left alone.
- **`.env`** — the serving DSN and embedder, for the CLI. **Not** for the MCP servers: a stdio server
  launched with an explicit `env` block inherits nothing, so each server carries its own variables.
  It also carries the atomic rescue settings, `RECALL_ATOMIC_RESCUE_MODE_BY_TENANT` (every tenant,
  `off` included), `RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT` and `RECALL_ATOMIC_RESCUE_WHEN_MISSING`, so a
  `recall generation promote` run in the project builds the artifact its server needs.
- **`CLAUDE.md`** and **`memory/MEMORY.md`** — block-scoped edits between markers, so your own file
  content is preserved.

Without `project_root` nothing is written or registered and the report says so. The wizard will not
guess which project these servers belong to.

## Atomic rescue

Atomic rescue finds a document whose answer is one sentence inside a long chunk. Each chunk is
split into overlapping 24 word views, the views are embedded, and an unscoped search moves at most
one parent chunk, the one whose best view matches the query most closely outside the dense top
five, to dense rank six before fusion, scored by its own cosine. Fusion can move it into the
final top five. It is on by default: the wizard builds each certified
generation's artifact after publishing its calibration and before promoting it, then registers
`RECALL_ATOMIC_RESCUE_MODE=active` and an absolute `RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT` on that
tenant's server. The report's `atomic rescue:` line under each corpus says what it built.

- **Where the artifacts go.** `<project_root>/.recall/atomic`. A `RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT`
  set in the shell that runs the wizard is used instead, and must be absolute. One root can serve
  several tenants.
- **With the Docker stack, a missing artifact is answered without it.** When `data_root` provisions
  the stack, the desktop app's uploads and calibrations run in those containers, which do not mount
  the host's artifact root, so they promote a generation, or publish a calibration, without an
  artifact. The wizard therefore also registers `RECALL_ATOMIC_RESCUE_WHEN_MISSING=skip` on the host
  servers: a search whose artifact is missing, for another calibration, or damaged answers exactly as
  with the rescue off and says why in `atomic_rescue_skipped`, and the server builds the artifact in
  the background (at most one build at a time across servers, retried after 60 s doubling to 600 s
  when it fails), then removes the artifacts of generations that cannot serve again. Everywhere else
  the setting stays `refuse`, today's behaviour. See
  [DESKTOP_ATOMIC_RESCUE_DESIGN.md](DESKTOP_ATOMIC_RESCUE_DESIGN.md).
- **Only a tenant this run built an artifact for is served active**, without the Docker stack. Every
  other server is pinned to `RECALL_ATOMIC_RESCUE_MODE=off`: the memory tenant has no generation, and
  a tenant left on an earlier generation may have no artifact; in active mode either would refuse
  every unscoped search. With the Docker stack (`skip`), only the memory tenant is pinned off.
- **A mode set in the shell is kept.** Run the wizard with `RECALL_ATOMIC_RESCUE_MODE=off` to leave
  the rescue off. A mistyped mode is refused before anything is built, because the server would
  refuse to start on it.
- **Without `numpy` and `threadpoolctl`** (the `atomic` extra), or with neither `project_root` nor
  a root in the shell, the rescue stays off and the report says why.
- **The Docker services themselves serve with the rescue off**, since they cannot reach the root;
  they serve only the desktop app, and the host servers build what their promotions leave missing.
- **A failed build refuses the promotion**, as a failed publish does: the generation is left
  unpromoted and released for `recall generation gc`.

What it costs: the first build embeds every view, about one per 12 words of the corpus, so it takes
longer than the index build; later builds embed only the views of chunk texts no earlier generation
held. **Not measured on PostgreSQL.** The one measurement is on the SQLite store with the default
local model on a loaded host (2026-11-15): 1,776 chunks took about 35 minutes to index and about two
hours to build the first artifact. With a hosted embedder the cost is the provider's, per view.

After install, `recall generation promote` and `rollback` and an upload through an MCP server
build the artifact before promotion when the rescue is active for the tenant. Publishing a new
calibration rebuilds the artifact after publication. If that rebuild fails, the calibration stays
published and applicable unscoped searches refuse until `recall generation rescue [generation-id]`
builds a matching artifact. **The CLI learns the mode from the project's `.env`**, so
run those commands from the project directory: from anywhere else the rescue reads as off, nothing
is built, and the server, which is active, refuses the promoted generation. See
[GENERATIONS.md](GENERATIONS.md#atomic-rescue).

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
installer wrote into the data folder, the desktop app's handoff file at `%APPDATA%/RE-call/runtime.json`
(outside the data folder, and only when it names *this* install), and the MCP registrations whose
`cwd` marks them as written by this install.
The wizard also removes its matching Codex MCP registrations from the project's
`.codex/config.toml`. Those entries must still match the installer's local ownership record
and this install's database. Unrelated or edited Codex entries and shared client hooks and skills
are kept.

For an install that uses an existing PostgreSQL database, pass the directory containing its
`wizard.json` as `--data-root`. This removes matching client registrations and keeps the database,
the configuration, and the corpus folders. `--purge-data` applies only to a wizard-managed volume.

Two things it leaves behind on purpose. It writes a timestamped backup of the MCP client config
(`~/.claude.json.recall-backup-<stamp>`) before editing it, and never deletes it: that file holds
every project the client tracks, and the edit is the one direction a mistake cannot be undone. And
if **anything** fails — most often because docker is not running — it **keeps** the stack file,
`wizard.json` and the handoff file, because those are the only things that name a stack which is
still up, and prints what went wrong. Start docker and run it again. That now includes a failure to
rewrite the MCP client config: before, only a container or volume failure kept them, so a failed
unregistration deleted the very files a retry needs.

`--purge-data` additionally removes every volume the stack declares that it also created, which
is where the built indexes live. A volume the stack marks `external:` was not created by this
install and is never removed, even with `--purge-data`. Off by default:
they are reproducible by re-indexing and expensive to rebuild, so whoever reinstalls next week and
whoever is reclaiming disk want opposite things.

## The Windows executable

GitHub releases do not ship this executable as a downloadable asset (none has, through 0.16.0).
Build it locally from the repository when you need a Python-free Windows bundle.

`packaging/recall-install.spec` builds a frozen bundle of the graphical installer, for somebody who
has no Python and is not going to get one.

```bash
python -m pip install -r .github/requirements/pyinstaller.txt
python -m pip install ".[desktop,mcp,fastembed,documents]"
cd packaging && python -m PyInstaller recall-install.spec --noconfirm --distpath ../dist
```

The MCP servers a frozen installer registers run `recall-install.exe --mcp-stdio` from the folder
the bundle was unpacked to, so that folder must stay where it is after the install; moving or
deleting it stops every registered server until the installer runs again. The installer says so
when it finishes.

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

The SignPath Foundation route needs a recognized open source licence. RE-call is licensed under
the AGPL-3.0, which is OSI approved, so the licence no longer blocks that route; its other
conditions must still be checked before applying. That is an application somebody has to make and be approved for; it
is not something the build can arrange for itself.

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
