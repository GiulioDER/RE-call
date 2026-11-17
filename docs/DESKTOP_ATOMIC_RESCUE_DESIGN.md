# Atomic rescue on desktop installs (the Docker stack)

Status: approved by the owner on 2026-11-17; not built yet.

## Why

The atomic rescue finds a memo whose answer is one sentence inside a long chunk. It is on by
default for a Lite install and for a PostgreSQL install that the wizard builds on the host
(`recall.lite.atomic.rescue_settings`, `recall.generation_rescue`). The owner decided on
2026-11-15 that it is on at first install everywhere, because it distinguishes RE-call from
competitors. A desktop install, where `data_root` provisions the Docker stack, is the one install
that still turns it off (`recall.wizard.headless._atomic_rescue_env`, its `off_reason`).

## How a desktop install works today

- The wizard builds, calibrates and promotes the first generations on the **host**, against the
  database's loopback port, and could build rescue artifacts there exactly as on PostgreSQL.
- The MCP servers that clients (Claude Code, Codex) search through are **host** processes,
  registered in local scope with the host DSN (`recall.wizard.wiring.mcp_config`).
- The desktop app (`recall.desktop`) is a host process. It reaches each tenant through
  `docker compose exec ... python -m recall_mcp.server`, so its uploads
  (`recall_mcp.desktop_ingest.generation_ingest`) and calibration publishes run **inside the tenant
  container**. There `RECALL_ATOMIC_RESCUE_MODE=off` is forced (`recall.wizard.stack.tenant_service`),
  so `ensure_for_promotion` and `after_publish` build nothing, and a new generation or calibration is
  promoted with no rescue artifact.
- A host server with the rescue `active` then refuses every unscoped search on the missing
  artifact (`recall.trust`, `load_atomic_rescue_artifact`). That refusal is why the wizard keeps
  the rescue off beside the Docker stack.
- No folder is shared between the containers and the host. A bind mount of the PostgreSQL data
  directory corrupted a database here (`recall.wizard.stack` module docstring), so sharing one is
  not an option this design uses.
- The desktop app never calls `recall_forget`. Forgets come from host servers or the CLI, which
  already erase under the configured root and every recorded one (`forget_roots`).

## Goals

1. A desktop install has the rescue on for every tenant the wizard builds, from first install.
2. After an upload or a recalibration made in the desktop app, search keeps answering throughout.
   While the rescue data for the new generation or calibration does not exist yet, search answers
   **without the rescue** and says so; once the data exists, the rescue applies again with no
   action from the user.
3. The host builds the missing data by itself.

## Non-goals

- Running the rescue inside the containers, or sharing a folder between containers and host.
- Persisting staged uploads across container recreation (a separate, pre-existing gap).
- Changing the desktop app's UI.

## Anti-goals (a failure even if it works)

- Any install other than a desktop one answering without the rescue when its artifact is missing.
  Everywhere else a missing artifact keeps refusing, as today: production deployments, the
  PostgreSQL CLI path and Lite are unchanged.
- A rescue that changes an answer's trust: the fallback serves exactly what the same search with
  the rescue `off` serves.
- Two host processes building the same artifact at once, or a build blocking a search.
- A failing build retried on every search.

## Approaches considered

The owner chose, on 2026-11-16, to answer without the rescue while its data is missing (rather
than to refuse until it exists, which would keep the container from promoting and make the host
build and promote).

1. **The host search server builds it (chosen).** A search that finds no artifact for its
   generation and calibration answers without the rescue and starts one background build. It
   covers every cause of a gap (a desktop upload, a recalibration, a CLI promote) and follows the
   pattern the Lite server already uses, where the server process builds artifacts after refreshes
   (`recall_mcp.lite_refresh`). Cost: a burst of CPU in a search server after the first search
   that follows an upload.
2. **The desktop app builds it** after its own upload or publish finishes, by running
   `recall generation rescue` on the host. Visible, and keeps search servers light, but misses
   every change the app did not make, and a build cut short by closing the app waits for the next
   upload.
3. **Both.** Two triggers to keep consistent, for little gain over 1.

## Design

### A setting that allows answering without the artifact

A new setting, `RECALL_ATOMIC_RESCUE_WHEN_MISSING`, with two values:

- `refuse` (the default, and today's behaviour): a missing or non-matching artifact raises
  `AtomicRescueArtifactError`, as now.
- `skip`: in `active` mode, when the artifact for the search's generation and calibration is
  missing or fails its lineage check, the search runs as if the rescue were `off` for that one
  search. It is reported, not silent: the response says the rescue was skipped and why (missing,
  or which lineage field did not match), alongside the existing `atomic_rescue` stage timing.

Only the absence of a usable artifact is skippable: missing, failing its lineage check, or damaged
(unreadable metadata, a digest mismatch, a partial directory). A damaged one is skipped and rebuilt
like a missing one, since `build_artifact` already replaces a damaged artifact (my call: refusing
forever on a damaged cache file would break goal 2). The report names which of the three it was;
the loader gives a missing and an unreadable manifest the same message, so the skip tells them
apart by whether the file exists. An invalid mode or placement, a root that is not absolute, a
store that cannot load parents, a search with no generation identity, or the `atomic` packages
missing still raise whatever the setting, because those are configuration errors a build cannot
fix.

### PostgreSQL artifacts keyed by generation and calibration (my call)

Today a PostgreSQL artifact is keyed by generation only, `<root>/<generation_id>/`
(`recall.trust`, `recall.generation_rescue.build_artifact`), and a recalibration replaces that
directory in place. On Windows that fails while any other host server (Claude Code, Codex, each
open session) still has the old matrix mapped: `build_artifact` raises "cannot be replaced while a
process has its matrix open", and nothing evicts the mapping. A recalibration in the desktop app
would then never recover until every server restarted, which breaks goal 2.

So a build writes `<root>/<generation_id>.<calibration_id>/`, as the Lite layout already keys by
both, and never replaces a directory another calibration's artifact occupies. Readers try that key
first and fall back to the old `<root>/<generation_id>/` with the same lineage check, so every
artifact built before this change (a production deployment's included) keeps serving with no rebuild. `remove_artifacts`,
gc, `erase_forgotten` and `forget_roots` handle both forms. The other options were rejected: evicting
a mismatched mapping in each server still fails builds until every server has searched once, and
"restart after a recalibration" fails goal 2.

### Removing superseded artifacts (my call)

On a desktop install nothing runs `recall generation gc` on the host, and the containers cannot
remove host artifacts (they have no root). Every upload promotes a new generation with a full-size
artifact, so without a step here the root grows with each upload. After a successful build the host
server removes this tenant's artifacts for generations that are neither active nor the most recent
retired one (the rollback target), best effort: a directory a process still has open stays and is
removed by a later build, as on Lite. Generation ids are global, so only directories whose
generation belongs to this tenant are touched, though tenants share the root.

The setting is read where the mode is read (`recall.trust`, from the same tenant-resolved
environment), validated at server start like the mode (`recall_mcp.settings`), and documented in
the settings table and in `docs/ENVIRONMENT_GENERATED.md` by the existing generator.

### The host server builds what is missing

When a search skips the rescue because the artifact is missing, the host server starts a build in
a background thread and returns the search result at once. The build is the existing
`recall.generation_rescue.build_artifact`, called with the server's own DSN, tenant, embedder and
root, so it is the same build `recall generation rescue` runs.

- **One build at a time per tenant and generation**, across processes: a lock file under the root,
  taken without waiting (the non-blocking OS lock `recall.lite.writer_lock` already implements for
  Windows and POSIX). A server that cannot take it does nothing; the holder's build serves both.
- **One attempt per key per server, then back off**: a failed build is logged and not retried for
  that `(generation_id, calibration_id)` until a backoff passes (same doubling backoff as the Lite
  auto-refresh, from 60 s to 600 s), so a broken build does not run on every search.
- **A build for a key no longer current is not started**: the key is the search's generation and
  calibration at the time of the search.
- A forget erases under the recorded roots as today (`build_artifact` records its root). Removal
  of superseded artifacts is the section above.
- Only the host MCP server does this. The CLI `recall search` with `skip` answers without the rescue
  and builds nothing.

### The wizard turns it on beside the Docker stack

On the `data_root` path `_atomic_rescue_env` stops forcing the rescue off. It registers, for the
host servers it wires:

- `RECALL_ATOMIC_RESCUE_MODE=active`
- `RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT=<project_root>/.recall/atomic` (as on the host PostgreSQL path)
- `RECALL_ATOMIC_RESCUE_WHEN_MISSING=skip`

The wizard builds the first artifacts on the host before promotion, as on PostgreSQL, so the first
search after install has the rescue. Tenant containers keep `MODE=off` and no root: they never
serve a client's search, and their promotions are what the host servers then catch up on. A mode
set in the wizard's shell still wins, as everywhere else. A non-Docker install keeps `refuse`.

`_with_rescue` currently pins every tenant without an artifact built in this run to `off`, because
`active` would refuse. With `skip`, on the desktop path that pin is dropped for tenants that have a
generation: they register `active` and build on first search. It stays for tenants with no
generation (the memory tenant), because `active` with no generation identity raises before any
artifact is looked up, and there is nothing to build for them. The `_atomic_rescue_env` docstring's
claim that the desktop app's forgets run in the containers is wrong (it never calls `recall_forget`)
and is corrected in the same change.

## Constraints

- No new PostgreSQL on the maintainer's server during the AML validation window; PostgreSQL tests run in CI, with red
  proof by a throwaway mutation commit.
- Every new behaviour test is red-proved (project `CLAUDE.md`).
- No new dependency: the desktop extra already carries `numpy` and `threadpoolctl` (PR 1335).
- Windows hosts: the lock and the artifact files must work with a running server holding a
  previous artifact's matrix mapped. That is why artifacts become keyed by generation and
  calibration and are never replaced in place (above).

## Left to the builder

- The exact names of the response fields that report a skipped rescue, following the existing
  diagnostics shape.
- Where the background build thread lives (likely beside the server's state, as the Lite
  refresher does), and how it is stopped at shutdown.
- The lock file's name and location under the root.
- Test layout, and which tests need PostgreSQL (CI only) versus a stubbed store.
