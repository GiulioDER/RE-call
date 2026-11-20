# Design

How the provider works, for a reader of the code. [README.md](../README.md) says what it does for a user.

- **Process.** `ReCallMemoryProvider.initialize()` starts RE-call's `recall-mcp` as a child process
  (`mcp_client.StdioMcpClient`: newline-delimited JSON-RPC over stdio, answers matched to requests by id)
  and waits until the MCP handshake completes, up to `START_TIMEOUT_S`. The child receives an allowlisted
  environment (`config.PASSTHROUGH_ENV` plus the variables `Settings.child_env` sets) and the
  provider's data folder as its working directory. A dead child is restarted at most once per
  `RESTART_INTERVAL_S`.
- **Store.** With no `dsn` and no `memo_folder`, `config.load_settings` takes the default Lite store from `config._store_dir`:
  the shared per-user directory `config.shared_dir()` (`<XDG_CONFIG_HOME, else ~/.config>/re-call/`) when
  its `memory.db` exists, else the profile's old `<hermes_home>/re-call/memory.db` when that exists (one
  log line per process says how to move it), else the shared directory. The owner's decision of
  2026-11-13 is "One shared Lite store", so two Hermes profiles share one memory unless one sets `dsn`.
  `Settings.data_dir` stays `<hermes_home>/re-call/` whatever the store: `config.json`, `mirror.json`,
  the embedding cache, `server.log` and the child's working directory are per profile. The shared
  store and `<shared>/memos` go together: a custom `memo_folder` without `dsn` keeps the profile's
  store, a `dsn` naming the shared store defaults to `<shared>/memos`, a `memo_folder` naming it gets
  the shared store, and `Settings.problem()` refuses either paired with anything else. The mirror
  journal records `store`, the store file's volume and file number (`provider._store_identity`), when
  it is bound, rewriting only that binding (other fields stay as read); a start that cannot read the
  identity keeps the recorded one. When the destination path changed, or the recorded identity differs, the journal is
  kept only if `provider._held_memos` finds that no mapped memo id names a memo (read as RE-call 0.15.0
  writes it) holding other text, and at least one is found. RE-call numbers memos in sequence, so
  another store reuses the ids; a file number can be reused once freed, so it is never proof alone.
  A mapping whose memo is not read at that check moves to the journal's `unconfirmed`, which
  `_apply` uses for an entry only after reading its memo holding the entry's text, and drops once
  the memo is read holding other text.
  One case stays unchecked: the store and its memos recreated at the same path, on a filesystem that
  gave the new store file the old one's file number.
- **Prefetch.** Each non-trivial user turn calls the server's `recall_search` within `PREFETCH_BUDGET_S`.
  `render.render_search` shows the hits RE-call lists as citable, flags superseded and expired ones as
  retired, and states an abstention when RE-call abstained.
- **Tools.** `surface.py` vendors RE-call's six assistant tools with RE-call's own names and schemas. The
  write tools are offered only to the primary agent on a Lite store; a subagent gets the read tools, and a
  non-Lite store gets `config.SEARCH_TOOLS`, the tools its server registers.
- **Mirror.** For the primary agent on a Lite store with `mirror_builtin` on, Hermes's built-in memory
  writes are journaled in `mirror.json` before anything is sent: `memos` maps an entry key (target and
  SHA-256 of the stripped text) to its memo id, `pending` holds the writes in order, `submitted` the ones
  handed to the server, and `destination` the store and assistant memo folder the journal belongs to. A write refused before it left the process is unclaimed and replayed; a write
  whose outcome is unknown stays claimed and blocks later mirrored writes until the journal is reconciled.
