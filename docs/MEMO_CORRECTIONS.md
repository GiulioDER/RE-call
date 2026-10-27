# Reliable corrections through the assistant surface

The existing `recall_remember` tool accepts three optional parameters. Calls that omit them retain
the exact legacy `{memo_id}` success reply and raising refresh failures. One check is new for them
too: a `supersedes` target that is not an existing memo file (`target_missing`) or already has a
successor (`stale_target`) is refused on every call.

* `request_id`: an opaque nonempty string, at most 128 characters. Reuse it only for the same
  text, fact, kind, dates, supersession target and revision precondition. The memo folder journals
  its reserved memo id and request hash before publication. A retry finishes the original write
  and refreshes that memo instead of allocating another one. Reusing an id for different intent
  returns `conflict="request_mismatch"`. Request metadata contains hashes and ids, never memo text.
* `expected_revision`: the lowercase SHA-256 digest of the indexed memo representation. Pass it
  together with `supersedes`. Search hits, including the compact assistant view, carry `revision`
  from the indexed source's content hash, never a fresh filesystem read beside stale search text.
  Sources without an indexed content hash report null. A changed target returns
  `conflict="revision_mismatch"` before writing anything.
  Markdown uses the existing index normalization: one leading UTF-8 BOM and NUL characters are
  removed, and CRLF/CR newlines become LF. Representation-only changes do not change this revision.
* `include_receipt`: request a detailed reply without an idempotency key. A request id or revision
  precondition also enables receipts automatically.

```json
{
  "memo_id": "pref-0043",
  "receipt": {
    "revision": "<SHA-256 of the indexed memo representation>",
    "supersedes": "pref-0042",
    "write_state": "written",
    "replayed": false,
    "index_state": "indexed",
    "calibration_status": "certified",
    "retrieval_ready": true
  }
}
```

`retrieval_ready` describes indexing plus the store's calibration readiness. It does not establish
that this memo answers a query, is currently valid, or passes that query's other trust checks.
Every read still passes through trusted search. A small store can report a written, indexed memo
while its calibration is missing and `retrieval_ready` is false. This adds no onboarding mode.

A failed refresh on a receipt-enabled call returns the persisted memo id with
`index_state="failed"`, unavailable calibration readiness, and an exception type without its
message. Retry with the same request id to finish indexing. A failure to inspect calibration
preserves successful indexing and reports unavailable readiness. Persistence failures still raise.

Only an existing assistant-authored memo can be a correction target. A memo with an authored
successor is refused as `stale_target`, including on legacy calls. The write lane checks this
before installing a correction, so two clients of the same SQLite file cannot install competing
corrections. Idempotent replays retain the original id even if that memo subsequently acquired a
successor; a replay never removes supersession metadata or makes the old memo current again.

A completed request's edited or removed memo is refused rather than overwritten or recreated.
Replay checks raw bytes independently of the indexed revision, so even a representation-only edit
is protected. Explicit forgetting durably cancels request reservations before unlinking, including
an interrupted request whose file was published but whose final journal update failed. If cancellation
cannot be saved, forgetting raises and leaves the file present. A memo that was published but never
indexed (a crash after publication with no refresh since) is forgotten the same way, rather than refused.
A prepared request without a published
memo can finish after rechecking correction preconditions; use explicit forgetting for deletion, since
a direct filesystem deletion in that interrupted state cannot be distinguished from non-publication.
The authorship ledger retains reservations and request identities when legacy calls write new memos.

Forgetting also drops every hash the journal derived from the memo's bytes, since an unsalted hash of a
short value in a known template can be reversed by enumeration. The memo's own requests keep only their
memo id, state and time, and every retry of one is refused as `memo_missing`, whatever its text. A
correction that named the memo as its target loses the hash of its revision precondition and keeps a hash
of its own content: its retry is still replayed when the reply was lost, and an unfinished one is refused
as `target_missing`.

The same erasure follows a memo that leaves the index any other way: `recall forget` and the
`recall_forget` tool erase the hashes of every memo they forget, whether or not its file is still on
disk, and the next index (the server's refresh, `recall lite refresh`, `recall index`) that prunes a
memo deleted by hand erases them before it deletes the rows. A memo deleted by hand that was never
indexed is not pruned, so its hashes stay until it is forgotten by name.

A reservation is not authorship. The ledger records the assistant as author only once its file
exists, so a file somebody else puts in the folder under a reserved name is never claimed: the
write, or the retry of an interrupted one, moves to the next free id and leaves that file alone.
A memo is published with a hard link where the filesystem has them, and with an exclusive create
otherwise (FAT32 and exFAT, many network shares); neither replaces an existing file. A memo of the
folder that cannot be read, or is not UTF-8, does not block a correction; one that declares a
successor in another encoding still counts.

The request journal is bounded. A prepared request is kept until it completes or is forgotten. A
completed or cancelled request is kept for 30 days and at most 1,000 of them, the newest first;
past that its request id is treated as new. Retry within that window.

Old memos are retained for historical inspection. This workflow does not infer supersession from
prose or rewrite imported notes. For those corrections, use the existing `recall_report_stale`
review queue and the dashboard or reviewed rewrite commands. No migration or model call is added.
