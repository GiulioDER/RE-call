"""RE-call's submitter-side adapters for the mem-bench benchmark.

**Prior work: this is a RELOCATION, not a new experiment.** These adapters already exist and have
already produced published figures — the RE-call rows on mem-bench's isolation and temporal axes.
Nothing here is measured for the first time. Context:
[[project-mem-bench-submission-pipe-2026-07-30]] (the submission pipeline),
[[project-mem-bench-adversarial-review-hardening-2026-07-31]] (what the axes can and cannot show),
[[feedback-verify-the-label-that-names-what-was-measured-2026-07-31]] (why `system_version` is now
read from the live package rather than typed).

`docs_search` could NOT be run: it is served by qwen-mcp on VPS2, whose root volume failed on
2026-07-31 ([[incident-vps2-filesystem-io-errors-2026-07-31]]). Recorded rather than skipped
silently — an unavailable check is not a passed one.

These live HERE, in RE-call's own repo, and deliberately not in mem-bench. mem-bench's `membench/`
package is stdlib-only and its entire "$0 to recompute" claim rests on staying that way; an adapter
that imports `recall` would break it on the first line. mem-bench's runners take
`--system module:Factory`, and that seam is exactly where a vendor's adapter is meant to live: in
the vendor's repo, versioned alongside the system it wraps.

Which is also why they were homeless until 2026-07-31. They existed only on VPS2 and in a session
scratchpad; when VPS2's disk died, one of the two copies went with it. **Code that is the only way
to reproduce a published figure belongs under version control.**
"""
