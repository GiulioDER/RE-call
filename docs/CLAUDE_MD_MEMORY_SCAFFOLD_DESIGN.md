# CLAUDE.md and memory/MEMORY.md scaffolding in `recall setup`

Date: 2026-08-12
Status: implemented (commits c4b027e..a9c802c on claude/recall-config-setup-f241a6)

## Problem

`recall setup` (`recall/setup.py`, wired into `recall/cli.py`) configures the DSN,
embedder, reranker, sparse backend, entailment judge and optional calibration, then
writes all of it into a delimited block in `.env`. That is the entire scope of the
wizard today: a grep across `recall/`, `recall_mcp/` and `recall_interop/` turns up no
code that reads, writes, or scaffolds `CLAUDE.md` or any `MEMORY.md`/`memory/`
convention.

That is a real gap. Installing recall and registering the MCP server
(`docs/USING_WITH_CLAUDE.md`) gives Claude the *tools*
(`recall_search`, `recall_evidence`, `recall_index`, `recall_forget`, `recall_stats`),
but nothing tells a fresh project's Claude *when* to reach for them or *how* to write
new facts back into memory. `docs/CASE_STUDY.md` documents the two-tier convention
(`MEMORY.md` as an always-loaded index, `memory/*.md` as one-file-per-fact with
frontmatter) that recall's own team used to build recall, but that convention lives
only as prose in a case study. A new, especially beginner, user gets a working search
tool and no scaffolding that makes Claude actually use it well. This is worse than a
missing feature: it means the product's real benefit depends on documentation the user
has to independently discover and hand-copy.

## Goal

Extend `recall setup` so that, by default, it also scaffolds a `CLAUDE.md` section
telling Claude how to use recall and how to write memory files, plus a starter
`memory/MEMORY.md`, and indexes that memory directory immediately so it is searchable
before the user's first real turn with Claude.

## Design

### Where it runs

A new step inside `run_setup_wizard` (`recall/setup.py`), placed after the existing
embedder/reranker/sparse/entailment prompts and before the calibration question. It
must run on every path through the wizard, including the early-return path taken when
calibration is skipped (the `if not queries_raw or not corpus_raw:` branch), since that
branch currently `return`s before reaching the shared tail at the bottom of the function. Concretely,
this means moving the new step above that `if _ask_yes_no(...)` calibration block, not
adding it after, so both exit paths pass through it.

The step is interactive, matching every other step in the wizard:

```
Scaffold CLAUDE.md and a memory/ directory for this project? [Y/n]
```

Default yes (`_ask_yes_no(..., default=True)`). Declining skips the entire step; no
files are created or modified, no indexing happens.

### CLAUDE.md handling

Reuses the `.env` block pattern (`SETUP_BEGIN`/`SETUP_END`, `_update_env_block` in
`recall/setup.py:961-977`) but adapted to Markdown, since HTML comments are inert in
rendered Markdown and won't corrupt the file's structure:

```
<!-- recall setup begin -->
...block content...
<!-- recall setup end -->
```

A new helper, e.g. `_update_markdown_block(path, begin, end, content)`, mirrors
`_update_env_block`'s logic:

- File does not exist: create it containing only the block.
- File exists, markers not present: append the block at the end (blank line separator
  if the file doesn't already end in one, matching `_update_env_block`'s handling of
  `.env`).
- File exists, markers present (a re-run of the wizard): replace only the content
  between the markers, leaving the rest of the file untouched.

Block content is instructional text for Claude, derived from the tool docstrings
already shipped in `recall_mcp/server.py` (`recall_search` at line ~805,
`recall_evidence` at line ~864), so the guidance in CLAUDE.md doesn't drift from what
the tools actually say:

- Call `recall_search` before proposing an idea, forming a hypothesis, or repeating
  past work; if a closed decision or falsified hypothesis surfaces, don't re-litigate
  it.
- Respect `abstained`: when true, say "I don't know" rather than answering from
  degraded hits.
- Use `recall_evidence` instead of `recall_search` when about to answer from memory
  rather than just consult it; every citation must be a `chunk_id` from `items`.
- Write new durable facts to `memory/`, one file per fact, indexed by `MEMORY.md`,
  following the frontmatter convention described in the scaffolded `MEMORY.md` itself
  (see below), so `recall_index` can pick them up.

### memory/MEMORY.md scaffold

- Only created if `memory/MEMORY.md` does not already exist. Never overwritten on a
  re-run, since by the second run it may hold real user facts.
- Creates the `memory/` directory (if missing) and a starter `MEMORY.md` containing:
  - A short explanation that this file is the always-loaded index; individual facts
    live in sibling files under `memory/`.
  - The frontmatter convention: `name` (kebab-case slug), `description` (one-line
    relevance summary), `metadata.type` one of `user | feedback | project | reference`.
  - The one-line-per-fact index format (`- [Title](file.md) — hook`) with an example
    row pointing at a placeholder `memory/example.md` OR, more simply, no example row
    at all, just the format description, so the scaffold doesn't ship a fake fact that
    has to be manually deleted. (Decision: no example row, keep the starter file
    genuinely empty of fabricated content.)

This mirrors the convention `docs/CASE_STUDY.md` already documents, just made
reusable as a template instead of one-off prose.

### Auto-index

After scaffolding, the wizard indexes `memory/` into the DB immediately, using the
same `Indexer`/`PgVectorStore` path `recall index` uses (`recall/cli.py:2365-2380`),
so `recall_search` can find the new memory files without the user needing to run
`recall index` by hand first.

One existing constraint applies: `recall/cli.py` blocks local filesystem indexing
outright when `RECALL_ENV=production` ("local filesystem indexing is
development-only; build from an immutable S3 manifest in production"). The new step
respects this: it always scaffolds the files, but checks `RECALL_ENV` first and skips
the indexing call in production, printing why (e.g. "Skipping auto-index: RECALL_ENV
is production. Index memory/ via your production build pipeline.").

If indexing fails for any other reason (e.g. DB unreachable), the wizard prints the
error and continues rather than aborting the whole setup run over a step that already
defaulted to best-effort.

### Non-goals

- No new standalone CLI command. This is a wizard step only, matching the "extend
  `recall setup`" decision; a user who wants to redo just this step re-runs
  `recall setup` (accepting defaults for the earlier prompts) or edits the files by
  hand per the new docs section.
- No modification of an existing, non-empty `memory/MEMORY.md`. The wizard only ever
  creates it when absent.
- No attempt to detect or merge conflicting prior CLAUDE.md content outside the
  delimited block; anything outside the markers is the user's own and is left alone.

## Docs

- New section in `docs/USING_WITH_CLAUDE.md`, e.g. "Configure your project's memory
  files", describing what the wizard writes, the CLAUDE.md block's content, the
  MEMORY.md convention, and how to redo either by hand or by re-running
  `recall setup`.
- A row added to `docs/README.md`'s product-path table and
  `docs/REPOSITORY_MAP.md`'s reader-paths table pointing at that section.

## Testing

- `_update_markdown_block` unit tests: create-when-absent, append-when-no-markers,
  replace-in-place-when-markers-present, and a stability check that content outside
  the markers survives untouched.
- `memory/MEMORY.md` already exists → left byte-for-byte alone, no error.
- `RECALL_ENV=production` → CLAUDE.md and memory/MEMORY.md are still scaffolded, the
  index call is skipped, and the skip message is printed.
- Wizard declines the new prompt → no files created or modified, no index call made.
- Indexing failure (e.g. store unreachable) → scaffolding already on disk is kept,
  wizard prints the error and continues rather than raising.
