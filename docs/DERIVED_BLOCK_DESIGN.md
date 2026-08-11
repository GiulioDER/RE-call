# Machine owned derived block

Date: 2026-08-11
Status: approved, not yet implemented

## Problem

RE-call's model of truth is authored frontmatter, and exactly three keys are recognised
(`recall/frontmatter.py:12`): `supersedes`, `valid_from`, `valid_until`. The trust layer acts on
those and nothing else.

The extraction work being built upstream produces `contradicts` and `same_entity` relations. Those
have nowhere to land. Frontmatter recognises three keys, and adding two more would put a machine's
inference into the same namespace a human authors, where the trust layer reads it as authored truth.

## The hazard this design exists to prevent

`content_hash` is computed over raw file bytes (`recall/index.py:518`). Writing a block changes the
bytes, so the file re-indexes. If the block is chunked, the next extraction pass reads its own prior
output as evidence and amplifies: a proposal becomes a citation for the next proposal, and the
corpus grows a self-referential belief no human ever stated.

So the block must be written into the file and simultaneously invisible to every path that reads a
document as evidence. That is the whole design.

## Decisions

### Placement: last in the body, after one blank line

Not a preference. `structure_chunks` computes offsets with `body.find(text, ...)`
(`recall/context.py:197`). If `human_body` is a strict prefix of `body`, every offset is identical
with or without the block, so `text_start` / `text_end` (`recall/index.py:600`) are invariant.
Prepending shifts every offset in every chunk of every file that gains a block.

End placement also keeps the block out of `document_title` (`recall/context.py:159`), which reads
only frontmatter and the first H1.

### Fences: HTML comments

Open `<!-- recall:derived v1 -->` alone on a line, close `<!-- /recall:derived -->` alone on a line.
Every markdown renderer hides them, and they can never be mistaken for frontmatter.

### Grammar

```
<!-- recall:derived v1 -->
contradicts: project_alpha_2026-03-02
  proposal: 3f9a...                       (64 hex)
  provider: recall.deterministic@session3-v1
  reviewer: giulio
  at: 2026-08-11T09:14:22Z
  note: both state a retention window and they differ
same_entity: project_alpha_v2_2026-06-01
  proposal: 8c21...
  provider: recall.deterministic@session3-v1
  reviewer: giulio
  at: 2026-08-11T09:14:22Z
status: superseded
  proposal: b104...
  provider: recall.deterministic@session3-v1
  reviewer: giulio
  at: 2026-08-11T09:14:22Z
digest: 5e7b...
<!-- /recall:derived -->
```

Entries are `<head>: <value>` with two-space-indented sub-keys. `proposal`, `provider`, `reviewer`
and `at` are required; `note` is optional. `digest:` is block level, unindented, and last.

Sub-key value shapes, so the refusal list below is unambiguous:

- `proposal` — exactly 64 lowercase hex characters (a proposal id is a content hash).
- `at` — a `Z`-suffixed UTC instant at whole-second precision, e.g. `2026-08-11T09:14:22Z`.
  `_is_utc_instant` parses with `strptime(value, "%Y-%m-%dT%H:%M:%SZ")`, which refuses fractional
  seconds (`2026-08-11T09:14:22.123456Z`) and refuses the `+00:00` offset form, so a value
  produced by this repo's own `datetime.now(timezone.utc).isoformat()` does not validate as is
  and must be formatted to the `Z`-suffixed shape first.
- `provider` — a non-empty single line. The `provider_id@revision` shape in the example is
  convention, not enforced; the digest covers the string verbatim either way.
- `reviewer` — a non-empty single line.
- `note` — a single line; everything after `note: ` is taken verbatim, colons included.

**Heads are `contradicts`, `same_entity`, `status` only.** `supersedes`, `valid_from` and
`valid_until` are forbidden here: they have frontmatter keys the trust layer reads, and a second
copy in the body is a second source of truth that can disagree with the first.

**`status` vocabulary is closed:** `open | adopted | closed | superseded | rejected | abandoned`.
It deliberately excludes `deprecated` and `obsolete`, which are in `CLOSURE_MARKERS`
(`recall/lint.py:46`) and would make the machine block trip the linter built to find prose closure.
Both normalise to `superseded` on the render path.

**Entries are sorted by `(head, value)`**, so a re-render is byte identical and a re-run never
churns. The three heads sort alphabetically, which is also their natural reading order.

**No block-level `generated_at`.** Timestamps are per entry and immutable. A block-level one churns
the file every run and puts a clock inside the digest.

### Digest over structure, not bytes

`digest:` is `canonical_sha256` (`recall/lineage.py:73`) over `{"v": 1, "entries": [...]}`.

Hashing raw bytes would report every CRLF checkout as tampered, and this repo reads `utf-8-sig` and
tolerates a BOM precisely because it lives on both Windows and Linux. The version lives inside the
hashed structure so a future v2 grammar cannot collide with a v1 digest.

### Refusal, never repair

Every rule in `parse_derived_block` is a refusal. The one apparent exception is not one: the render
path *accepts* `deprecated` / `obsolete` and normalises them to `superseded`, because that is a
proposal's vocabulary arriving at the boundary; the parse path *refuses* a file that literally
contains them, because that is a file claiming something the grammar does not permit. Accepting on
the way in and refusing on the way out is the same posture as `recall/fix.py:264` refusing to
overwrite what a human wrote.

### `content_hash` is left alone

A block write should re-index that file. Chunk text is byte identical because `_pack` strips every
block (`recall/index.py:150`), so embeddings serve from cache (`recall/cache.py:85`); the cost is
one `replace_sources`. Chunk ids and graph node ids are unaffected, which keeps evidence ids, and
therefore proposal ids, stable across a write.

## Module: `recall/derived_block.py`

Pure, no I/O.

| Function | Contract |
|---|---|
| `split_derived_block(body) -> DerivedSplit` | Total, never raises. `.human` is **always** a prefix of `body` — block or not, well-formed or not. `.block_text` is the tail from the first open fence, `""` otherwise. |
| `parse_derived_block(text) -> DerivedBlock` | Raises `DerivedBlockError`. Refusal only. |
| `render_derived_block(entries) -> str` | Sorts, normalises status aliases, computes the digest, returns text ending in the close fence and one `\n`. |
| `derived_digest(entries) -> str` | `canonical_sha256({"v": 1, "entries": [...]})`. |
| `verify_derived_block(text) -> DerivedBlock` | Parse, recompute, raise on mismatch. The function part (B) calls before touching a file. |
| `diagnose_derived_block(body) -> list[tuple[str, str]]` | The lint's view. Returns codes, raises nothing. |

### The `rstrip()`, and why the no-fence branch carries it

`human_body = body[:fence_start].rstrip()`, and `split_derived_block` rstrips on the **no-fence
branch too**. That second rstrip looks redundant and is the load-bearing half.

Trace the first write. Pre-write the body ends `"...adopted.\n"`. Post-write the block sits after
one blank line, so `body[:fence_start]` ends `"...adopted.\n\n"`. Both rstrip to `"...adopted."`.
Without the rstrip on the no-fence branch, the pre-write value is `"...adopted.\n"` and the
post-write value is `"...adopted."` — the extraction cache key changes on the first write and the
fixed point fails on iteration one, which will look exactly like model nondeterminism.

The cache at risk is the **extraction** cache, not the embedding one: `_pack` calls `b.strip()` on
every block, so chunk text is whitespace-invariant at block boundaries either way.

`s[:n].rstrip()` is still a prefix of `s`, so the rstrip does not cost the offset invariant.

**That is true of the prefix invariant only, not of the chunker contract.** Every body is now
rstripped, block or not — the no-fence branch's `.rstrip()` runs unconditionally. It is free for
`chunk_text` and `chunk_code` today only because `_pack` (`recall/index.py:151`) strips each block
before chunking, so a chunker that itself preserved trailing whitespace would never see the
difference. A future chunker that preserves trailing whitespace would silently change its output
for the entire corpus the day it lands, not just for files with a block. And in
`recall/generations.py`, chunk ids are a hash of the piece text, so that change would rotate every
markdown chunk id at once.

### Malformed blocks on the read path

`index.py` and `generations.py` do not run lint, so `split_derived_block` cannot raise: a corpus
that indexes today would start crashing on a file some tool half-wrote.

**The rule is one rule: strip from the first open fence to EOF, always.** Two blocks, an unclosed
fence, content after the close fence — all the same. This keeps `human_body` a strict prefix of
`body` in every case, so the offset invariance holds unconditionally rather than only on
well-formed files. A qualified guarantee ("identical offsets, unless the file is malformed") is the
kind that stops being checked.

The cost, stated plainly: prose a human appends *after* a block is not indexed. That case is
`derived-block-not-last`, an error, and its message carries the byte count being excluded from
retrieval, so the error names its own consequence rather than leaving it to be discovered.

## Seam: `recall/document.py`

```python
@dataclass(frozen=True)
class ParsedDocument:
    meta: dict[str, str]
    human_body: str    # frontmatter gone, derived block gone, rstripped
    derived_text: str  # the raw block, "" when absent

def parse_document(text: str) -> ParsedDocument
```

There are exactly six production callers of `parse_frontmatter`. All six migrate to
`parse_document`. Only `lint.py` reads `derived_text`; the other five never see it, and that is the
isolation.

| Site | Note |
|---|---|
| `recall/index.py:568` | `contextual_passages(raw, body, ...)` keeps taking the unstripped `raw` for `document_title`, which reads frontmatter and the first H1 — both above the block. The `body` argument becomes `human_body`. |
| `recall/generations.py:500` | Already inside the `media_type in {"text/markdown", ...}` branch, so non-markdown sources are untouched by construction. **Not optional:** `recall index` is refused under `RECALL_ENV=production` (`recall/cli.py:1209`), so hooking only the index path leaves the one build path that runs in production uncovered. |
| `recall/lint.py:124` | The only reader of `derived_text`. |
| `recall/check.py:53` | `_ANY_REF` over the body would otherwise hand the author the machine's own values back as `supersedes:` candidates. |
| `recall/semantic_lint.py:126` | Fixes the `is_closed_decision` collision: `_DECISION_STATUS` matches `status:\s*superseded`, which is exactly the shape of the block's own `status:` entry. This module reaches the corpus twice — here, and via `Indexer.index_path` at `:138`, which the `index.py` site covers. |

**The import ban.** A test asserts that within `recall/`, `recall_mcp/` and `recall_interop/`, the
only module importing `parse_frontmatter` is `recall/document.py`.
`benchmarks/check_temporal_inert.py` is on an explicit allowlist with its reason recorded: it
discards the body entirely and inspects only `meta`. This is the guard that makes a seventh call
site impossible to add silently.

## Lint codes

Four errors, not warnings, all sourced from `diagnose_derived_block` and sorted into the existing
`(level, file)` ordering. `lint.py`'s module docstring error list gains them.

| Code | Fires on |
|---|---|
| `derived-block-duplicated` | more than one open fence in the file |
| `derived-block-not-last` | non-whitespace after the close fence; message carries the excluded byte count |
| `derived-block-tampered` | the block parses, and the digest disagrees |
| `derived-block-malformed` | anything `parse_derived_block` refuses that is not a digest disagreement |

`derived-block-malformed` is an addition to the original brief, which named three codes. An
unclosed fence or a forbidden head has no digest to disagree with. Folding those into `tampered`
would report a half-written file as an integrity breach and send whoever reads it looking for an
attacker.

**One code per file, most specific first.** A second open fence is a parse refusal *and* a
structural one; `diagnose_derived_block` reports it as `derived-block-duplicated` only. The
precedence is `duplicated`, then `not-last`, then `tampered`, then `malformed`, and
`diagnose_derived_block` returns at most one code so a single mistake cannot be reported four ways.

## Refusals in `parse_derived_block`

One `pytest.raises(match=<field>)` each:

- head not in `contradicts | same_entity | status`
- head in `supersedes | valid_from | valid_until` — its own message, because this is the
  second-source-of-truth case and deserves to say so rather than read as an unknown key
- `status` value outside the closed vocabulary, with `deprecated` and `obsolete` named
- `contradicts` / `same_entity` value that is not already a bare stem, i.e. `supersedes_key(v) != v`
  — refuse the wikilink, do not unwrap it
- `proposal` not 64 lowercase hex characters
- `at` not a `Z`-suffixed UTC ISO-8601 instant
- `provider` or `reviewer` empty
- missing required sub-key, unknown sub-key, sub-key not two-space indented
- entries not in `(head, value)` order
- missing or malformed `digest:`
- zero entries — a block with nothing in it is pure churn; the writer removes the block instead
- unclosed fence, or a second open fence

Plus one non-over-rejection test: a fully populated well-formed block parses.

## Tests: `tests/test_derived_block_contract.py`

Flat, per house convention. No DB, so no `@requires_db` and no `make_store`; every property is
reachable with `tmp_path` at most. Module docstring enumerates the properties, one test per
property.

**Grammar and digest**

1. round trip — `parse(render(entries)).entries == entries`
2. re-render byte identical — `render(parse(render(e))) == render(e)`
3. entries sort by `(head, value)` regardless of input order
4. digest tamper detection — mutate one value, `verify_derived_block` raises
5. digest is over structure, not bytes — a CRLF variant and a BOM-prefixed variant produce the
   *same* digest
6. the refusal list above, one test each, plus the non-over-rejection test

**The `rstrip()` fixed point**

7. `test_rstrip_makes_human_body_a_fixed_point_on_the_first_write` — body ending `"...adopted.\n"`,
   take `human_body`, append a rendered block after one blank line, take `human_body` again, assert
   byte equality. Mutation: drop the `rstrip()` from the **no-fence branch alone** and watch it fail.
8. `human_body` is a prefix of `body`, parametrised over: no block, one block, two blocks,
   block-not-last, unclosed fence.

**Isolation**

9. a file with a block chunks identically to the same file without one (`chunk_text`, element-wise)
10. `text_start` / `text_end` identical across that same pair, through `contextual_passages`
11. `status: superseded` inside a block does not make `is_closed_decision` (`recall/semantic_lint.py:51`)
    return True. Paired assertion: the *same* string in ordinary prose still does, otherwise the
    test passes against a function that always returns False.
12. `recall check` on a file whose block names it does not offer the machine's own filename back as
    a `supersedes:` candidate
13. no production module outside `recall/document.py` imports `parse_frontmatter`

Every guard is mutated and watched go red before it is claimed to work. Test output is reported,
not assertions about test output.

## Scope

In: `recall/derived_block.py`, `recall/document.py`, the six call-site migrations, the four lint
codes, `tests/test_derived_block_contract.py`.

Out, deliberately:

- **No writer.** No `recall rewrite apply`, no `write_derived_block`. `verify_derived_block` is
  exported as the function the write path will call, and its refusal is pinned as a unit property
  of that function rather than as an end-to-end CLI test. A file writer is where the atomic-replace,
  encoding and CRLF questions live, and answering those badly is worse than deferring them.
- **No proposal plumbing.** Acceptance, reviewer identity and the promotion invariant
  (`recall/promotion.py:184`) are untouched. No proposal reaches corpus metadata without a named
  human, and nothing here changes that.

## Known costs

- `derived-block-not-last` drops a human's appended prose from retrieval. Mitigated only by the
  lint error naming its byte count.
- `verify_derived_block` refusing is a property of the function, not a demonstrated CLI refusal,
  until part (B) lands.
- **The code-fence example in this file's own Grammar section costs this file most of itself.**
  `_first_fence_offset` matches any line whose `.strip()` equals the literal `OPEN_FENCE` string,
  with no awareness of whether that line sits inside a markdown code fence or in real prose. The
  Grammar section above puts the literal fence text inside a ```` ``` ```` example so a reader can
  see it, and `_first_fence_offset` cannot tell that occurrence from a real one. This file is a
  live instance of the cost it describes: measured against the checked-in file, of 20394 total
  bytes `human_body` keeps 1949 and 18445 are cut, the read stopping mid-document inside the
  Grammar section's own code fence. `derived-block-not-last` (`recall/lint.py`) catches this file
  if it is linted, but `recall/index.py` and `recall/generations.py` do not run lint on the corpus
  they read, so at index time the loss is silent: the tail is simply gone from what gets embedded,
  with no error surfaced anywhere in that path. The lint message itself also understates the loss
  when it does fire: it reports only the byte count of the tail after the pseudo-block's own close
  fence (17860 bytes here), not the pseudo-block's own body between the fake open and close fence
  (a further 585 bytes that are just as unindexed but never named in the message).
  **This is the accepted behaviour, not a bug to fix.** Making `_first_fence_offset` fence-aware
  was considered and rejected: a detector that skips a fence line sitting inside an unbalanced
  code span in ordinary prose would let a REAL derived block hide from the strip instead, and a
  block that survives into the indexed body is silent amplification, the exact hazard this module
  exists to prevent (`recall/derived_block.py:12`) — a machine's own inference re-entering the
  corpus as evidence with nothing to flag it. The failure mode of the current rule is merely lost
  retrieval, and `derived-block-not-last` names that loss loudly whenever lint runs. Between a
  silent integrity failure and a loud availability one, the module is built to err toward
  stripping more, never less, and that is the correct tradeoff here even though it costs this file
  its own back half.

### For the write path

The digest binds the block's structure, not the document that contains it. `derived_digest` hashes
`{"v": 1, "entries": [...]}` and nothing in that structure names the file the block sits in. A
well-formed block copied verbatim from memo A into memo B verifies clean in memo B, because the
digest only checks that the entries hash to the digest they carry, while the block's `contradicts:`
/ `same_entity:` entries go on asserting relations about memo A. Harmless today, because nothing
yet consumes a block as evidence. But part (B)'s write path should bind the containing document's
stem into the hashed structure before any v1 block exists in the wild, because deciding to do that
after v1 blocks are already written costs a grammar version bump (`DERIVED_BLOCK_VERSION`) just to
retrofit an identity the digest should have carried from the start. This is a decision to make
before the writer lands, not a cost to defer past it.

## Invariants this design must not break

- No proposal reaches corpus metadata without a named human
  (`reviewed_promotion_is_trusted_metadata`, `recall/promotion.py:186`).
- The planner never calls a model; `max_model_calls` defaults to 0. Extraction runs on the ingest
  path, never the query path.
- Proposal ids are content hashes and `recall/reasoning_proposals/_providers.py:86` recomputes them,
  raising on mismatch. The model supplies semantics only; the library computes ids.
