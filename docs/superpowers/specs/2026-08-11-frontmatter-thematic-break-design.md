# Frontmatter versus thematic break

**Date:** 2026-08-11
**Branch:** `claude/silly-heyrovsky-1136f9`
**Status:** design approved, implementation not started

## The defect

`recall/frontmatter.py::parse_frontmatter` decides that a document carries a frontmatter block by
testing two things: line 0 is `---`, and some later line is `---`. Markdown's thematic break is
also `---`. A memo that opens with a horizontal rule and contains a second rule anywhere later is
therefore parsed as a frontmatter block, and everything between the two rules is deleted from the
returned body with no signal.

Reproduced on this branch:

```python
from recall.frontmatter import parse_frontmatter
text = "---\n\n# Release notes\n\nThis release supersedes archive_policy_2026-01-05.md.\n\n---\n\nContact ops.\n"
parse_frontmatter(text)
# ({}, 'Contact ops.\n')
```

The whole first section is gone. `meta` is empty, so nothing was gained in exchange.

### It is not confined to one function

Three call sites independently re-implement the same open test, and all three misfire on the same
document.

| Site | Behavior on a leading thematic break |
| --- | --- |
| `recall/frontmatter.py::parse_frontmatter` | first section silently deleted from the body |
| `recall/context.py::document_title` | lifts a `title:` line out of *prose* as the document title |
| `recall/fix.py::apply_proposal` | **writes** `supersedes:` into the middle of the author's prose |

The third is a write path and is the most damaging. Verified:

```python
apply_proposal(d, Proposal(edit_file="notes.md", target="archive_policy.md", ...))
# '---\n\n# Release notes\n\nOld policy retired.\n\nsupersedes: archive_policy.md\n---\n\nContact ops.\n'
```

`recall lint --fix` modifies the memo on disk, the inserted key then *does* parse as frontmatter,
and the entire prose section drops out of retrieval as a side effect.

`document_title` verified on the same shape:

```python
t = "---\n\nSome prose.\n\ntitle: not a real title\n\n---\n\n# Real Heading\n"
document_title(t, parse_frontmatter(t)[1], "notes.md")
# 'not a real title'
```

### Corrections to the originating report

Neither changes the direction of the work, both are recorded so the trail is accurate.

1. `recall/truth_extraction/` does not exist on this branch. It lives on
   `claude/suspicious-ardinghelli-3d3e31`.
2. `_KEY_LINE_RE` does not exist on that branch either. `refuse_unclosed_frontmatter` there uses an
   inline `line.split(":", 1)[0].strip() == key` test against `VALIDITY_KEYS`. The idea the report
   describes is present; the named regex is not.

## Design

### 1. One predicate, three consumers

New in `recall/frontmatter.py`:

```python
_KEY_LINE = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]*\s*:")

def frontmatter_span(text: str) -> int | None:
    """Index of the closing fence line, or None when the leading `---` is a thematic break."""
```

Line 0 must be `---` after tolerating a BOM, exactly as today. Then walk from line 1:

| Line | Action |
| --- | --- |
| `---` | close the block and return its index |
| blank | continue |
| starts with whitespace | continue, a YAML continuation or sub-object member |
| matches `_KEY_LINE` at column 0 | continue, and count it |
| anything else | return `None`, this is not a frontmatter block |
| end of input, no fence | return `None`, unclosed behavior unchanged |

A block must contain **at least one** column 0 key line. Consequence: `---\n\n---` and `---\n---`,
two adjacent thematic breaks, are no longer paired. An empty block declares no metadata, so no
`meta` is lost by refusing it; only the two fence lines stay in the body.

Indented lines are accepted as block members because `recall/context.py` already documents that a
nested `title:` belongs to a sub-object, which means nested blocks are a real shape in these
corpora. A rule stricter than this would leak them into retrieved bodies.

#### Residual ambiguity, stated rather than solved

These remain paired as frontmatter and will be asserted as such in tests, so the boundary is
explicit instead of assumed away:

- `---\nNote: something\n---`. A one line prose block whose first word is followed by a colon is
  genuinely indistinguishable from YAML without a real parser.
- `---\nhttp://example.com\n---`. A bare URL at column 0 matches the key shape.

Both require the *entire* candidate block to consist of such lines. A rule followed by a heading, a
sentence, or a bullet list is no longer paired, which is the reported failure.

Rejected alternative: additionally requiring the block to declare a known key
(`valid_from` / `valid_until` / `supersedes` / `title`). It nearly eliminates false pairing, but a
block carrying only other keys, such as the `color: blue` in the existing test document, would leak
into the body. That changes retrieval for legitimately frontmattered documents, which is a worse
trade than the residual above.

### 2. Call site changes

**`parse_frontmatter`.** Span `None` returns `({}, text)`. Otherwise keys are extracted from
`lines[1:span]` by the existing logic, with the `VALIDITY_KEYS` filter and the quote stripping
unchanged, and the body is `lines[span + 1:]` with leading newlines stripped, as today.

**`context.document_title`.** Replaces its inline scan with `frontmatter_span`. It keeps its own
indented key skip. The predicate accepts an indented line as a block *member*; the title lookup must
still refuse it as *the title*. Those are two different questions and stay separate.

**`fix.apply_proposal`.** Span `None` takes the path the function already has for a file with no
frontmatter: prepend `---`, the key, `---` above the existing content. A memo opening with a rule
gets a valid block above the rule, and the rule survives as the first line of the body. The prose is
never touched.

### 3. Targeted migration

`recall/index.py` computes `content_hash` over the raw file bytes and folds it into
`_index_fingerprint`. A corpus whose files have not changed is skipped on the next run, so the
corrected body would never reach an existing index.

Adding an unconditional term to `_index_fingerprint` changes the joined string for every file and
re-embeds every corpus, charging a full re-embed to every user for a bug most corpora never hit.
Instead, perturb the `content_hash` input **only** for a file whose pairing actually changed: the
old rule paired it and the new rule does not. Unaffected files hash bit-identically to today and
keep skipping. The raw text is already read before the skip check, so the test costs one string
scan.

This requires a helper that retains the old rule:

```python
def legacy_pairing_differs(text: str) -> bool:
    """True when the pre-2026-08-11 rule paired a block that `frontmatter_span` now refuses."""
```

It stays permanently and is documented as load bearing for index freshness, not as dead code.
Deleting it later would revert those files' hashes and cost a second re-index, so it is not a
temporary shim and must not be labelled one.

### 4. Tests

Every new guard is mutated and observed failing before it is claimed to work.

**`tests/test_frontmatter.py`**
- the reproduction above, asserting the body is returned whole
- rule, then heading, then rule
- rule, then bullet list, then rule
- `Note:` and bare URL residuals, asserted as still paired
- nested block, still parsed, keys still extracted
- empty block `---\n---` and `---\n\n---`, both returned as body
- existing tests unchanged, including the `color: blue` document and the BOM case

**`tests/test_context_modes.py`**, which is where `document_title` is currently covered
- title not lifted from prose between two rules
- indented `title:` still skipped
- real frontmatter title still found

**`tests/test_fix.py`**
- `apply_proposal` on a thematic break document prepends a block, prose byte identical
- `apply_proposal` on real frontmatter unchanged byte for byte

**`tests/test_context_modes_index.py`**, which is where the skip guard is currently covered. This is
the guard most likely to be written so that it cannot fail:
- a file whose pairing changed re-indexes even though the file did not change
- a file whose pairing did not change still skips

### 5. Verification

1. Full suite, backgrounded. It takes about 12 minutes locally, not the 3 the pyproject claims.
2. `python -m ruff check`. Not `ruff format`, which is not this repo's convention.
3. Old parser versus new parser over every `.md` in the repository, reporting exactly which files'
   bodies move. This list is reported before merge, not after.

### 6. Documentation

- The `recall/frontmatter.py` module docstring states the pairing rule, since it currently says only
  that a document "may begin with a `---` line".
- `docs/ENTERPRISE_RETRIEVAL.md` title precedence row is reviewed against the new predicate.

## Recommended follow up, not in this scope

`parse_frontmatter` reads an **indented** validity key as top level metadata:

```python
parse_frontmatter("---\nsome_object:\n  valid_from: 2026-01-01\n---\nbody")
# ({'valid_from': '2026-01-01'}, 'body')
```

`key.strip()` discards the indentation that is the only thing distinguishing a top level key from a
sub-object member. This is the identical defect `recall/context.py::document_title` documents having
fixed for `title:`, and it is inconsistent with a predicate that now relies on that distinction to
decide what a block member is.

It is left out because fixing it changes `meta` for any corpus that declares validity keys under a
mapping, which is a separate regression surface from the one this work opens. It needs its own
before and after corpus check.
