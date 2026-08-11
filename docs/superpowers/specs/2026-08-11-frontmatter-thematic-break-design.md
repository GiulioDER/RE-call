# Frontmatter versus thematic break

**Date:** 2026-08-11
**Branch:** `claude/silly-heyrovsky-1136f9`
**Status:** implemented. See the verification record at the end.

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

A block must contain **at least one** key line. Consequence: `---\n\n---` and `---\n---`, two
adjacent thematic breaks, are no longer paired. An empty block declares no metadata, so no `meta`
is lost by refusing it; only the two fence lines stay in the body.

Indented lines are accepted as block members because `recall/context.py` already documents that a
nested `title:` belongs to a sub-object, which means nested blocks are a real shape in these
corpora. An indented line that IS a key counts toward the at-least-one rule, so a block whose
every line is indented is not refused over its indentation alone.

#### Refusing is not symmetric with pairing

**This is the governing principle, and the first draft of this design did not have it.** Pairing a
block the old rule also paired is, at worst, no worse than the old rule. **Refusing** a block the
old rule accepted is strictly worse: the validity metadata is lost *and* the raw block is handed
to the chunker as prose. So the predicate must bias toward accepting, and every refusal has to
earn its place by naming the markdown shape it protects.

Two audit rounds found nine shapes of valid YAML frontmatter that the draft rules refused: column
0 block sequences, comments, quoted keys, digit leading keys, non-ASCII keys, wholly indented
blocks, multi-line flow collections closed at column 0, explicit key syntax, and unquoted keys
containing a space. All nine now parse.

#### Order is what separates YAML from markdown

`- archive` and `# Notes` are a sequence item and a comment, and a bullet and a heading, with
nothing in the text to tell them apart. **Order** tells them apart: a sequence belongs to the key
that opened it, so a column 0 `-` or `#` counts as a member only once a key has been seen.

```
---                        ---
tags:                      <blank>
- archive                  - first point
valid_until: 2020-01-01    - second point
---                        ---
frontmatter                a rule, a list, another rule
```

A bare key may contain spaces and may lead with a digit or a non-ASCII letter. What it may **not**
lead with is any character markdown uses to open a line: `#`, `-`, `*`, `+`, `>`, `|`, a backtick,
or `[`. None of those is a plausible unquoted YAML key, and excluding them is what stops
`**Warning**: text`, `[spec]: https://example.com`, `` `config`: x `` and `> quoted: x` from
reading as keys.

#### What this does not fix

The obvious reading of the rule above is more generous than the truth, so this is stated flatly
and asserted in tests.

**One key unlocks the rest of the block.** After any key, every comment and sequence item is
accepted. A prose section led by `Note: ...` and followed by a heading and a bullet list is still
paired and still deleted. That is identical to the old behaviour, so it is not a regression and
`legacy_pairing_differs` is correctly `False`.

**The bar for "key shaped" is low, and that is the price of the paragraph above.** Spaces are
allowed inside a key so `date created:` parses, which means *any* sentence with a colon anywhere
in it is a key. So is a bare `http://example.com`, and so is a line opening `:` or `?`, both of
which render as ordinary paragraphs.

So what *is* fixed is narrower than it first looks: a section whose first non-blank line is a
heading, a bullet, a blockquote, a link reference definition, a table row, or a sentence **with no
colon in it**. That is the reported defect and the common shape, and each of those is asserted in
a test.

Two shapes of valid YAML remain refused, both recorded rather than fixed:

- A `#` comment **before** the first key. At that position it cannot be told apart from
  the markdown heading in the reported defect, and a heading right after a rule is much the more
  common document. A comment after a key is accepted.
- `%YAML 1.2` refuses the block. A YAML directive belongs before the opening fence, not inside
  it, so this shape is malformed anyway.

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

**Two index paths, not one.** The design as first written assumed `recall/index.py` was the only
freshness guard. `recall/generations.py::_reuse_source` is a second one: it copies an earlier
generation's chunks whenever tenant, URI, object sha256 and pipeline fingerprint all match, and it
returns *before* `parse_frontmatter` runs. None of those four terms moved, and `PipelineIdentity`
covers only the schema version, embedder, chunker and FTS configuration. An object indexed before
the fix would therefore carry its truncated chunk set into every generation built afterwards.
`_body_rule_changed` gates reuse on the same narrow trigger.

**The trigger covers the title, not only the body.** An unclosed block carrying a `title:` key
moves no body, because neither rule ever paired it, but `document_title` used to scan an unclosed
block to its end and take the title out of it. The title is embedded into every passage in
`section` and `neighbor` mode, so a body-only trigger would pin a stale title permanently. The
same applies to a document containing U+2028, a form feed, a vertical tab or NEL: `document_title`
split with `splitlines()`, which honours those, while the span is counted over `split("\n")`,
which does not, so the two scans could address different lines. Flagged on the presence of the
separator alone, which is cheap and exact.

**`_body_rule_changed` does not self-heal, and that is a known cost.** `_body_derivation_hash`
stores its perturbed fingerprint, so an affected file rebuilds once. The generations guard is a
pure function of the object's text with nowhere to record that the rebuild happened, because
reuse is keyed on tenant, URI, sha256 and pipeline fingerprint. An affected object is therefore
re-chunked on every future generation build. Accepted because the cost falls only on objects that
contain the defect, and removing it means either a new `PipelineIdentity` term (which re-embeds
every corpus) or a body rule column threaded through `_reuse_source`'s SELECT. Recorded as a
follow up rather than claimed as symmetric.

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

## Verification record

Run on 2026-08-11, on this branch.

| Check | Result |
| --- | --- |
| Full suite | 3171 passed, 517 skipped, 1 xfailed, exit 0 |
| `python -m ruff check` | All checks passed (ruff 0.16.2) |
| Old versus new body diff, 140 `.md` files | 0 bodies moved, 0 meta changed, 0 flagged for re-index |
| YAML shape sweep | every shape swept parses except `%YAML 1.2` and a `#` comment before the first key |
| Markdown prose sweep | every section swept correctly left in the body |
| Guard mutation | 19 of 19 mutations produced a failure |
| Adversarial audit | three rounds, 11 findings, all acted on |

Four things the table does not say on its own.

**The 516 skips are the `requires_db` tests.** No PostgreSQL is configured in this worktree. The
index skip guard was verified through `_body_derivation_hash` as a pure function in both
directions, and the generations reuse guard through `_body_rule_changed`, also pure.
`test_a_thematic_break_object_is_never_reused_into_a_new_generation` covers the call site, and it
is the assertion that dies if the guard is deleted, but it is `@requires_db` and **did not execute
here**. It runs for the first time in CI.

**Zero moved bodies is the absence of a regression, not evidence the fix works.** None of this
repository's 140 markdown files opens with a thematic break, so the corpus diff proves only that
no existing chunk boundary shifts. The evidence that the defect is fixed is the test suite, and
the diff harness was itself checked against a synthetic affected document before its zero was
believed.

**Mutation testing found a guard that asserted nothing.** The ordering rule that separates a YAML
sequence from a markdown bullet list survived being deleted, because the bullet list test was
passing on the at-least-one-key rule instead. Two tests were added in which a later line supplies
the key, so the ordering rule is now the only thing refusing the block. This is the reason the
mutation pass exists and it is worth recording that it paid.

**The audit is what made this correct, and it cost three rounds.** The first found that the draft
predicate refused six shapes of valid YAML. The second, after the repair, found three more plus an
overclaiming docstring. The third found a hole in the separator guard the second round had just
prompted (its fence precondition split one way while the divergence it models comes from the
other, so the separator could skip the check that existed for it), a second overclaiming
docstring, and a false claim in a test comment. Every round produced concrete failing inputs, and
every finding was verified against the real code before being acted on.

The lesson to carry forward is the asymmetry principle above: a rule that decides whether to parse
something must be judged on what it *refuses*, not only on what it accepts. The second lesson is
that a fix prompted by a review needs the same scrutiny as the original code, since round three's
main finding was a defect introduced by round two's repair.

The 17 mutations covered: the at-least-one-key rule, the key line test in both directions, the
markdown lead-in exclusions, the key pattern narrowed back to ASCII-letter-leading, indented
continuation handling, an indented key counting as a key, sequence items and comments rejected
outright, the ordering rule dropped, explicit key syntax, the flow collection closer, the exotic
separator trigger, the title-only trigger, the media type check in `_body_rule_changed`,
`legacy_pairing_differs` never flagging, `_body_derivation_hash` flagging everything, and both
rewired call sites reverted to scanning forward for the next fence. Each was applied to the real
source, run, and reverted.

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
