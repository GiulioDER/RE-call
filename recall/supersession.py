"""The supersession resolution rule: pure, DB-free, and shared by every reader of the graph.

Moved out of `recall.store` so the graph, state and evidence readers can use it without
loading the storage layer. `recall.store` re-exports every name.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from recall.frontmatter import supersedes_key, supersedes_targets


def chunk_supersedes_targets(metadata: Mapping[str, object]) -> tuple[str, ...]:
    """The references a chunk's metadata declares, read as `SUPERSEDES_TARGET_ROWS_SQL` reads them.

    Every Python reader of chunk metadata goes through this, so a fallback path cannot see an
    edge the store's scan does not, or the reverse. Compiled AML records declare RECORD ids, not
    file references, and are read as declaring none, here and in the SQL; see there for why.
    """
    if metadata.get("record_type") == "compiled":
        return ()
    return supersedes_targets(metadata.get("supersedes"))


#: One row per declared `supersedes` reference, for a chunk row aliased ``c``. Used with
#: ``CROSS JOIN LATERAL (...) AS t`` and selected as ``t.supersedes``.
#:
#: A markdown memo declaring several references stores them as a JSON ARRAY (one reference is
#: still a plain string, see `recall.frontmatter._supersedes_value`), and ``->>`` on an array
#: returns its JSON text, which resolved as ONE dangling target named ``["a.md", "b.md"]``.
#: Arrays are therefore expanded to one row each, and a row with no reference, or an empty
#: array, still yields one row with NULL, because every file must reach the resolver.
#:
#: Compiled AML records (``record_type = 'compiled'``) declare NOTHING here: each yields one
#: NULL row, exactly as `chunk_supersedes_targets` reads them. They store an array under this
#: key too, of RECORD ids rather than file references, and `recall_aml` resolves those itself
#: (`PgVectorStore.explicit_superseded_chunk_ids`). Two other readings were rejected:
#:
#: - ``->>`` text, which they had until this was written, made every compiled row a dangling
#:   claim on a target named ``[]`` or ``["mem_…"]``. `supersedes_key` keeps the quotes, so
#:   these never matched a file, but they filled `supersession_all`'s edges and candidates, and
#:   the store-backed reasoning graph reported edges its own fallback did not see.
#: - Expanding the array would make those references LIVE, because a compiled record's file is
#:   ``{chunk_id}.md`` and the ids name chunks. That changes the trust layer's verdicts for an
#:   AML table, which is a measured decision about `recall_aml`'s semantics, not a parser fix.
SUPERSEDES_TARGET_ROWS_SQL = """
    SELECT jsonb_array_elements_text(c.metadata->'supersedes') AS supersedes
    WHERE COALESCE(jsonb_typeof(c.metadata->'supersedes'), '') = 'array'
      AND COALESCE(c.metadata->>'record_type', '') <> 'compiled'
    UNION ALL
    SELECT CASE
             WHEN COALESCE(c.metadata->>'record_type', '') = 'compiled' THEN NULL
             WHEN COALESCE(jsonb_typeof(c.metadata->'supersedes'), '') = 'array' THEN NULL
             ELSE c.metadata->>'supersedes'
           END
    WHERE CASE
            WHEN COALESCE(c.metadata->>'record_type', '') = 'compiled' THEN true
            WHEN COALESCE(jsonb_typeof(c.metadata->'supersedes'), '') = 'array'
            THEN jsonb_array_length(c.metadata->'supersedes') = 0
            ELSE true
          END
"""


def _basename(file: str) -> str:
    """Stem of a root-relative (posix) file identifier — the key a `supersedes:` target resolves
    to. A stem rather than a full basename so `name`, `name.md`, `[name]` and `[[name]]` all
    designate the same document; see `supersedes_key`."""
    return supersedes_key(file)


#: A superseded file -> every document claiming to supersede it, in scan order, each with when
#: that claim was first written (``None`` when unknown). A LIST rather than one winner, because
#: choosing a single superseder discards the information a point-in-time replay needs: where two
#: documents supersede the same target, the one live at a past instant is often not the one live
#: today, and picking by any time-independent rule answers a different question.
EdgeCandidates = dict[str, list[tuple[str, "datetime | None"]]]


def resolve_supersession_candidates(
    rows: list[tuple[str | None, str | None, datetime | None]],
) -> tuple[dict[str, str], frozenset[str], EdgeCandidates]:
    """``(winner, unresolved, candidates)`` from ``(file, supersedes, first_indexed)`` rows.

    An edge ``A -> B`` becomes assertable when B is written, because the claim lives in B's
    `supersedes:` frontmatter. So it is dated by the EARLIEST `indexed_at` among the chunks of B
    that CARRY that claim: a chunk existing implies the frontmatter existed, and the earliest is
    the conservative reading. Per claim, not per file, because one file can carry several
    different `supersedes` values without any authoring mistake (`Indexer._prune_vanished` notes a
    corpus may be indexed under several roots, and `metadata['file']` is root-relative while
    `replace_sources` deletes by absolute `source`).

    ⚠️ **Known limit: this dates the CHUNK's first write, not the CLAIM's.** Chunk ids are derived
    from the file path, so editing a memo preserves them and `replace_sources` restores the
    original `first_indexed_at`. Adding a `supersedes:` line to a memo that already existed
    therefore back-dates the new edge to that memo's CREATION, and a replay between the two
    reports `superseded` at a moment the claim had not been made. Dating the claim rather than the
    row needs a per-(file, supersedes) first-seen, which this column is the wrong shape to carry:
    `first_indexed_at` answers "when did this row appear", which is the right input for the hit
    path and an approximation for the edge path. Stated rather than left to be discovered.

    `winner` is what `supersession()` has always returned and is unchanged. `candidates` is the
    superset a replay needs; see `recall.trust.resolve_successor` for how it is consumed.

    Dates come from `first_indexed_at`, the FIRST write, preserved across re-indexing. Using
    `indexed_at` (the last write) meant editing a superseding memo re-dated its edge, so a past
    replay dropped a long-standing claim and served the stale memory as current.

    Pure and DB-free, so the rule is unit-testable without a database.
    """
    winner, unresolved, candidates = _resolve_rows(rows)
    return winner, unresolved, candidates


def _resolve_rows(
    rows: list[tuple[str | None, str | None, datetime | None]],
) -> tuple[dict[str, str], frozenset[str], EdgeCandidates]:
    """The one resolution pass. `resolve_supersession` and the candidate map both come from here.

    Deriving them separately is what made the previous version wrong twice: two functions matching
    claims to targets by their own copy of the rule can disagree, and did (a normalised lookup
    against a raw dangling key, and a per-file minimum against a per-claim group). One pass cannot.

    `winner` reproduces last-row-wins EXACTLY, including the case where one file claims the same
    target twice, so callers who never ask for a past instant see no change at all.
    """
    # DEDUPED. `rows` carries one entry per (file, supersedes) pair, so a file asserting two
    # different claims appeared TWICE and made ITSELF read as an ambiguous basename: its own
    # incoming edge was dropped and it was named in `unresolved`, telling the operator to
    # disambiguate a basename that exactly one document carries. Ambiguity is a property of two
    # FILES sharing a stem, never of one file carrying two claims.
    files = list(dict.fromkeys(f for f, _s, _d in rows if f))
    by_base: dict[str, list[str]] = {}
    for f in files:
        by_base.setdefault(_basename(f), []).append(f)

    winner: dict[str, str] = {}
    unresolved: set[str] = set()
    order: dict[str, list[str]] = {}
    when: dict[tuple[str, str], datetime | None] = {}

    for file, supersedes, first_indexed in rows:
        if not file or not supersedes:
            continue
        target_basename = _basename(supersedes)
        matches = by_base.get(target_basename, [])
        if len(matches) == 1:
            key = matches[0]
        elif len(matches) == 0:
            # Dangling: key on the raw basename as written. Normalisation exists to make MATCHING
            # tolerant of how humans spell a reference; this key matches no real file either way,
            # so it keeps the author's form rather than inventing a normalised one.
            key = supersedes.rsplit("/", 1)[-1]
        else:
            # Ambiguous: don't guess — but don't stay silent either. Dropping the edge alone
            # would leave the (possibly superseded) memories looking perfectly `ok`, which is
            # the same wrong answer the trust layer exists to prevent. Naming them lets the
            # read path fail closed and tell the operator what to fix.
            unresolved.update(matches)
            continue

        winner[key] = file  # last row wins, exactly as before
        slot = order.setdefault(key, [])
        if file in slot:
            slot.remove(file)
        slot.append(file)
        pair = (key, file)
        if pair in when:
            prev = when[pair]
            # An undated row makes the whole claim undated. Fail closed: unknown age keeps
            # demoting rather than silently reviving a memory the corpus marks as stale.
            when[pair] = (
                min(prev, first_indexed) if prev is not None and first_indexed is not None else None
            )
        else:
            when[pair] = first_indexed

    candidates: EdgeCandidates = {
        target: [(f, when[(target, f)]) for f in claimants] for target, claimants in order.items()
    }
    return winner, frozenset(unresolved), candidates


def resolve_supersession(
    rows: list[tuple[str | None, str | None]],
) -> tuple[dict[str, str], frozenset[str]]:
    """Build the superseded -> superseding map from ``(file, supersedes)`` rows.

    ``file`` is a root-relative path; ``supersedes`` references its target by basename (the
    authoring convention). Three cases:

    - **Unambiguous** (exactly one indexed file bears that basename): resolve to its
      root-relative path. This is the fix for the original bug — a naive basename key would
      have collided with an unrelated same-named file in another directory.
    - **Dangling** (no indexed file bears that basename — the predecessor was never indexed,
      or was deleted): fall back to the raw basename as the key. There is nothing to
      disambiguate, so this cannot mis-map; dropping it would just as silently discard a valid
      supersession claim (e.g. a memo intentionally superseding a doc that was since removed).
    - **Ambiguous** (two or more indexed files share that basename): do not guess — a silent
      mis-map to the wrong file is worse than a broken chain, since we cannot tell which one the
      author meant. The candidates are returned in ``unresolved`` so the read path can fail
      closed on them; dropping the edge and saying nothing would leave a possibly-superseded
      memory looking perfectly healthy.

    Both keys and values in the mapping are root-relative paths (or a bare basename for the
    dangling case). ``unresolved`` holds root-relative paths.

    Pure and DB-free so the resolution rule can be unit-tested without a database.

    A thin projection of `resolve_supersession_candidates`, deliberately: the resolution rule
    lives in exactly one place so the winner map and the candidate map cannot drift apart.
    """
    winner, unresolved, _candidates = _resolve_rows([(f, s, None) for f, s in rows])
    return winner, unresolved
