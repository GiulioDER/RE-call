"""The supersession resolution rule: pure, DB-free, and shared by every reader of the graph.

Moved out of `recall.store` so the graph, state and evidence readers can use it without
loading the storage layer. `recall.store` re-exports every name.
"""

from __future__ import annotations

import posixpath
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from recall.frontmatter import supersedes_key, supersedes_targets, validity_bounds


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


def supersession_rows_sql(table: str, *, source_column: str, scope: str) -> str:
    """The supersession scan every PostgreSQL reader runs, one row per (file, claim, source).

    `scope` is the WHERE condition on the chunk alias ``c`` (tenant, and generation where the
    table has them). Rows are ``(file, supersedes, first_indexed, source, valid_from,
    valid_until)``, the shape `resolve_supersession_graph` reads: the source, because files are
    told apart by ROOT as well as path, and the validity bounds as written, because a successor
    hides its predecessor only inside its own window. `min`/`max` over a file's chunks, which
    share one frontmatter. `ORDER BY` is load-bearing (`PgVectorStore.supersession_all`).

    Every interpolated part is an identifier or a condition fixed by the calling store, never a
    request value; the tenant and generation stay bound parameters.
    """
    return f"""
        SELECT c.metadata->>'file' AS file,
               t.supersedes AS supersedes,
               -- COALESCE, mirroring the hit path's fallback: a row predating the column has no
               -- first write recorded, and its last write is the only evidence there is.
               min(COALESCE(c.first_indexed_at, c.indexed_at)) AS first_indexed,
               c.{source_column} AS source,
               min(c.metadata->>'valid_from') AS valid_from,
               max(c.metadata->>'valid_until') AS valid_until
        FROM {table} AS c
        CROSS JOIN LATERAL ({SUPERSEDES_TARGET_ROWS_SQL}) AS t
        WHERE {scope} AND c.metadata ? 'file'
        GROUP BY 1, 2, 4
        ORDER BY 1, 2, 4
    """  # noqa: S608


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

#: A document's validity window as `validity_bounds` reads it: tz-aware UTC, `valid_until`
#: inclusive to the end of its day, either bound None when open.
Window = tuple["datetime | None", "datetime | None"]

#: One scan row: ``(file, supersedes, first_indexed)`` and, from a store that reads them, the
#: chunk's absolute ``source`` and its raw ``valid_from`` and ``valid_until``. A three-element row
#: (a hand-built test, an older caller) has no root and no window.
SupersessionRow = Sequence[object]

#: Extensions a reference may omit, compared case-insensitively. Only the markdown family the
#: indexer reads as documents: stripping every extension would make `foo` name both `foo.md` and
#: `foo.py` in a code corpus, and `release-1.2` would lose its `.2`.
_DOCUMENT_EXTENSIONS = (".markdown", ".mdx", ".md")


def parse_window(valid_from: object, valid_until: object) -> Window | None:
    """`(start, end)` for a document's declared bounds, or None when they cannot be read.

    None is UNKNOWN, and an unknown window keeps the edge applying at every instant, which is the
    behaviour before windows were read: fail closed, because a successor whose window cannot be
    parsed is still a document that says the predecessor is stale. An inverted window (start
    after end) is malformed and therefore unknown too, rather than a window that never holds,
    which would quietly revive the predecessor.
    """
    try:
        start, end = validity_bounds({"valid_from": valid_from, "valid_until": valid_until})
    except (TypeError, ValueError):
        return None
    if start is not None and end is not None and start > end:
        return None
    return start, end


def window_contains(window: Window | None, instant: datetime) -> bool:
    """Whether `instant` lies in `window`; an unknown window (None) holds at every instant."""
    if window is None:
        return True
    start, end = window
    moment = instant if instant.tzinfo is not None else instant.replace(tzinfo=UTC)
    return (start is None or start <= moment) and (end is None or moment <= end)


def _fold(value: str) -> str:
    """How two spellings of one name are compared: NFC, then case-folded."""
    return unicodedata.normalize("NFC", value).casefold()


def _strip_document_extension(value: str) -> str:
    lowered = value.lower()
    for extension in _DOCUMENT_EXTENSIONS:
        if lowered.endswith(extension) and len(value) > len(extension):
            return value[: -len(extension)]
    return value


def reference_path_key(value: str) -> str:
    """A reference or a root-relative file as a folded path, for the exact-path match.

    Wikilink brackets and surrounding space are stripped, a backslash is read as a separator, a
    leading ``./`` or ``/`` is dropped and so is a markdown-family extension, so `[[a/Notes]]`,
    `a\\notes.md` and `./a/notes.MD` all name `a/notes.md`.
    """
    v = value.strip()
    while len(v) >= 2 and v[0] == "[" and v[-1] == "]":
        v = v[1:-1].strip()
    v = v.replace("\\", "/")
    while v.startswith("./"):
        v = v[2:]
    v = v.lstrip("/")
    normalised = posixpath.normpath(v) if v else v
    return _fold(_strip_document_extension(normalised if normalised != "." else ""))


def reference_stem_key(value: str) -> str:
    """The folded stem of a reference or file: its last path segment without the extension."""
    return reference_path_key(value).rsplit("/", 1)[-1]


def root_of(source: object, file: str) -> str | None:
    """The root a chunk was indexed under: its absolute `source` less the root-relative `file`.

    None when the row carries no source (no root is known, and every such file shares one).
    A source that does not end in its file (a URI with percent-escapes, a hand-built fixture)
    stands for its own root: two such files never share a root, which only matters when they
    also share a root-relative path, and then keeping them apart is the point.
    """
    if not isinstance(source, str) or not source:
        return None
    posix_source = source.replace("\\", "/")
    posix_file = file.replace("\\", "/")
    if posix_source == posix_file:
        return ""
    if posix_source.endswith("/" + posix_file):
        return posix_source[: -len(posix_file)]
    return posix_source


@dataclass(frozen=True)
class SupersessionGraph:
    """One resolution of a corpus's ``supersedes:`` claims, read the same way by every reader.

    `edges`, `unresolved` and `candidates` are what `supersession_all()` has always returned.
    `windows` holds each CLAIMANT's validity window, keyed like the edges, for the window rule: a
    successor hides its predecessor only while the successor itself is valid. A claimant missing
    from `windows` has an unknown window and its edges apply at every instant.

    Files are keyed by their root-relative path. A path that more than one ROOT holds (two
    projects' `README.md` in one tenant) is in `collided`, and each of those files is keyed by its
    absolute source instead, so the two stay two documents; `key` computes a hit's key either way.
    """

    edges: dict[str, str] = field(default_factory=dict)
    unresolved: frozenset[str] = frozenset()
    candidates: EdgeCandidates = field(default_factory=dict)
    windows: Mapping[str, Window] = field(default_factory=dict)
    collided: frozenset[str] = frozenset()
    #: claimant -> the references it declares that name several files (so resolve to none).
    ambiguous_claims: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_parts(
        cls,
        edges: Mapping[str, str],
        unresolved: frozenset[str] = frozenset(),
        candidates: EdgeCandidates | None = None,
        windows: Mapping[str, Window] | None = None,
    ) -> SupersessionGraph:
        """A graph from the three-part shape a store or a hand-built caller already has."""
        return cls(
            edges=dict(edges),
            unresolved=frozenset(unresolved),
            candidates=dict(candidates or {}),
            windows=dict(windows or {}),
        )

    def key(self, file: str, source: str | None = None) -> str:
        """The key a chunk with this root-relative `file` and absolute `source` is known by."""
        if file in self.collided and source:
            return source
        return file

    def parts(self) -> tuple[dict[str, str], frozenset[str], EdgeCandidates]:
        """``(edges, unresolved, candidates)``, copied, as `supersession_all()` returns them."""
        return (
            dict(self.edges),
            self.unresolved,
            {target: list(claims) for target, claims in self.candidates.items()},
        )

    def copy(self) -> SupersessionGraph:
        """A copy whose maps a caller may change without reaching a store's cache."""
        edges, unresolved, candidates = self.parts()
        return SupersessionGraph(
            edges,
            unresolved,
            candidates,
            dict(self.windows),
            self.collided,
            dict(self.ambiguous_claims),
        )

    def successor(
        self, key: str, now: datetime | None = None, known_as_of: datetime | None = None
    ) -> str | None:
        """What hides `key` at valid time `now` and transaction time `known_as_of`, or None."""
        return resolve_successor(
            key, self.edges, self.candidates, known_as_of, now=now, windows=self.windows
        )

    def successor_path(
        self, key: str, now: datetime | None = None, known_as_of: datetime | None = None
    ) -> tuple[str, ...]:
        """The chain from `key`'s first successor to the one `successor` names."""
        return successor_path(
            key, self.edges, self.candidates, known_as_of, now=now, windows=self.windows
        )

    def live_claimants(
        self, key: str, now: datetime | None = None, known_as_of: datetime | None = None
    ) -> tuple[str, ...]:
        """Every document claiming `key`, asserted by `known_as_of` and valid at `now`."""
        _primary, every = _live_claims(key, self.edges, self.candidates, known_as_of)
        if now is None:
            return tuple(every)
        return tuple(f for f in every if window_contains(self.windows.get(f), now))

    def on_cycle(self, key: str, known_as_of: datetime | None = None) -> bool:
        """Whether following `key`'s claims, the primary one at each step, comes back round."""
        seen = {key}
        cur = key
        while True:
            _primary, every = _live_claims(cur, self.edges, self.candidates, known_as_of)
            if not every:
                return False
            if every[0] in seen:
                return True
            seen.add(every[0])
            cur = every[0]


def as_supersession_graph(
    value: SupersessionGraph | Mapping[str, str],
    unresolved: frozenset[str] = frozenset(),
    candidates: EdgeCandidates | None = None,
) -> SupersessionGraph:
    """`value` itself when it is a graph (with `unresolved` added), else a graph of the parts."""
    if isinstance(value, SupersessionGraph):
        if not unresolved and candidates is None:
            return value
        return SupersessionGraph(
            value.edges,
            value.unresolved | frozenset(unresolved),
            value.candidates if candidates is None else candidates,
            value.windows,
            value.collided,
            value.ambiguous_claims,
        )
    return SupersessionGraph.from_parts(value, unresolved, candidates)


def _live_claims(
    cur: str,
    supersession: Mapping[str, str],
    edge_candidates: EdgeCandidates | None,
    known_as_of: datetime | None,
) -> tuple[str | None, list[str]]:
    """``(primary, every)``: the claim a chain follows from `cur`, and every live claimant.

    `primary` is `supersession[cur]` with no instant to replay, and with one the claim asserted
    LAST among those asserted by then (`resolve_successor` explains why). `every` is the primary
    first, then the other claimants in scan order, those asserted after `known_as_of` left out.
    A self-claim is no claim at any step: a document cannot supersede itself.
    """
    primary: str | None
    if known_as_of is None or edge_candidates is None:
        primary = supersession.get(cur)
    else:
        claims = edge_candidates.get(cur)
        if claims is None:
            # ABSENT from the map. Candidates and `supersession` come from one pass, so this means
            # hand-built, inconsistent input. Fail closed: keep demoting.
            primary = supersession.get(cur)
        else:
            live = [(f, when) for f, when in claims if f != cur and (when is None or when <= known_as_of)]
            primary = None
            best_when: datetime | None = None
            for f, when in live:
                # An undated claim is of unknown age and loses to any dated one, so a known
                # assertion decides the answer where one exists. Ties go to the later scan
                # position, which is the same rule `supersession`'s single winner uses.
                if primary is None or best_when is None or (when is not None and when >= best_when):
                    primary, best_when = f, when
    if primary == cur:
        primary = None
    every: list[str] = [primary] if primary is not None else []
    if edge_candidates is not None:
        for f, when in edge_candidates.get(cur, ()):
            if f == cur or f in every:
                continue
            if known_as_of is not None and when is not None and when > known_as_of:
                continue
            every.append(f)
    return primary, every


def successor_path(
    file: str,
    supersession: Mapping[str, str],
    edge_candidates: EdgeCandidates | None = None,
    known_as_of: datetime | None = None,
    *,
    now: datetime | None = None,
    windows: Mapping[str, Window] | None = None,
) -> tuple[str, ...]:
    """The successors from `file` to the one that hides it, or ``()`` when nothing does.

    **The window rule.** A successor hides its predecessor only while the successor is itself
    valid, at valid time `now`, by its own `valid_from` and `valid_until` in `windows`. So a
    successor dated from November leaves its predecessor current until then, a bounded successor
    that has expired gives the predecessor back, and a 2020 to 2022 correction leaves today
    alone. A claimant whose window is unknown (absent from `windows`) applies at every instant,
    which is the rule before windows were read, and so does every edge when `now` or `windows` is
    None.

    **Chains.** `file` is hidden at `now` when ANY document of its chain is valid then, and the
    one named is the DEEPEST valid member: in a -> b -> c with b temporary, `a` is hidden by `b`
    inside b's window, by `c` inside c's, and current in a gap between them. The chain followed is
    the primary claim at each step; only when no member of it is valid are the other claimants
    (fan-in) searched, breadth first, so the answer with every window unknown is exactly the one
    before windows: the chain's terminal successor, or its first member for a cycle.

    With `known_as_of`, only the claims asserted by then are followed, per step.
    """

    def valid(member: str) -> bool:
        if now is None or windows is None:
            return True
        return window_contains(windows.get(member), now)

    chain: list[str] = []
    seen = {file}
    cycle = False
    cur = file
    while True:
        nxt, _every = _live_claims(cur, supersession, edge_candidates, known_as_of)
        if nxt is None:
            break
        if nxt in seen:
            cycle = True
            break
        chain.append(nxt)
        seen.add(nxt)
        cur = nxt
    # A cycle answers its first member (the walk cannot pick a deepest), anything else its
    # deepest; each falls back to the next valid member in the same order.
    order = range(len(chain)) if cycle else range(len(chain) - 1, -1, -1)
    for index in order:
        if valid(chain[index]):
            return tuple(chain[: index + 1])
    # No member of the primary chain holds at `now`. Another claimant can (fan-in): search them
    # all, breadth first, and name the deepest valid one, the first found among equals.
    parents: dict[str, str] = {}
    visited = {file}
    frontier = [file]
    best: str | None = None
    best_depth = -1
    depth = 0
    while frontier:
        depth += 1
        following: list[str] = []
        for node in frontier:
            _primary, every = _live_claims(node, supersession, edge_candidates, known_as_of)
            for claimant in every:
                if claimant in visited:
                    continue
                visited.add(claimant)
                parents[claimant] = node
                following.append(claimant)
                if depth > best_depth and valid(claimant):
                    best, best_depth = claimant, depth
        frontier = following
    if best is None:
        return ()
    path = [best]
    while parents.get(path[-1], file) != file:
        path.append(parents[path[-1]])
    return tuple(reversed(path))


def resolve_successor(
    file: str,
    supersession: Mapping[str, str],
    edge_candidates: EdgeCandidates | None = None,
    known_as_of: datetime | None = None,
    *,
    now: datetime | None = None,
    windows: Mapping[str, Window] | None = None,
) -> str | None:
    """The successor that hides `file` at `now`, or None if none does; see `successor_path`.

    A cycle (a.md -> b.md -> a.md) cannot loop: the walk stops on the first revisit and the
    cycle member resolves to its direct successor. A self-claim (`supersedes:` the file's own
    name, an authoring mistake) is ignored at every step, not only the first: in a -> b -> c with
    c claiming itself, `a` resolves to `c`.

    With `known_as_of`, the walk sees only the edges asserted by that instant, so a point-in-time
    replay resolves to the successor that was current *then*. The filter is applied per STEP
    rather than to the final answer: in a chain a -> b -> c where only the first edge predates the
    instant, the answer is `b`. Gating the terminal successor instead would answer `c`, a document
    that had not yet superseded anything. Of several claims live at the instant, the step follows
    the one asserted LAST.

    ⚠️ **So asking about NOW two ways can give two answers, deliberately.** With fan-in where the
    later-asserted claim is not the last scan row, a plain search follows `supersession` and
    answers the scan-order winner, while `known_as_of=<any instant after both>` answers the
    latest-ASSERTED one. Pinned by `test_replay_at_now_may_differ_from_a_plain_search`.

    An edge with no recorded date applies, which is the inverse of the rule for hits with no
    recorded write time at all and deliberately so. Both are fail-closed: hiding a hit of unknown
    age would silently empty result sets for stores predating the column, while ignoring an edge
    of unknown age would serve a memory the corpus explicitly marks as stale.
    """
    path = successor_path(
        file, supersession, edge_candidates, known_as_of, now=now, windows=windows
    )
    return path[-1] if path else None


def _row_parts(
    row: SupersessionRow,
) -> tuple[str | None, str | None, datetime | None, str | None, object, object, bool]:
    """``(file, supersedes, first_indexed, source, valid_from, valid_until, has_window)``."""
    file, supersedes, first_indexed = row[0], row[1], row[2]
    source = row[3] if len(row) > 3 else None
    has_window = len(row) > 5
    valid_from = row[4] if has_window else None
    valid_until = row[5] if has_window else None
    return (
        file if isinstance(file, str) else None,
        supersedes if isinstance(supersedes, str) else None,
        first_indexed if isinstance(first_indexed, datetime) else None,
        source if isinstance(source, str) else None,
        valid_from,
        valid_until,
        has_window,
    )


def resolve_supersession_graph(rows: Sequence[SupersessionRow]) -> SupersessionGraph:
    """The one resolution pass every store and reader goes through; see `_resolve_rows`."""
    return _resolve_rows(rows)


def resolve_supersession_candidates(
    rows: Sequence[SupersessionRow],
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
    return _resolve_rows(rows).parts()


class ReferenceIndex:
    """The matching half of the resolver: every file keyed by root and path, and the rule that
    finds the file a `supersedes:` reference names (`_resolve_rows` states it).

    Built from ``(file, source)`` pairs. A root-relative `file` that two roots hold is in
    `collided`, and each of those is keyed by its `source`; every other file by its path.
    """

    def __init__(self, files: Iterable[tuple[str, str | None]]) -> None:
        pairs = list(files)
        roots: dict[str, set[str]] = {}
        for file, source in pairs:
            members = roots.setdefault(file, set())
            root = root_of(source, file)
            if root is not None:
                members.add(root)
        self.collided = frozenset(f for f, members in roots.items() if len(members) > 1)
        #: key -> (root-relative file, root, first source seen)
        self.nodes: dict[str, tuple[str, str | None, str | None]] = {}
        for file, source in pairs:
            self.nodes.setdefault(self.key(file, source), (file, root_of(source, file), source))
        self._by_path: dict[str, list[str]] = {}
        self._by_stem: dict[str, list[str]] = {}
        for key, (file, _root, _source) in self.nodes.items():
            self._by_path.setdefault(reference_path_key(file), []).append(key)
            self._by_stem.setdefault(reference_stem_key(file), []).append(key)

    def key(self, file: str, source: str | None) -> str:
        return source if file in self.collided and source else file

    def match(self, reference: str, claimant: str) -> tuple[str, list[str]]:
        """``("resolved" | "ambiguous" | "dangling" | "self", candidate keys)``."""
        claimant_root = self.nodes[claimant][1] if claimant in self.nodes else None
        stages = (
            ("path", self._by_path, reference_path_key(reference)),
            ("stem", self._by_stem, reference_stem_key(reference)),
        )
        for stage, index, folded in stages:
            if not folded:
                continue
            found = index.get(folded, [])
            others = [key for key in found if key != claimant]
            if not others:
                if found:
                    return ("self", []) if stage == "path" else ("dangling", [])
                continue
            if len(others) > 1:
                same_root = [key for key in others if self.nodes[key][1] == claimant_root]
                if len(same_root) == 1:
                    return "resolved", same_root
                return "ambiguous", others
            return "resolved", others
        return "dangling", []


def _resolve_rows(rows: Sequence[SupersessionRow]) -> SupersessionGraph:
    """The one resolution pass. Every store, `resolve_supersession` and every reader come here.

    Deriving the maps separately is what made an earlier version wrong twice: two functions
    matching claims to targets by their own copy of the rule can disagree, and did (a normalised
    lookup against a raw dangling key, and a per-file minimum against a per-claim group). One pass
    cannot.

    How a reference finds its target, in order:

    1. **The exact root-relative path**, folded (NFC, case-insensitive, a markdown-family
       extension optional): `a/notes.md` picks `a/notes.md` out of two `notes.md` files.
    2. **The stem** of the last segment, folded the same way.

    At either stage the claimant itself is no candidate (a self-claim is dropped here, so it can
    neither win last-row against a real claim nor make its own claimant ambiguous), and of several
    candidates the one in the claimant's own ROOT wins when it is the only one there. Several left
    is ambiguous: none is guessed at and all are named in `unresolved`. A reference whose path
    stage finds only the claimant is a self-claim; one whose stem stage does is DANGLING (a moved
    file naming its old location), keyed on the raw name as written.

    `winner` is last-row-wins over the rows in scan order, including the case where one file
    claims the same target twice, so a corpus whose names were already unique and exact sees the
    same winner it always did.
    """
    parsed = [_row_parts(row) for row in rows]
    index = ReferenceIndex((file, source) for file, _s, _w, source, _vf, _vu, _h in parsed if file)
    identity = index.key
    match = index.match
    collided = index.collided

    windows: dict[str, Window] = {}
    unknown: set[str] = set()
    winner: dict[str, str] = {}
    unresolved: set[str] = set()
    ambiguous_claims: dict[str, list[str]] = {}
    order: dict[str, list[str]] = {}
    when: dict[tuple[str, str], datetime | None] = {}

    for file, supersedes, first_indexed, source, valid_from, valid_until, has_window in parsed:
        if not file or not supersedes:
            continue
        claimant = identity(file, source)
        if has_window:
            window = parse_window(valid_from, valid_until)
            if window is None or windows.get(claimant, window) != window:
                # Unparseable, or two rows of one document disagreeing: unknown, so its edges
                # keep applying at every instant.
                unknown.add(claimant)
            else:
                windows[claimant] = window
        kind, found = match(supersedes, claimant)
        if kind == "self":
            continue
        if kind == "ambiguous":
            # Ambiguous: don't guess, but don't stay silent either. Dropping the edge alone would
            # leave the (possibly superseded) memories looking perfectly `ok`, which is the same
            # wrong answer the trust layer exists to prevent. Naming them lets the read path fail
            # closed and tell the operator what to fix.
            unresolved.update(found)
            claimed = ambiguous_claims.setdefault(claimant, [])
            if supersedes not in claimed:
                claimed.append(supersedes)
            continue
        if kind == "resolved":
            key = found[0]
        else:
            # Dangling: key on the raw name as written. Normalisation exists to make MATCHING
            # tolerant of how humans spell a reference; this key matches no indexed file either
            # way, so it keeps the author's form rather than inventing a normalised one.
            key = supersedes.replace("\\", "/").rsplit("/", 1)[-1]
            if key == claimant:
                continue
        winner[key] = claimant  # last row wins, exactly as before
        slot = order.setdefault(key, [])
        if claimant in slot:
            slot.remove(claimant)
        slot.append(claimant)
        pair = (key, claimant)
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
    # An open window holds at every instant, exactly like an unknown one, so only bounded windows
    # are kept: the map stays as small as the set of time-bounded successors.
    bounded = {
        key: window
        for key, window in windows.items()
        if key not in unknown and window != (None, None)
    }
    return SupersessionGraph(
        winner,
        frozenset(unresolved),
        candidates,
        bounded,
        collided,
        {claimant: tuple(refs) for claimant, refs in ambiguous_claims.items()},
    )


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
    winner, unresolved, _candidates = _resolve_rows([(f, s, None) for f, s in rows]).parts()
    return winner, unresolved
