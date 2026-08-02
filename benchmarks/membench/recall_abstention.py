"""RE-call as a mem-bench axis-1 (abstention) system.

Prior work searched 2026-08-02, `docs_search(source_type="memory")`, VPS2 back up and the corpus
serving again (`gap_warning: false`, top cosine 0.63). Four hits bear on this file and two of them
are binding:

- [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals measured, all
  failed. **Do not re-litigate by tuning the threshold or swapping in a bigger judge.** This adapter
  therefore reports the library's shipped decision verbatim and tunes nothing.
- [[project-mem-bench-submission-pipe-2026-07-30]] — "the declared decision rule is the spine":
  sweeping the threshold on ONE unchanged response file moves the verdict from FAIL (0.50) to PASS
  (0.85), nothing else changing. That is precisely why the rule is declared in the header and
  re-derived by the verifier rather than chosen here.
- [[project-recall-nearmiss-signal-exhaustion-2026-07-29]] — **the 0.50 floor is provably inert on
  bge-small**, and `top_cosine` reads a document that is not the evidence 63% of the time. So the
  `score` column this adapter records is a weak instrument on the default arm, and a defaults run
  should be EXPECTED to abstain rarely. That is a prediction to check against, not a defect to fix
  here.
- [[project-membench-leaderboard-redesign-2026-08-01]] — the full 1200-instance abstention manifest
  separates nothing (whole spread 0.0079 against a 0.010 band), and the two arenas disagree about
  which config leads. Read any new axis-1 figures against that before calling a difference a result.

SUBMITTER-SIDE CODE, for the same reason its two siblings are: it imports `recall`, and
mem-bench's `membench/` package is stdlib-only because the whole $0-verification claim rests on
that. The runner's `--system module:Factory` seam is what lets this live here.

**Written 2026-08-02, and its absence is why it exists.** Axis 1 carried six rows on the board, more
than the other two axes combined, and nothing in either repository could produce one. The adapter
that made them was never under version control, so no published abstention figure could be
regenerated, bisected against a library version, or re-scored on a corrected manifest. This closes
that for axis 1; `recall_isolation.py`'s own history is the cautionary case, its predecessor having
died with a failed volume.

**`ingest` REPLACES, and that is the whole axis.** Isolation adds tenant after tenant because there
must be something to leak from. Here the manifest removes a question's evidence and asks whether the
system notices; a system that retained anything from the previous instance answers from evidence
this axis believes it deleted, and scores a beautifully shaped curve of our harness failing to
excise. `membench.axes.abstention_run.assert_slice_delivered` checks that after every single
ingest, so a mistake here aborts the run instead of publishing.

Replacement is done by REUSING ONE WORK DIRECTORY and letting `Indexer.index_path` prune what is no
longer on disk, rather than by dropping the table. A fresh `mkdtemp` per call is the known trap:
`replace_sources` keys on the resolved source path, so a new directory INSERTS instead of replacing.
That exact bug produced 840 rows for 420 documents on the temporal axis, every fact present twice,
and it was caught by counting rows rather than by reading a score.

Run it:

    python -m membench.axes.abstention_run \\
        --system benchmarks.membench.recall_abstention:RecallAbstention \\
        --out artifact.jsonl --submitter github:you --system-name RE-call \\
        --run-started <ISO> --config '{"embedder": "BAAI/bge-small-en-v1.5", "dim": 384}' \\
        --decision-rule '{"kind":"threshold","field":"score","op":"<","value":0.5}'

**The declared `decision_rule` must match the library's actual floor.** `trusted_search` makes its
own abstention decision and this adapter reports it verbatim; the rule in the header is what
mem-bench re-derives that decision against, row by row. Declaring a rule the library does not
implement is caught by `membench.decision.rederive_abstentions` as a contradiction on every row it
disagrees with, which is the point of declaring it at all.

`--system-version` is deliberately absent: `system_version` below reports the live package, and
mem-bench aborts if a flag contradicts it.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

from membench.axes.abstention.adapter import Document, Response

from benchmarks.membench import _env
from recall.index import Indexer
from recall.store import PgVectorStore
from recall.trust import trusted_search

DSN = os.environ.get("MEMBENCH_DSN", "postgresql:///membench_abs")
TABLE = os.environ.get("MEMBENCH_TABLE", "abs_chunks")
TENANT = os.environ.get("MEMBENCH_TENANT", "membench-abstention")


class RecallAbstention:
    name = "RE-call"

    @property
    def system_version(self) -> str:
        """Read from the live `recall` package -- see `_env.system_version`."""
        return _env.system_version()

    def __init__(self) -> None:
        cfg = _env.from_env()
        self._embedder = cfg.embedder
        self._rr = cfg.reranker
        self._k = cfg.k
        self._ck = cfg.candidate_k
        self._store = PgVectorStore(DSN, dim=self._embedder.dim, tenant=TENANT, table=TABLE)
        self._store.ensure_schema()
        # ONE directory for this adapter's whole life. See the module docstring: a per-call
        # mkdtemp is the double-indexing bug, not a tidiness preference.
        self._work = Path(tempfile.mkdtemp(prefix="membench-abs-"))
        self._indexer = Indexer(self._store, self._embedder)

    def ingest(self, docs: Iterable[Document]) -> None:
        """Replace the corpus with `docs`, retaining nothing from the previous call.

        The directory is emptied and rewritten, then re-indexed. `Indexer.index_path` prunes rows
        whose file has vanished from under the root, so a document excised for this instance loses
        its rows here rather than lingering to be retrieved. Emptying the directory is what makes
        that pruning fire; writing the new slice on top of the old one would leave every previously
        seen document on disk and therefore in the index.
        """
        for stale in self._work.iterdir():
            if stale.is_file():
                stale.unlink()
            else:
                shutil.rmtree(stale)
        for d in docs:
            # The doc_id IS the filename stem, so `indexed_doc_ids`, `cited_ids` and the manifest
            # all speak one vocabulary. LOCOMO ids contain `/` and `:` (e.g. `conv-26/D3:13`),
            # neither of which is a legal Windows filename character and the first of which would
            # silently create a subdirectory on POSIX -- so the id is encoded here and decoded in
            # `_doc_id_of`. Round-tripped rather than sanitised: a lossy mapping would collide two
            # ids into one file and silently drop a document from the slice.
            (self._work / f"{_encode(d.doc_id)}.md").write_text(d.text, encoding="utf-8")
        self._indexer.index_path(self._work)

    def indexed_doc_ids(self) -> frozenset[str]:
        """Every doc id currently retrievable. Read by the runner's invariant-1 check after every
        ingest, which is the only thing standing between a caching bug and a published curve."""
        out: set[str] = set()
        for chunk in self._store.iter_chunks():
            did = _doc_id_of(chunk.metadata.get("file"))
            if did is not None:
                out.add(did)
        return frozenset(out)

    def query(self, question: str) -> Response:
        """The library's OWN abstention decision, reported verbatim.

        `answer=None` is the abstention, and `trusted_search` decides it: `abstained` is True when
        no hit earned verdict `ok`. Nothing here re-derives it from a threshold. An adapter that
        applied its own cutoff would be scoring a decision rule this benchmark invented for RE-call
        rather than the one RE-call ships, and the whole axis would measure the adapter.

        The answer STRING is a placeholder and is never scored: axis 1 has no judge, and
        `membench/axes/abstention/score.py` says so in its own field name (`answered_answerable`,
        not `correct_answer`). Only the presence or absence of an answer is read.
        """
        result = trusted_search(
            self._store, self._embedder, question,
            k=self._k, candidate_k=self._ck, reranker=self._rr,
        )
        cited: list[str] = []
        for hit in result.hits:
            did = _doc_id_of(hit.chunk.metadata.get("file"))
            if did is not None and did not in cited:
                cited.append(did)
        # `cosine` of the top hit, and `hits` is ordered verdict-ok first. Reported even when the
        # system abstained, because that is exactly the row on which the declared threshold rule is
        # re-derived: a `score` recorded only for answered rows would make the re-derivation
        # vacuous on the half of the axis that matters.
        top = result.hits[0].cosine if result.hits else 0.0
        answered = not result.abstained and bool(result.hits)
        tokens = sum(len(h.chunk.text.split()) for h in result.hits)
        return Response(
            answer="answered" if answered else None,
            cited_ids=tuple(cited),
            tokens=tokens,
            top_cosine=float(top),
        )


#: `/` and `:` are the two characters LOCOMO ids carry that a filename cannot. Percent-style and
#: not `str.replace`, so the mapping is INJECTIVE: `a/b` and `a:b` are different documents and must
#: not become one file. `%` itself is escaped first, or an id containing a literal `%2F` would
#: decode to a different id than it encoded from.
_ESCAPES = (("%", "%25"), ("/", "%2F"), (":", "%3A"))


def _encode(doc_id: str) -> str:
    for raw, enc in _ESCAPES:
        doc_id = doc_id.replace(raw, enc)
    return doc_id


def _decode(name: str) -> str:
    for raw, enc in reversed(_ESCAPES):
        name = name.replace(enc, raw)
    return name


def _doc_id_of(file_name: str | None) -> str | None:
    """`None` for a chunk with no `file` metadata, never a guessed id.

    A chunk this adapter cannot attribute must not be reported as a document the manifest names:
    on this axis that would be a citation of evidence the run may never have held, and the delivery
    layer reads `cited_ids` against the manifest's excision map.
    """
    if not file_name:
        return None
    return _decode(file_name.rsplit(".", 1)[0])
