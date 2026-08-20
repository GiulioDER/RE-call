"""Apparatus check for the regression set, and NOTHING else. Deliberately blind to the outcome.

The fourth record's regression set failed its own apparatus check: 6 of 10 queries never retrieved
the superseded document they were written to drag in, so they could not show a displacement. The
repair is to reauthor those queries. The hazard in that repair is obvious and is the reason this
file exists separately from the probe:

    ⛔ reauthoring until the DISPLACEMENT number improves is fitting the fixture to the answer.

The apparatus criteria are checkable without running a single ordering arm:

1. does the query actually retrieve the superseded document it names, and
2. is the gold document the top trusted answer under SHIPPED ordering.

Both are properties of retrieval, not of any arm. So this script computes those two and refuses to
compute anything else. `SuccessorExpansionPolicy` is never constructed here and `ordering` is never
mentioned. Iterating against this output cannot tune the result, because this output does not
contain it.

Run it until every query passes, THEN register a prediction, THEN run the probe.

    eval "$(scripts/session-db.sh up)"
    python -m benchmarks.successor_regression_check
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.find_spec("recall")
_ORIGIN = Path(_SPEC.origin).resolve() if _SPEC is not None and _SPEC.origin else None
if _ORIGIN is None or _HERE not in _ORIGIN.parents:
    raise SystemExit(
        f"refusing to run: `recall` resolves to {_ORIGIN}\n"
        f"                 but this check lives under {_HERE}\n"
        "run it as `python -m benchmarks.successor_regression_check` from the worktree root"
    )

from recall.calibration import from_samples  # noqa: E402
from recall.embeddings import FastEmbedEmbedder, embedding_profile_id  # noqa: E402
from recall.eval._research_trust import research_search  # noqa: E402
from recall.eval.calibrate import measure_top_cosines  # noqa: E402
from recall.index import Indexer  # noqa: E402
from recall.store import PgVectorStore  # noqa: E402

from benchmarks.successor_fixture import (  # noqa: E402
    CALIBRATION_ANSWERABLE,
    CALIBRATION_UNANSWERABLE,
    REGRESSIONS,
    documents,
)

K = 5
DISTRACTOR_DIR = Path(__file__).resolve().parent.parent / "docs"


def main() -> int:
    dsn = os.environ.get("RECALL_TEST_DSN")
    if not dsn:
        print('RECALL_TEST_DSN is unset. Run: eval "$(scripts/session-db.sh up)"', file=sys.stderr)
        return 2

    embedder = FastEmbedEmbedder()
    calib = [{"query": q, "answerable": True} for q in CALIBRATION_ANSWERABLE]
    calib += [{"query": q, "answerable": False} for q in CALIBRATION_UNANSWERABLE]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, text in documents().items():
            (root / name).write_text(text, encoding="utf-8")
        for doc in sorted(DISTRACTOR_DIR.glob("*.md")):
            shutil.copyfile(doc, root / f"docs__{doc.name}")

        store = PgVectorStore(dsn, dim=embedder.dim, table="regr_check_" + uuid.uuid4().hex[:8])
        try:
            store.ensure_schema()
            Indexer(store, embedder).index_path(root)
            cal = from_samples(
                embedding_profile_id(embedder), *measure_top_cosines(store, embedder, calib)
            )
            rows = []
            for reg in REGRESSIONS:
                # No policy passed. This is shipped behaviour with no expansion at all, which is
                # what both apparatus criteria are defined against.
                result = research_search(store, embedder, reg.query, k=K, calibration=cal)
                files = {h.provenance.file or "" for h in result.hits}
                ok = [h.provenance.file or "" for h in result.hits if h.verdict == "ok"]
                rows.append(
                    {
                        "slug": reg.slug,
                        "pulled_stale": reg.expects_stale in files,
                        "gold_top": bool(ok) and ok[0] == f"{reg.slug}.md",
                        "top": ok[0] if ok else "(abstained)",
                    }
                )
        finally:
            try:
                store.drop_table()
            finally:
                store.close()

    print(f"{'slug':<22}{'pulls predecessor':<20}{'gold is top':<14}top answer")
    for row in rows:
        print(
            f"{row['slug']:<22}{'yes' if row['pulled_stale'] else 'NO':<20}"
            f"{'yes' if row['gold_top'] else 'NO':<14}{row['top']}"
        )
    usable = [r for r in rows if r["pulled_stale"] and r["gold_top"]]
    print(f"\nusable: {len(usable)} of {len(rows)}")
    if len(usable) < len(rows):
        print("Not every query is usable yet. Reauthor the failures and run this again.")
        print("Do NOT run the probe until this is clean: a partial denominator is what made the")
        print("fourth record's displacement column uninterpretable.")
    return 0 if len(usable) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
