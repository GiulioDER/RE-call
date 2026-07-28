"""BEIR datasets as corpora the `labelled` harness can score.

BEIR ships `corpus.jsonl`, `queries.jsonl` and graded `qrels/*.tsv`. The harness wants a directory
of files plus a question list carrying `relevant_files`. This module is the adapter, and it is
almost entirely concerned with the ways that translation can corrupt ground truth without
erroring: a filename collision merging two documents, a relevance-0 judgement counted as relevant,
a gold document dropped by subsampling.

Every corpus produced here is deterministic given `seed`, because the study's whole claim is a
correlation across corpora and a corpus that differs between runs cannot be part of one.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: Characters that are legal in a BEIR document id and illegal (or dangerous) in a filename on at
#: least one supported platform. Windows additionally rejects `:` and is case-insensitive, which
#: is its own collision source — hence the hash below rather than sanitisation alone.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe_filename(doc_id: str) -> str:
    """A legal, collision-free filename for a BEIR document id.

    The hash suffix is the entire point. Sanitisation alone maps `doc/1` and `doc_1` onto the same
    name; the second write clobbers the first, both remain referenced by the qrels, and one
    question is scored against the wrong document with no error anywhere. The readable stem is
    kept only so a human can navigate the directory — correctness rests on the digest.

    Case is folded into the digest input rather than the stem, so two ids differing only in case
    stay distinct on a case-insensitive filesystem.
    """
    digest = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:12]
    stem = _UNSAFE.sub("_", doc_id)[:60].strip("._") or "doc"
    return f"{stem}-{digest}.txt"


def select_documents(
    qrels: Mapping[str, Mapping[str, int]],
    all_doc_ids: Sequence[str],
    *,
    max_docs: int,
    seed: int = 0,
) -> list[str]:
    """`max_docs` document ids: every judged-relevant document, then random negatives.

    Gold documents are never dropped. A dropped gold turns a scorable question into an
    unanswerable one, which lowers the measured score for a reason that has nothing to do with the
    embedder under test — and does so silently, since an unfindable document and a findable one
    the model missed look identical in the metric.

    When the gold set alone exceeds `max_docs` the budget yields rather than the ground truth, so
    the returned list can be longer than requested. That is visible to the caller (`len`) and is
    the honest failure: a corpus larger than planned is a reportable deviation, a corrupted qrels
    file is not detectable at all.
    """
    gold = {doc_id for judged in qrels.values() for doc_id, rel in judged.items() if rel > 0}
    present = set(all_doc_ids)
    selected = sorted(gold & present)

    remaining = [d for d in sorted(present - gold)]
    room = max_docs - len(selected)
    if room > 0:
        rng = random.Random(seed)
        selected += rng.sample(remaining, min(room, len(remaining)))
    return sorted(selected)


def build_questions(
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    selected: set[str],
) -> list[dict[str, Any]]:
    """Question records in the shape `recall.eval.labelled` consumes.

    Two filters, both of which change the measured result rather than merely tidying it:

    Relevance 0 in BEIR means *judged and NOT relevant*. Counting those as gold inflates hit@5 for
    both embedders and compresses the gap toward zero — which is this study's response variable.

    A question whose gold documents all fell outside the subsample is dropped. Keeping it would
    score the subsample rather than the embedder, and it would score it as failure for both.
    """
    out: list[dict[str, Any]] = []
    for query_id in sorted(queries):
        relevant = sorted(
            safe_filename(doc_id)
            for doc_id, rel in qrels.get(query_id, {}).items()
            if rel > 0 and doc_id in selected
        )
        if not relevant:
            continue
        out.append({
            "id": query_id,
            "query": queries[query_id],
            "answerable": True,
            "relevant_files": relevant,
        })
    return out


def read_beir(root: Path) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    """Read an extracted BEIR dataset directory into `(documents, queries, qrels)`.

    Only the `test` split is read. BEIR's `title` and `text` fields are joined because the
    harness scores whole files and a title carries real retrieval signal — dropping it would
    handicap both embedders on corpora whose titles are informative (PEPs, SciFact) and neither on
    corpora whose titles are empty, i.e. it would vary the difficulty by corpus.
    """
    documents: dict[str, str] = {}
    with (root / "corpus.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            title = (record.get("title") or "").strip()
            body = (record.get("text") or "").strip()
            documents[record["_id"]] = f"{title}\n\n{body}".strip()

    queries: dict[str, str] = {}
    with (root / "queries.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            queries[record["_id"]] = record["text"]

    qrels: dict[str, dict[str, int]] = {}
    with (root / "qrels" / "test.tsv").open(encoding="utf-8") as fh:
        header = fh.readline()  # query-id \t corpus-id \t score
        if "query" not in header.lower():
            fh.seek(0)  # not every mirror ships the header row
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            query_id, doc_id, score = parts[0], parts[1], int(parts[2])
            qrels.setdefault(query_id, {})[doc_id] = score
    return documents, queries, qrels


def materialize(
    root: Path,
    out_dir: Path,
    *,
    max_docs: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Write a BEIR dataset out as a corpus directory plus `questions.json`.

    Returns the manifest — document and question counts, the requested budget and the actual one.
    Those differ whenever the gold set exceeded the budget, and the study must report the real
    number rather than the planned one.
    """
    documents, queries, qrels = read_beir(root)
    selected = select_documents(qrels, list(documents), max_docs=max_docs, seed=seed)
    questions = build_questions(queries, qrels, set(selected))

    out_dir.mkdir(parents=True, exist_ok=True)
    for doc_id in selected:
        (out_dir / safe_filename(doc_id)).write_text(documents[doc_id], encoding="utf-8")
    (out_dir / "questions.json").write_text(json.dumps(questions, indent=2), encoding="utf-8")

    return {
        "name": root.name,
        "documents_requested": max_docs,
        "documents_written": len(selected),
        "questions": len(questions),
        "questions_available": len(queries),
        "seed": seed,
    }
