"""Audit one corpus: history, restated claims, optionally stale answers, one markdown report.

Two steps, because indexing already exists and should not be reimplemented here:

    python -m recall_consistency ../their-repo --out report.md --corpus-out ./history
    RECALL_TRUST_MODE=development python -m recall.cli --table audit index ./history
    RECALL_TRUST_MODE=development python -m recall_consistency ../their-repo --out report.md \\
        --questions questions.json --serving-dsn "$DSN" --table audit

The audit step needs `RECALL_TRUST_MODE=development` too: the calibration gate in
`recall.trust.trusted_search` sits above retrieval, and a freshly emitted, freshly indexed
corpus has no calibration bound to it yet. Without the development mode the gate refuses on
question 1 and the run aborts before answering any of them.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from recall_consistency.claim_drift import drifts
from recall_consistency.findings import ClaimDrift, StaleAnswer
from recall_consistency.history_corpus import revisions, tracked_markdown, write_history_corpus
from recall_consistency.report import render

if TYPE_CHECKING:
    from recall.embeddings import Embedder
    from recall.store import PgVectorStore


#: Stems meaning a field carries direct contact data. The audit needs a question's text and
#: nothing else, so a file arriving with these is refused rather than read past: quietly ignoring
#: a field still means the file was accepted, sits on disk, and falls under whatever was promised
#: about retention. Refusing is a guard the operator cannot forget to apply; a written rule is one
#: they can.
#:
#: Matched as substrings of a lowercased key, not as exact field names. An exact-name list leaked
#: three ways at once: `contactEmail` and `phoneNumber` never matched a snake_case vocabulary,
#: plurals like `emails` matched nothing, and a nested `{"meta": {"email": ...}}` was never looked
#: at. Enumerating the spellings people use is a losing game, so this states the rule instead.
#:
#: It over-matches deliberately, and not only on `contact`: `phone` is inside `phonetic` and
#: `microphone`, `mobile` is inside `automobile`, `mail` is inside `mailbox`, and `discord` is an
#: ordinary English word that predates the chat app. A corpus about linguistics or vehicles can
#: trip these on an innocent field name. That is accepted: a rename costs the owner a minute, and
#: accepting a file that turned out to carry contact data costs a promise.
#:
#: Known limit: matching is on ASCII substrings, so a homoglyph such as a Cyrillic `а` inside
#: `email` reads as contact data to a person and not to this code. The threat model here is an
#: inattentive owner rather than someone hand-crafting lookalike keys.
CONTACT_STEMS = (
    "email",
    "mail",
    "phone",
    "mobile",
    "whatsapp",
    "telegram",
    "discord",
    "linkedin",
    "contact",
)


#: How deep the guard walks before refusing. `json.loads` accepts far deeper nesting than Python
#: will recurse over, so an exporter bug produced a `RecursionError` instead of a refusal.
MAX_CONTACT_DEPTH = 32


class _TooDeep(Exception):
    """A questions entry nested deeper than the guard will walk."""


def _contact_stems(value: object, depth: int = 0) -> list[str]:
    """Which contact stems appear in an entry's field names, nested objects and lists included.

    Returns the STEMS matched, never the field names. A field name is not safe to print: JSON
    objects can be keyed BY a contact value, and `{"jane.doe@gmail.com": ...}` matches on `mail`
    through its own domain, so echoing the name would put the address into the refusal message.
    That is exactly the data this guard exists to keep out of messages, so it would invert the
    guard for the most common personal email domains. The stem plus the entry's position is
    enough for an owner to find the field in their own file.

    Lowercasing is the whole normalisation, because stem substrings already cover separators and
    camelCase: `contactEmail` and `contact-email` both contain `email` once lowered. A camelCase
    splitter was written here first and deleted, because no mutation of it could turn any test
    red, which means it guaranteed nothing.
    """
    if depth > MAX_CONTACT_DEPTH:
        raise _TooDeep
    found: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            token = str(key).lower()
            found.update(stem for stem in CONTACT_STEMS if stem in token)
            found.update(_contact_stems(nested, depth + 1))
    elif isinstance(value, list):
        for item in value:
            found.update(_contact_stems(item, depth + 1))
    return sorted(found)


def _load_questions(path: Path) -> list[str]:
    """A JSON list of strings, or a JSON list of objects each carrying a `query` key.

    Errors name the path, the entry's position, and at most a rejected field's NAME, never any
    value: these questions are corpus-owner data, and an argument error is not a reason to print
    one.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit(f"{path}: expected a JSON list, found {type(raw).__name__}")
    out: list[str] = []
    for position, item in enumerate(raw, start=1):
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            try:
                stems = _contact_stems(item)
            except _TooDeep:
                raise SystemExit(
                    f"{path}: entry {position} nests deeper than {MAX_CONTACT_DEPTH} levels, so "
                    "it cannot be checked for contact fields. Flatten it and rerun."
                ) from None
            if stems:
                raise SystemExit(
                    f"{path}: entry {position} has field names matching {', '.join(stems)}. "
                    "This tool needs the question text and nothing else. Remove those fields "
                    "and rerun."
                )
            if not isinstance(item.get("query"), str):
                raise SystemExit(
                    f"{path}: entry {position} is neither a string nor an object with a string "
                    f"`query` key"
                )
            out.append(item["query"])
        else:
            raise SystemExit(
                f"{path}: entry {position} is neither a string nor an object with a string "
                f"`query` key"
            )
    if not out:
        raise SystemExit(f"{path}: no questions found")
    return out


def _run_probe(
    dsn: str,
    table: str,
    questions: list[str],
    *,
    make_embedder: Callable[[], Embedder] | None = None,
    make_store: Callable[[str, int, str], PgVectorStore] | None = None,
) -> list[StaleAnswer]:
    """Import the database path lazily: the claim scan must run without psycopg installed.

    `make_embedder` and `make_store` default to the real `FastEmbedEmbedder` and `PgVectorStore`,
    imported here rather than at module scope for the same reason. Tests inject fakes instead of
    monkeypatching those imports, which is the only way to reach this function's wiring at all.
    """
    if make_embedder is None:
        from recall.embeddings import FastEmbedEmbedder

        make_embedder = FastEmbedEmbedder
    if make_store is None:
        from recall.store import PgVectorStore

        make_store = PgVectorStore

    from recall.schema import SchemaIncompatible
    from recall_consistency.stale_probe import probe

    embedder = make_embedder()
    with make_store(dsn, embedder.dim, table) as store:
        try:
            # `PgVectorStore` itself never checks the dim it was constructed with against what
            # the table actually holds, so opening one with an embedder of the same dimension as
            # the one that built it, but a different model, raises nothing: the mismatch is
            # invisible until the query embeddings come back nonsense. `check_schema` reads the
            # actual table (SELECT only) and is the one thing that catches a real mismatch before
            # any question is sent. Its `SchemaIncompatible` message already names both
            # dimensions and the table, and nothing else, so it is safe to surface as-is.
            store.check_schema()
        except SchemaIncompatible as exc:
            raise SystemExit(
                f"{exc}: index with the same embedder you are probing with"
            ) from None
        return probe(store, embedder, questions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recall_consistency")
    parser.add_argument("repo", type=Path, help="path to the git repository being audited")
    parser.add_argument("--out", type=Path, required=True, help="where to write the report")
    parser.add_argument(
        "--corpus-out", type=Path, default=None, help="emit the history corpus here"
    )
    parser.add_argument("--glob", default="**/*.md", help="which tracked files to audit")
    parser.add_argument("--name", default="", help="corpus name for the report header")
    parser.add_argument(
        "--questions", type=Path, default=None, help="JSON list of questions to check"
    )
    parser.add_argument("--serving-dsn", default="", help="DSN of the indexed history corpus")
    parser.add_argument("--table", default="chunks", help="table holding the indexed corpus")
    args = parser.parse_args(argv)

    if args.questions is not None and not args.serving_dsn:
        parser.error("--questions needs --serving-dsn pointing at the indexed history corpus")
    # Load and validate before the git walk, and make the report's directory before anything is
    # written: a bad path should cost the operator a second, not a full history scan.
    questions = _load_questions(args.questions) if args.questions is not None else []
    args.out.parent.mkdir(parents=True, exist_ok=True)

    paths = tracked_markdown(args.repo, args.glob)
    found: list[ClaimDrift] = []
    total_revisions = 0
    for rel_path in paths:
        revs = revisions(args.repo, rel_path)
        total_revisions += len(revs)
        found.extend(drifts(revs))

    if args.corpus_out is not None:
        written = write_history_corpus(args.repo, paths, args.corpus_out)
        memos = "memo" if len(written) == 1 else "memos"
        print(f"wrote {len(written)} revision {memos} to {args.corpus_out}")

    stale: list[StaleAnswer] = []
    if args.questions is not None:
        stale = _run_probe(args.serving_dsn, args.table, questions)

    args.out.write_text(
        render(
            args.name or args.repo.name,
            found,
            stale,
            documents=len(paths),
            revisions=total_revisions,
            questions=len(questions),
        ),
        encoding="utf-8",
        newline="\n",
    )
    claims = "claim" if len(found) == 1 else "claims"
    documents = "document" if len(paths) == 1 else "documents"
    print(f"{len(found)} restated {claims} across {len(paths)} {documents} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
