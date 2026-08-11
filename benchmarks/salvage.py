"""Resume a crashed benchmark run from its incremental sidecar instead of re-paying for it.

::

    # a run died at conversation 5 of 10 - score only the missing conversations
    python -m benchmarks.salvage --partial benchmarks/results/mem0_..._10conv_...partial.jsonl \\
        --arm mem0 --model openai/gpt-4o-mini --k 5 --data locomo10.json \\
        --out benchmarks/results/mem0_..._10conv_...salvaged.json

    # the run scored everything and died before writing the final JSON - no scoring, no spend
    python -m benchmarks.salvage --partial ...partial.jsonl --arm mem0 --merge-only \\
        --data locomo10.json --out ...salvaged.json

Why this exists
---------------
`benchmarks.run` writes one JSONL line per scored question into ``<stamp>.partial.jsonl`` after
each conversation, and the full ``<stamp>.json`` only at the very end. The sidecar is therefore
the only survivor of a crash — and it is a survivor that has already been PAID for: a generator
call, plus a judge call, for every record it holds, plus (on the Mem0 arms) the per-session LLM
extraction that ingesting those conversations cost. `benchmarks.run` has no resume, so relaunching
it starts at conversation 1 and buys all of that a second time.

This module is the resume. It reads the sidecar(s), works out which conversations are still
unscored, scores ONLY those, and assembles an artifact in exactly the schema `benchmarks.run`
writes — same top-level keys, same per-question record shape, the aggregate recomputed by the same
`benchmarks.pipeline.aggregate` — so every downstream tool (`benchmarks.analyze`,
`benchmarks.rejudge`) reads a salvaged file without knowing it is one.

Three properties exist because the input is, by definition, a file some process was killed while
writing:

- a TRUNCATED FINAL LINE is tolerated and discarded, and the kept/discarded counts are reported.
  The crash case is exactly the case where the last line is half-written. A malformed line
  anywhere EARLIER is not tolerated: mid-file corruption is not a partial write, and silently
  skipping it would drop a paid-for record without saying so.
- the input sidecar is never appended to, never rewritten, never deleted. New records go to a NEW
  sidecar, so a salvage that itself crashes leaves the original evidence exactly as it found it.
- the artifact says it was salvaged. A top-level ``salvaged: true`` plus a ``salvage`` block naming
  every partial consumed, how many records came from each, how many were re-scored, and what could
  NOT be verified. A number assembled from two processes is not the same object as a number from
  one, and a reader has to be able to see which one they are holding.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.llm import Completer, OpenRouterLLM
from benchmarks.pipeline import Outcome, aggregate
from benchmarks.rejudge import to_outcome
from benchmarks.run import (
    _append_records,
    _build_system,
    _gold_by_id,
    _load,
    _outcome_record,
    _positive_int,
    _results_payload,
    _run_config,
    _text_by_id,
    run_arm,
    validate_openrouter_key,
)
from benchmarks.systems import DEFAULT_K, MemorySystem, sample_id_of
from benchmarks.artifact_contract import load_published_artifact

#: The fields of a scored `Outcome`, i.e. the part of a sidecar record that is load-bearing.
#: ``question`` and ``gold`` are joined in beside them by `benchmarks.run._outcome_record` and are
#: tolerated as absent here for the same reason `benchmarks.rejudge` tolerates it: they are text
#: carried for re-scoring, not inputs to any rate.
_OUTCOME_KEYS = (
    "question_id",
    "category",
    "is_adversarial",
    "context",
    "answer",
    "abstained",
    "correct",
)

#: The suffix `benchmarks.run` gives its incremental sidecar.
PARTIAL_SUFFIX = ".partial.jsonl"

#: `benchmarks.run._run_stamp`'s output, read backwards. The arm and the model are recoverable from
#: the filename alone, which is the only cross-check available when the run died before writing an
#: artifact. Longest arm first: ``mem0-default`` must not be matched as ``mem0`` with a model that
#: happens to start with ``-default``.
_STEM = re.compile(
    r"^(?P<arm>recall|mem0-default|mem0)"
    r"_(?P<model>.+)"
    r"_(?P<conversations>\d+)conv"
    r"_(?P<stamp>\d{8}T\d{6}Z)$"
)


@dataclass(frozen=True)
class StampFields:
    """What `benchmarks.run`'s filename stem encodes: arm, model, slice size, run time."""

    arm: str
    model: str
    conversations: int
    stamp: str


@dataclass(frozen=True)
class PartialLoad:
    """One sidecar, parsed: the records kept and how many trailing lines had to be thrown away."""

    path: Path
    records: list[dict[str, Any]]
    discarded_lines: list[int]

    @property
    def kept(self) -> int:
        return len(self.records)

    def provenance(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "records": self.kept,
            "discarded_lines": len(self.discarded_lines),
        }


def partial_stem(path: Path) -> str:
    """The run stem of a sidecar or an artifact — ``<stem>.partial.jsonl`` or ``<stem>.json``."""
    name = path.name
    if name.endswith(PARTIAL_SUFFIX):
        return name[: -len(PARTIAL_SUFFIX)]
    return path.stem


def parse_stem(stem: str) -> StampFields | None:
    """`StampFields` for a stem `benchmarks.run` produced, or None if it did not produce it.

    None rather than an exception: a hand-renamed file is not an error, it is simply a file that
    cannot be cross-checked, and the salvage records that it could not be rather than refusing to
    run or — worse — pretending the check passed.
    """
    match = _STEM.match(stem)
    if match is None:
        return None
    return StampFields(
        arm=match["arm"],
        model=match["model"],
        conversations=int(match["conversations"]),
        stamp=match["stamp"],
    )


def _parse_line(line: str) -> dict[str, Any]:
    """One sidecar line as a record, or ValueError describing what is wrong with it."""
    obj = json.loads(line)
    if not isinstance(obj, dict):
        raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
    missing = [key for key in _OUTCOME_KEYS if key not in obj]
    if missing:
        raise ValueError(f"record is missing {missing}")
    # Rebuild it as an `Outcome`: `Outcome.__post_init__` re-checks the "correct is None iff
    # adversarial" invariant, so a corrupt record fails here instead of silently changing what the
    # accuracy rate is a rate OF.
    to_outcome(obj)
    record: dict[str, Any] = obj
    return record


def read_partial(path: Path) -> PartialLoad:
    """Parse one ``.partial.jsonl``, tolerating a truncated FINAL line and only a final one.

    A process killed mid-write leaves the last line half-formed; that line is dropped and counted.
    A malformed line anywhere earlier is a different animal — the file was written append-only, one
    complete line at a time, so an earlier bad line means the bytes on disk were damaged rather
    than merely cut short. Skipping it would quietly shrink a paid-for result set, so it raises.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    discarded: list[int] = []
    for number, line in enumerate(lines, start=1):
        try:
            records.append(_parse_line(line))
        except ValueError as exc:  # json.JSONDecodeError is a ValueError
            if number != len(lines):
                raise ValueError(
                    f"{path}:{number} is malformed and is NOT the final line "
                    f"({exc}) — a partial write can only damage the last line, so this file is "
                    "corrupt rather than truncated and salvaging it would silently drop a record "
                    "that was already paid for"
                ) from exc
            discarded.append(number)
    return PartialLoad(path=path, records=records, discarded_lines=discarded)


def load_partial(path: Path) -> list[dict[str, Any]]:
    """The records of one sidecar. See `read_partial` for the kept/discarded counts."""
    return read_partial(path).records


def sample_id_from_question_id(question_id: str) -> str:
    """The conversation a question belongs to, undoing `benchmarks.run._load`'s composition.

    That composition is ``f"{sample_id}:{index}"`` with an integer index, so the split is on the
    LAST colon — a `sample_id` containing one would otherwise be truncated into a scope that
    matches nothing and silently mark its conversation unscored (re-paying for it).
    """
    head, sep, tail = question_id.rpartition(":")
    if not sep or not head or not tail.isdigit():
        raise ValueError(
            f"question_id {question_id!r} is not '<sample_id>:<index>' — the sidecar does not come "
            "from benchmarks.run, and its conversation scope cannot be derived"
        )
    return head


def scored_conversations(records: Iterable[dict[str, Any]]) -> set[str]:
    """The `sample_id`s these records represent."""
    return {sample_id_from_question_id(str(record["question_id"])) for record in records}


def merge_records(record_lists: Iterable[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Concatenate record lists, de-duplicating by `question_id`, in first-seen order.

    An identical duplicate is dropped silently: the same record written twice (a salvage of a
    salvage, overlapping sidecars) carries no information the first copy did not.

    A CONFLICTING duplicate raises. Two records under one `question_id` that differ in content came
    from two different scorings — different retrieved context, a different generated answer, or a
    different verdict — and keeping either one would publish a headline number assembled from two
    runs that disagree, with nothing in the artifact to say so.
    """
    merged: dict[str, dict[str, Any]] = {}
    for records in record_lists:
        for record in records:
            question_id = str(record["question_id"])
            existing = merged.get(question_id)
            if existing is None:
                merged[question_id] = record
                continue
            if existing != record:
                differing = sorted(
                    key
                    for key in set(existing) | set(record)
                    if existing.get(key) != record.get(key)
                )
                raise ValueError(
                    f"conflicting records for question_id {question_id!r}: they disagree on "
                    f"{differing} — the two copies were scored separately, and silently keeping "
                    "one would publish a number assembled from two inconsistent runs"
                )
    return list(merged.values())


def missing_conversations(
    all_conversations: Sequence[dict[str, Any]], records: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The conversations with NO record at all, in the dataset's original order.

    Order matters beyond tidiness: both adapters scope retrieval to the last conversation ingested,
    so the resume walks conversations one at a time exactly as `benchmarks.run.main` does.
    """
    scored = scored_conversations(records)
    return [conv for conv in all_conversations if sample_id_of(conv) not in scored]


def conversations_to_score(
    conversations: Sequence[dict[str, Any]],
    questions: Sequence[dict[str, Any]],
    records: Iterable[dict[str, Any]],
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """(conversation, its still-unscored questions) for every conversation with work left.

    Finer-grained than `missing_conversations`, and it has to be. Dropping a truncated final line
    leaves a conversation that is PRESENT but one record SHORT; treating presence as completeness
    would publish an artifact quietly missing that question. Working per QUESTION also means the
    surviving records of such a conversation are kept rather than re-bought — the conversation is
    re-ingested (both adapters need that before they can retrieve), but only the genuinely missing
    questions are sent to the generator and the judge.
    """
    have = {str(record["question_id"]) for record in records}
    todo_by_sample: dict[str, list[dict[str, Any]]] = {}
    for question in questions:
        if str(question["question_id"]) in have:
            continue
        todo_by_sample.setdefault(str(question["sample_id"]), []).append(question)
    work: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for conv in conversations:
        todo = todo_by_sample.get(sample_id_of(conv))
        if todo:
            work.append((conv, todo))
    return work


def score_missing(
    system: MemorySystem,
    completer: Completer,
    work: Sequence[tuple[dict[str, Any], list[dict[str, Any]]]],
    text_by_id: dict[str, str],
    gold_by_id: dict[str, str],
    sidecar: Path | None = None,
) -> list[dict[str, Any]]:
    """Ingest and score only the outstanding questions, appending to a NEW sidecar as we go.

    Mirrors `benchmarks.run.main`'s loop — ingest one conversation, score its questions, persist
    before touching the next — because the salvage can crash for exactly the reasons the original
    run did, and a salvage without its own incremental write would need salvaging in turn.
    """
    written: list[dict[str, Any]] = []
    for position, (conv, todo) in enumerate(work):
        sample_id = sample_id_of(conv)
        system.ingest(conv)
        outcomes, _ = run_arm(system, completer, todo)
        records = [_outcome_record(o, text_by_id, gold_by_id) for o in outcomes]
        if sidecar is not None:
            _append_records(sidecar, records)
        written.extend(records)
        print(
            f"  [{position + 1}/{len(work)}] {sample_id}: {len(records)} questions re-scored",
            flush=True,
        )
    return written


def _ordered(
    records: Sequence[dict[str, Any]], questions: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Records in the dataset's own question order, so a salvaged artifact reads like a clean one.

    Records whose `question_id` is not in the loaded slice keep their encounter order at the end
    rather than being dropped: they were paid for, and losing them silently is the one thing this
    module exists to prevent. They are also reported in the provenance block.
    """
    by_id = {str(record["question_id"]): record for record in records}
    ordered: list[dict[str, Any]] = []
    for question in questions:
        record = by_id.pop(str(question["question_id"]), None)
        if record is not None:
            ordered.append(record)
    ordered.extend(by_id.values())
    return ordered


def consistency_report(
    paths: Sequence[Path], arm: str, model: str, k: int
) -> dict[str, Any]:
    """Cross-check the requested arm/model/k against whatever the inputs can prove, or raise.

    Resuming a ``recall`` sidecar with ``--arm mem0`` would score the remaining conversations with
    a different memory system and publish the mixture as one number — the failure this whole module
    would otherwise make easy. Two sources can testify:

    - the sidecar's FILENAME, which `benchmarks.run._run_stamp` builds from the arm and the model;
    - a sibling ``<stem>.json`` artifact, which additionally carries ``config.k``.

    A crashed run has no sibling artifact, so ``k`` is usually unverifiable. That is reported in
    ``unverified`` and carried into the published provenance verbatim — an unverifiable check is
    said out loud rather than quietly counted as a pass.
    """
    verified: list[dict[str, Any]] = []
    unverified: list[str] = []
    mismatches: list[str] = []
    filename_model = model.replace("/", "-")

    for path in paths:
        stem = partial_stem(path)
        fields = parse_stem(stem)
        if fields is None:
            unverified.append(
                f"{path.name}: filename is not a benchmarks.run stem, so neither the arm nor the "
                "model could be checked against it"
            )
        else:
            verified.append(
                {"source": f"{path.name} (filename)", "arm": fields.arm, "model": fields.model}
            )
            if fields.arm != arm:
                mismatches.append(f"{path.name} was produced by --arm {fields.arm}, not {arm}")
            if fields.model != filename_model:
                mismatches.append(
                    f"{path.name} was produced by --model {fields.model}, not {filename_model}"
                )

        sibling = path.with_name(f"{stem}.json")
        if not sibling.exists():
            unverified.append(
                f"{path.name}: no sibling {sibling.name}, so k={k} could not be checked against "
                "the run that produced these records"
            )
            continue
        try:
            # Through the publication check, not a bare load: this sibling IS a `benchmarks.run`
            # results artifact, and its config is cited below as verified provenance. A refused
            # artifact must not be able to launder its config into a newly published one.
            doc = load_published_artifact(sibling)
            config = doc["config"]
        except SystemExit as exc:
            unverified.append(f"{sibling.name}: {exc}")
            continue
        except (OSError, ValueError, KeyError, TypeError) as exc:
            unverified.append(f"{sibling.name}: unreadable as a results artifact ({exc})")
            continue
        verified.append(
            {
                "source": f"{sibling.name} (config)",
                "arm": config.get("arm"),
                "model": config.get("model"),
                "k": config.get("k"),
            }
        )
        for field_name, requested in (("arm", arm), ("model", model), ("k", k)):
            found = config.get(field_name)
            if found is not None and found != requested:
                mismatches.append(
                    f"{sibling.name} records {field_name}={found!r}, not {requested!r}"
                )

    if mismatches:
        raise ValueError(
            "refusing to salvage: the partials disagree with the requested run — "
            + "; ".join(mismatches)
            + ". Resuming under the wrong arm/model/k mixes two runs into one published number."
        )
    return {
        "requested": {"arm": arm, "model": model, "k": k},
        "verified": verified,
        "unverified": unverified,
    }


class _NoSystem:
    """Stands in for the arm under ``--merge-only``, which ingests and retrieves nothing.

    It has no ``describe()``, so the rebuilt ``config.system`` block comes out empty — which is the
    honest record: a merge-only salvage never constructed the memory system, so it has nothing to
    report about its configuration and must not copy a plausible-looking one from somewhere else.
    """

    name = "merge-only"

    def ingest(self, conversation: dict[str, Any]) -> None:
        raise RuntimeError("--merge-only never ingests")

    def retrieve(self, question: str) -> str:
        raise RuntimeError("--merge-only never retrieves")


def resume(
    loads: Sequence[PartialLoad],
    conversations: Sequence[dict[str, Any]],
    questions: Sequence[dict[str, Any]],
    skipped: dict[str, Any],
    *,
    arm: str,
    model: str,
    config: dict[str, Any],
    consistency: dict[str, Any],
    system: MemorySystem | None = None,
    completer: Completer | None = None,
    sidecar: Path | None = None,
) -> dict[str, Any]:
    """Assemble the salvaged artifact: partials + (optionally) the re-scored remainder.

    ``system``/``completer`` are None for a merge-only salvage. In that mode nothing is scored, so
    any question still outstanding is COUNTED and named in the provenance rather than dropped
    quietly — an artifact that is short of questions must say how short, or its n silently
    contradicts every other run of the same slice.
    """
    salvaged = merge_records([load.records for load in loads])
    work = conversations_to_score(conversations, questions, salvaged)

    # Question text and gold come from the dataset, falling back to the records for any id the
    # loaded slice does not contain — the artifact promises to be re-scorable on its own, and a
    # record whose text was silently blanked would break that promise without failing anything.
    text_by_id = {
        **{str(r["question_id"]): str(r.get("question", "")) for r in salvaged},
        **_text_by_id(list(questions)),
    }
    gold_by_id = {
        **{str(r["question_id"]): str(r.get("gold", "")) for r in salvaged},
        **_gold_by_id(list(questions)),
    }

    rescored: list[dict[str, Any]] = []
    if system is not None and completer is not None:
        rescored = score_missing(system, completer, work, text_by_id, gold_by_id, sidecar)

    records = _ordered(merge_records([salvaged, rescored]), questions)
    outcomes: list[Outcome] = [to_outcome(record) for record in records]
    payload = _results_payload(
        arm,
        model,
        list(conversations),
        text_by_id,
        gold_by_id,
        outcomes,
        aggregate(outcomes),
        config,
        skipped,
        # A salvaged artifact is stitched from partials that never carried token usage, and any
        # re-scored questions are only a subset — so a token total here would be a lie. Marked
        # unmetered rather than reported as zero.
        {"note": "usage not metered on salvaged runs"},
    )

    known_ids = {str(q["question_id"]) for q in questions}
    final_ids = {str(record["question_id"]) for record in records}
    absent = [sample_id_of(conv) for conv in missing_conversations(conversations, salvaged)]
    # Derived from what was actually written, not from what was planned: a scoring pass that
    # covered only some of the outstanding work must not report the plan as if it were the result.
    outstanding = [
        str(q["question_id"]) for q in questions if str(q["question_id"]) not in final_ids
    ]
    rescored_samples = list(
        dict.fromkeys(sample_id_from_question_id(str(r["question_id"])) for r in rescored)
    )
    payload["salvaged"] = True
    payload["salvage"] = {
        "sources": [load.provenance() for load in loads],
        "records_from_partials": len(salvaged),
        "records_rescored": len(rescored),
        "conversations_rescored": rescored_samples,
        "conversations_absent_from_partials": absent,
        "conversations_short_in_partials": [
            sample_id_of(conv) for conv, _todo in work if sample_id_of(conv) not in absent
        ],
        "questions_still_unscored": outstanding,
        "question_ids_not_in_this_slice": sorted(
            {str(r["question_id"]) for r in salvaged} - known_ids
        ),
        "new_sidecar": None if sidecar is None else str(sidecar),
        "merge_only": system is None,
        "consistency": consistency,
    }
    return payload


def _default_conversations(paths: Sequence[Path]) -> int | None:
    """The slice size the partials were produced from, when they all agree on one.

    Loading MORE conversations than the original run would invent missing work and re-pay for
    conversations the run was never asked to score; loading fewer would strand records outside the
    slice. Taken from the filename when it is readable, and left to the caller's ``--conversations``
    when it is not.
    """
    sizes = {fields.conversations for fields in map(parse_stem, map(partial_stem, paths)) if fields}
    if len(sizes) == 1:
        return sizes.pop()
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m benchmarks.salvage",
        description=(
            "Resume a crashed benchmark run from its .partial.jsonl sidecar, scoring only the "
            "conversations it never reached, and write a schema-identical results artifact."
        ),
    )
    p.add_argument(
        "--partial",
        type=Path,
        action="append",
        required=True,
        metavar="PATH",
        help="incremental sidecar to salvage; repeat to merge several",
    )
    p.add_argument("--arm", choices=["recall", "mem0", "mem0-default"], required=True)
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--k", type=_positive_int, default=DEFAULT_K)
    p.add_argument("--data", type=Path, default=Path("locomo10.json"))
    p.add_argument(
        "--conversations",
        type=_positive_int,
        default=None,
        help="dataset slice the original run used; defaults to the count in the partial's filename",
    )
    p.add_argument("--out", type=Path, required=True, help="artifact to write")
    p.add_argument(
        "--sidecar",
        type=Path,
        default=None,
        help="where re-scored records are appended (default: <out stem>.salvage.partial.jsonl)",
    )
    p.add_argument(
        "--merge-only",
        action="store_true",
        help="rebuild the artifact from the partials alone: no ingest, no scoring, no spend",
    )
    args = p.parse_args(argv)

    partials: list[Path] = list(args.partial)
    for path in partials:
        if not path.exists():
            p.error(f"{path} not found")
    if args.out.exists():
        p.error(f"{args.out} already exists - refusing to overwrite a results artifact")
    if not args.data.exists():
        p.error(
            f"{args.data} not found. Fetch it with:\n"
            "  curl -sLO https://raw.githubusercontent.com/snap-research/locomo/main/data/"
            "locomo10.json"
        )

    sidecar: Path | None = None
    if not args.merge_only:
        sidecar = args.sidecar or args.out.with_name(f"{args.out.stem}.salvage.partial.jsonl")
        # Resolved, not compared as spelled. Appending to an input sidecar would edit the only
        # surviving evidence of the crashed run while reading it.
        resolved = {path.resolve() for path in partials}
        if sidecar.resolve() in resolved:
            p.error("--sidecar must differ from every --partial: the inputs are never modified")
        if sidecar.exists():
            p.error(f"{sidecar} already exists - refusing to append to an existing sidecar")

    try:
        consistency = consistency_report(partials, args.arm, args.model, args.k)
    except ValueError as exc:
        p.error(str(exc))

    # Checked BEFORE anything is ingested, exactly as `benchmarks.run.main` does: on the Mem0 arms
    # ingest spends money on per-session LLM extraction, and discovering a bad key afterwards would
    # waste it. A merge-only salvage makes no calls at all, so it does not require a key.
    # ASCII only: argparse writes this to stderr, and a Windows cp1252 console mangles anything
    # else.
    key = ""
    if not args.merge_only:
        try:
            key = validate_openrouter_key(os.environ.get("OPENROUTER_API_KEY"))
        except ValueError as exc:
            p.error(str(exc))

    loads = [read_partial(path) for path in partials]
    for load in loads:
        if load.discarded_lines:
            print(
                f"{load.path.name}: kept {load.kept} records, discarded "
                f"{len(load.discarded_lines)} truncated trailing line(s)",
                flush=True,
            )

    limit: int | None = args.conversations
    if limit is None:
        limit = _default_conversations(partials)
    convs, questions, skipped = _load(args.data, limit)

    llm = OpenRouterLLM(model=args.model, api_key=key)
    system: MemorySystem | None = None
    completer: Completer | None = None
    if not args.merge_only:
        # The stamp the arm is built under is this salvage's own output stem, so a Mem0 arm names
        # its vector store after THIS process and cannot reopen the dead run's accumulated store.
        system = _build_system(args.arm, args.model, key, args.k, args.out.stem)
        completer = llm.complete

    payload = resume(
        loads,
        convs,
        questions,
        skipped,
        arm=args.arm,
        model=args.model,
        config=_run_config(args.arm, args.model, args.k, llm, system or _NoSystem()),
        consistency=consistency,
        system=system,
        completer=completer,
        sidecar=sidecar,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    salvage: dict[str, Any] = payload["salvage"]
    unscored: list[str] = salvage["questions_still_unscored"]
    if unscored:
        print(
            f"WARNING: {len(unscored)} question(s) are still unscored and are NOT in this "
            "artifact - it is short of a complete run; see salvage.questions_still_unscored",
            flush=True,
        )
    print(json.dumps(payload["aggregate"], indent=2))
    print(
        f"salvaged {salvage['records_from_partials']} records from "
        f"{len(salvage['sources'])} partial(s), re-scored {salvage['records_rescored']}"
    )
    print(f"salvaged artifact -> {args.out}")
    if sidecar is not None:
        print(f"new sidecar       -> {sidecar}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
