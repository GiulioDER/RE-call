"""Run one MemorySystem over a manifest: ingest per distinct corpus state, record abstain-or-answer.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Instances are grouped by their EXCISED SET, not iterated one by one. LOCOMO is ~2 000 turns; a
re-ingest per instance would spend hours of embedding to produce a corpus state it already had.

Rows are flushed per instance and `--resume` skips by instance id, so a run that dies overnight
resumes without re-paying. Resume does NOT verify the system config — resuming across a config
change silently mixes two arms into one artifact.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from benchmarks.ladder.adapter import Document, MemorySystem, Response
from benchmarks.ladder.invariants import (
    assert_excised_absent,
    assert_originals_were_answered,
    assert_ring_zero_has_survivors,
    assert_survivors_present,
)
from benchmarks.ladder.manifest import RING_ORIGINAL, Instance, read_manifest


class AdapterSmokeCheckFailed(RuntimeError):
    """A `MemorySystem` passed `isinstance` but broke on its first real call.

    `MemorySystem` is `runtime_checkable`, which checks method NAMES only, not signatures (Task 7
    review). An adapter with the right names and the wrong shape passes the isinstance check and
    then fails deep inside a run that can run for hours — this exists so it fails in the first
    second instead.
    """


#: A distinctive id/text pair the smoke check ingests and immediately queries back. Namespaced so
#: it cannot collide with a real LOCOMO doc id (which is always "{sample_id}/{dia_id}", never this
#: literal prefix).
_SMOKE_DOC_ID = "__ladder_smoke_check__/probe"
_SMOKE_TEXT = "The quick brown fox jumps over the lazy dog."
_SMOKE_QUESTION = "What does the fox jump over?"


def smoke_check(system: MemorySystem) -> None:
    """Exercise `ingest`, `indexed_doc_ids`, and `query` once, for real, before the real run.

    Cheap and small on purpose: one document, one query. Its only job is to turn a bad adapter
    (wrong parameter count, wrong return shape, an exception on first use) into a failure at
    second zero of an overnight job, rather than at minute forty deep inside `run()`.
    """
    try:
        system.ingest([Document(_SMOKE_DOC_ID, _SMOKE_TEXT)])
        indexed = system.indexed_doc_ids()
        if _SMOKE_DOC_ID not in indexed:
            raise AdapterSmokeCheckFailed(
                f"{system.name}: ingest() accepted a probe document but indexed_doc_ids() does "
                f"not report it back. ingest() and indexed_doc_ids() disagree about what is "
                f"indexed — fix before running the full manifest."
            )
        response = system.query(_SMOKE_QUESTION)
        if not isinstance(response, Response):
            raise AdapterSmokeCheckFailed(
                f"{system.name}: query() returned {type(response).__name__}, not a "
                f"benchmarks.ladder.adapter.Response."
            )
    except AdapterSmokeCheckFailed:
        raise
    except Exception as exc:  # noqa: BLE001 - deliberately broad: ANY failure here is the point
        raise AdapterSmokeCheckFailed(
            f"smoke check failed for system {getattr(system, 'name', '?')!r}: {exc!r}. "
            f"MemorySystem is runtime_checkable, which only checks method names — this adapter "
            f"passed isinstance() and broke on the first real call."
        ) from exc


def _recorded(out_path: Path) -> dict[str, bool]:
    """instance_id -> abstained, for every row already written.

    Returns the flags, not just the ids, because invariant 3 must still fire on a RESUMED run. A
    resume where every answerable original was already scored would otherwise leave
    `answered_originals` empty and skip the check silently — the failure shape where grepping for
    success turns a failure into no output at all.
    """
    if not out_path.exists():
        return {}
    recorded: dict[str, bool] = {}
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            recorded[row["instance_id"]] = bool(row["abstained"])
    return recorded


def _cluster_id_of(instance: Instance) -> str:
    # Doc ids are "{cluster_id}/{dia_id}", so the gold id names its own cluster.
    if not instance.gold_doc_ids:
        return ""
    return instance.gold_doc_ids[0].split("/", 1)[0]


def run(
    manifest_path: Path,
    system: MemorySystem,
    out_path: Path,
    *,
    documents: Mapping[str, str],
    cluster_members: Mapping[str, Sequence[str]],
    resume: bool = True,
) -> int:
    """Returns the number of instances scored in this invocation."""
    instances, _header = read_manifest(manifest_path)
    recorded = _recorded(out_path) if resume else {}

    # Keyed by (cluster, excised) — a question is scored against its OWN conversation only, which
    # is what recall/eval/locomo.py already does and what LOCOMO's protocol assumes. It is also
    # the difference between indexing 646 turns per state and 5 882: at ~1 500 states that is the
    # difference between a run an adopter can finish and one nobody will.
    by_state: dict[tuple[str, tuple[str, ...]], list[Instance]] = {}
    for inst in instances:
        if inst.instance_id in recorded:
            continue
        key = (_cluster_id_of(inst), tuple(sorted(inst.excised_doc_ids)))
        by_state.setdefault(key, []).append(inst)

    # Seeded from what is already on disk, so invariant 3 sees the WHOLE artifact rather than only
    # this invocation's slice.
    answered_originals: dict[str, bool] = {
        inst.instance_id: not recorded[inst.instance_id]
        for inst in instances
        if inst.ring == RING_ORIGINAL and inst.instance_id in recorded
    }
    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        for (cluster_id, excised), group in sorted(by_state.items()):
            members = cluster_members.get(cluster_id, ())
            dropped = set(excised)
            keep = [Document(d, documents[d]) for d in sorted(members) if d not in dropped]
            system.ingest(keep)
            indexed = system.indexed_doc_ids()
            for inst in sorted(group, key=lambda i: i.instance_id):
                assert_excised_absent(inst, indexed)
                assert_ring_zero_has_survivors(inst, indexed, members)
                # The POSITIVE check, and the one that matters most. Every other invariant here
                # confirms that what should be gone is gone, which cannot tell a correct excision
                # from an ingest that silently did nothing. A partial ingest passes all of them,
                # abstains on nearly everything, flattens the curve, and would be recorded as an
                # H1 FAIL — retiring the benchmark on a harness bug.
                assert_survivors_present(inst, indexed, members)
                response = system.query(inst.question)
                if inst.ring == RING_ORIGINAL:
                    answered_originals[inst.instance_id] = not response.abstained
                fh.write(
                    json.dumps(
                        {
                            "instance_id": inst.instance_id,
                            "system": system.name,
                            "abstained": response.abstained,
                            "cited_ids": list(response.cited_ids),
                            "tokens": response.tokens,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                fh.flush()
                written += 1

    if answered_originals:
        assert_originals_were_answered(answered_originals, instances)
    return written


def main(argv: list[str] | None = None) -> int:
    from benchmarks.ladder.sources.locomo import load_locomo
    from benchmarks.ladder.systems.recall_system import RecallSystem

    parser = argparse.ArgumentParser(description="Run a system over the Answerability Ladder.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    corpus = load_locomo(args.locomo)
    system = RecallSystem(args.dsn)
    # Fail in the first second, not the fortieth minute: see `smoke_check`'s docstring. Run against
    # the REAL adapter instance before it ever touches the manifest, so a broken signature never
    # gets to burn even one real corpus state.
    smoke_check(system)
    n = run(
        args.manifest,
        system,
        args.out,
        documents=dict(corpus.documents),
        cluster_members=corpus.cluster_members,
        resume=not args.no_resume,
    )
    print(f"scored {n} instances into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
