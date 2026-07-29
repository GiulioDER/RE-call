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
import inspect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from benchmarks.ladder.adapter import Document, MemorySystem, Response
from benchmarks.ladder.invariants import (
    assert_excised_absent,
    assert_manifest_digest,
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


def _assert_adapter_signatures(system: MemorySystem) -> None:
    """`runtime_checkable` validates method NAMES only, never signatures.

    So an adapter with `query(self)` instead of `query(self, question)` satisfies `isinstance`
    and then fails forty minutes into a run. This inspects the signatures without calling
    anything — a functional smoke call would move the ingest counters the tests pin.
    """
    required = {"ingest": 1, "indexed_doc_ids": 0, "query": 1}
    for name, arity in required.items():
        method = getattr(system, name, None)
        if method is None or not callable(method):
            raise AdapterSmokeCheckFailed(
                f"{type(system).__name__} has no callable {name!r}; a MemorySystem needs "
                f"ingest, indexed_doc_ids and query."
            )
        try:
            params = [
                p
                for p in inspect.signature(method).parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.default is p.empty
            ]
        except (TypeError, ValueError):  # builtins and C callables have no signature
            continue
        if len(params) != arity:
            raise AdapterSmokeCheckFailed(
                f"{type(system).__name__}.{name} takes {len(params)} required positional "
                f"argument(s), expected {arity}. runtime_checkable does not catch this, so it "
                f"would otherwise surface deep inside a long run."
            )


def _recorded(out_path: Path) -> dict[str, bool]:
    """instance_id -> abstained, for every row already written.

    Returns the flags, not just the ids, because invariant 3 must still fire on a RESUMED run. A
    resume where every answerable original was already scored would otherwise leave
    `answered_originals` empty and skip the check silently — the failure shape where grepping for
    success turns a failure into no output at all.

    A process killed mid-write (SIGKILL, OOM, power loss — the realistic ways an overnight run
    dies) leaves a truncated final line. That must be treated as "not yet recorded", not fatal —
    the whole point of `--resume` is surviving exactly this. A malformed line that is NOT last is
    a different signal (corruption somewhere in the middle of an otherwise-flushed file, not a
    write interrupted at the tail) and must still be loud rather than silently dropped.

    When the tail is discarded as truncated, the file is rewritten with only the valid lines. The
    caller reopens `out_path` in append ("a") mode afterwards; without this rewrite, a partial
    line with no trailing newline would have the next row glued onto its end, corrupting the
    JSONL format for every future read of a file that was otherwise fine.
    """
    if not out_path.exists():
        return {}
    lines = [line for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    recorded: dict[str, bool] = {}
    valid_lines: list[str] = []
    truncated = False
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise ValueError(
                    f"{out_path}: line {i + 1} of {len(lines)} is not valid JSON and is NOT the "
                    f"last line — this is corruption in the middle of the file, not a truncated "
                    f"trailing write, and must not be silently treated as 'not yet recorded'."
                ) from None
            truncated = True
            break
        recorded[row["instance_id"]] = bool(row["abstained"])
        valid_lines.append(line)
    if truncated:
        out_path.write_text(
            "".join(f"{ln}\n" for ln in valid_lines), encoding="utf-8"
        )
    return recorded


def _cluster_id_of(instance: Instance) -> str:
    # Doc ids are "{cluster_id}/{dia_id}", so the gold id names its own cluster.
    if not instance.gold_doc_ids:
        return ""
    return instance.gold_doc_ids[0].split("/", 1)[0]


def _scope_of(instance: Instance) -> tuple[str, ...]:
    """The clusters this instance's ingested slice is drawn from.

    v2 states this explicitly: `scope_cluster_ids` is the sorted union of the question's own
    conversation plus its distractor conversations (`build_v2.build_v2_instances`). v1 instances
    have an empty `scope_cluster_ids` — `manifest.py`'s documented meaning is "inferred from the
    gold id's own cluster" — so that case falls back to `_cluster_id_of`, which is exactly v1's
    existing behaviour and must not change for the frozen v1 manifest.
    """
    if instance.scope_cluster_ids:
        return tuple(sorted(instance.scope_cluster_ids))
    return (_cluster_id_of(instance),)


def run(
    manifest_path: Path,
    system: MemorySystem,
    out_path: Path,
    *,
    documents: Mapping[str, str],
    cluster_members: Mapping[str, Sequence[str]],
    resume: bool = True,
    expected_digest: str | None = None,
) -> int:
    """Returns the number of instances scored in this invocation.

    Validates the adapter's method SIGNATURES (`_assert_adapter_signatures`) before doing
    anything else, since `run()` — not just `main()` — is what a third-party integration or the
    test suite calls directly. This does NOT functionally smoke-call the adapter: a real
    `ingest`/`query` round trip here would perturb the pinned `ingest_calls` counters. `main()`
    still performs that functional check via `smoke_check()` before ever calling `run()`.

    `expected_digest`, when given, arms invariant 4 (`assert_manifest_digest`) against a
    caller-supplied KNOWN-PUBLISHED digest, before any scoring happens. `read_manifest` already
    refuses a body that does not match its own header — a narrower guarantee, since a forged
    header/body pair that agree with EACH OTHER pass it silently. Proving "the instances being
    scored are the instances that were published" needs a digest that came from somewhere other
    than the file being checked. When `expected_digest` is omitted, this prints one line saying
    the check is not armed — a skipped gate and a passed gate must never look the same.
    """
    _assert_adapter_signatures(system)
    instances, header = read_manifest(manifest_path)
    if expected_digest is not None:
        assert_manifest_digest(instances, {**header, "digest": expected_digest})
    else:
        print(
            "manifest digest check: NOT ARMED (no --expected-digest given) — the loaded "
            "manifest was NOT compared against any known-published digest."
        )
    recorded = _recorded(out_path) if resume else {}

    # Keyed by (scope, excised) — v1's key was (cluster, excised), scoring a question against its
    # own conversation only (what recall/eval/locomo.py already does and what LOCOMO's protocol
    # assumes). v2 widens "cluster" to "scope": the sorted union of clusters in
    # `instance.scope_cluster_ids` when it is set, else the v1 fallback (`_scope_of`). Two
    # questions that share a scope AND an excision set still collapse into one ingest — the same
    # cost discipline as v1, just keyed on a set of clusters instead of one.
    by_state: dict[tuple[tuple[str, ...], tuple[str, ...]], list[Instance]] = {}
    for inst in instances:
        if inst.instance_id in recorded:
            continue
        key = (_scope_of(inst), tuple(sorted(inst.excised_doc_ids)))
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
        for (scope, excised), group in sorted(by_state.items()):
            # The FULL ingested slice: every cluster in scope, own conversation plus distractors.
            # v1's `members` was one cluster; this is its v2 generalisation — a plain union, since
            # v1's scope is always a 1-tuple and this reduces to v1's `members` unchanged there.
            members = tuple(
                sorted(
                    {
                        doc_id
                        for cluster_id in scope
                        for doc_id in cluster_members.get(cluster_id, ())
                    }
                )
            )
            dropped = set(excised)
            keep = [Document(d, documents[d]) for d in members if d not in dropped]
            system.ingest(keep)
            indexed = system.indexed_doc_ids()
            for inst in sorted(group, key=lambda i: i.instance_id):
                assert_excised_absent(inst, indexed)
                # "Did the topic survive at the near rung?" — asked of the question's OWN cluster
                # ONLY, never the full scope. Distractors always survive (they are never excised),
                # so passing the full scope here would make this pass trivially and silently
                # disable the check it exists to run.
                own_cluster = cluster_members.get(_cluster_id_of(inst), ())
                assert_ring_zero_has_survivors(inst, indexed, own_cluster)
                # The POSITIVE check, and the one that matters most. Every other invariant here
                # confirms that what should be gone is gone, which cannot tell a correct excision
                # from an ingest that silently did nothing. A partial ingest passes all of them,
                # abstains on nearly everything, flattens the curve, and would be recorded as an
                # H1 FAIL — retiring the benchmark on a harness bug. This gets the FULL ingested
                # slice (`members`, distractors included) — at r=1.00 this is what proves the
                # distractors are really indexed, the whole point of v2.
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
                            "top_cosine": response.top_cosine,
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
    from benchmarks.ladder.systems.recall_system import DEFAULT_TABLE, DEFAULT_TENANT, RecallSystem

    parser = argparse.ArgumentParser(description="Run a system over the Answerability Ladder.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dsn", required=True)
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help=(
            f"Postgres table this run's RecallSystem owns exclusively (default: {DEFAULT_TABLE}). "
            "Two runs against the same --dsn at the same time MUST use distinct --table and/or "
            "--tenant values -- there is no other isolation between them, and the two would "
            "otherwise collide on the same rows."
        ),
    )
    parser.add_argument(
        "--tenant",
        default=DEFAULT_TENANT,
        help=(
            f"Tenant id this run's RecallSystem scopes count()/delete_sources() by "
            f"(default: {DEFAULT_TENANT}). PgVectorStore scopes both by tenant, so a distinct "
            "--tenant per run is real isolation on a --table two runs happen to share."
        ),
    )
    parser.add_argument(
        "--expected-digest",
        default=None,
        help=(
            "A known-published manifest digest (see manifest.py:manifest_digest) to verify the "
            "loaded --manifest against before any scoring starts. Omitted by default -- when "
            "omitted, the run prints a one-line notice that this check is not armed rather than "
            "silently skipping it."
        ),
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    corpus = load_locomo(args.locomo)
    system = RecallSystem(args.dsn, table=args.table, tenant=args.tenant)
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
        expected_digest=args.expected_digest,
    )
    print(f"scored {n} instances into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
