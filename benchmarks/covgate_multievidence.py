"""ALL-evidence coverage by depth, per LOCOMO category, over an already-indexed table.

::

    python -m benchmarks.covgate_multievidence --table covgate_a_chunks --candidate-k 20 \
        --depths 1,3,5,10,20 --verify benchmarks/results/covgate_armA_ck20.json

Prior work: searched with ``docs_search(source_type="memory", ...)`` before writing this.
[[project-recall-nearmiss-signal-exhaustion-2026-07-29]] §7 measured a pooled LOCOMO depth curve
(hit@50 0.910 at candidate_k=250) and [[project-recall-mtrag-retrieval-coverage-bottleneck-2026-08-06]]
supplies the ~0.95 saturation threshold. Neither breaks the curve out by category, and neither
scores ALL-evidence coverage, which is what this module adds.

Why this exists
---------------
``recall.eval.locomo`` scores ``hit@k`` as ``any(e in retrieved for e in evidence)``. For a
single-evidence question that is the right criterion. For LOCOMO **cat1** it is not: cat1 carries a
mean of **3.13** evidence turns, so "any" credits a question as a hit when the retriever found one
of three needed turns and the generator still cannot answer it. An any-evidence curve therefore
OVERSTATES coverage exactly where multi-hop questions live, which is the population the SPLADE
decision turns on.

This module recomputes both criteria from the same ranking:

- ``any``  — reproduces the harness exactly. It is the apparatus check, not a result.
- ``all``  — every evidence turn present at depth d. What a multi-hop answer actually requires.
- ``frac`` — mean fraction of a question's evidence turns present at depth d.

The verification pass
---------------------
``--verify`` reads a report produced by ``recall.eval.locomo`` and asserts that the ``any`` column
reproduces its ``depth_curve`` cell for cell. This module reimplements the retrieval loop, so
without that check a divergence in tenant naming, candidate_k, embedder or truncation order would
silently produce a different number and be read as a finding. A reimplementation that is never
checked against the thing it reimplements is a second opinion from an unverified instrument.

It reads an EXISTING table and never indexes. Re-indexing to measure would double a corpus that
``index_conversation`` deliberately refuses to double.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from recall.eval.labelled import _make_embedder
from recall.eval.locomo import (
    ANSWERABLE_CATEGORIES,
    CATEGORY_NAMES,
    _hit_by_depth,
    _retrieved_dia_ids,
)
from recall.eval.metrics import wilson_ci
from recall.retriever import HybridRetriever
from recall.store import PgVectorStore

DEFAULT_DSN = "postgresql://recall:recall@localhost:55432/recall"


def _rate(flags: list[bool]) -> dict[str, Any]:
    lo, hi = wilson_ci(flags)
    return {
        "n": len(flags),
        "rate": round(sum(flags) / len(flags), 4) if flags else float("nan"),
        "ci95": [round(lo, 4), round(hi, 4)],
    }


def score(
    data_path: Path, *, dsn: str, table: str, candidate_k: int, depths: list[int], embedder_name: str
) -> dict[str, Any]:
    conversations = json.loads(Path(data_path).read_text(encoding="utf-8"))
    embedder = _make_embedder(embedder_name)
    max_d = max(depths)

    any_by: dict[int, dict[int, list[bool]]] = {d: defaultdict(list) for d in depths}
    all_by: dict[int, dict[int, list[bool]]] = {d: defaultdict(list) for d in depths}
    frac_by: dict[int, dict[int, list[float]]] = {d: defaultdict(list) for d in depths}
    n_evidence: dict[int, list[int]] = defaultdict(list)
    # Questions still missing at least one evidence turn at the deepest depth scored, kept so the
    # decision can be made on an absolute count and not only on a rate.
    residual: list[dict[str, Any]] = []

    for i, conv in enumerate(conversations):
        sample_id = conv.get("sample_id") or f"conv{i}"
        tenant = f"locomo-{sample_id}"
        with PgVectorStore(dsn, dim=embedder.dim, tenant=tenant, table=table) as store:
            retriever = HybridRetriever(store, embedder, candidate_k=candidate_k, reranker=None)
            for q in conv.get("qa") or []:
                cat = q.get("category")
                question = q.get("question")
                if not question or cat not in ANSWERABLE_CATEGORIES:
                    continue
                evidence = [e for e in (q.get("evidence") or []) if isinstance(e, str)]
                if not evidence:
                    # Same skip the harness applies: no gold to score against.
                    continue

                hits = retriever.search(question, k=max_d).hits
                by_depth = _hit_by_depth(hits, evidence, depths)
                n_evidence[cat].append(len(evidence))

                for d in depths:
                    # Truncate the CHUNK hits, then derive ids — the same order the harness uses.
                    # Slicing the id list instead would reach deeper than depth d ever returned.
                    got = set(_retrieved_dia_ids(hits[:d]))
                    found = sum(1 for e in evidence if e in got)
                    any_by[d][cat].append(by_depth[d])
                    all_by[d][cat].append(found == len(evidence))
                    frac_by[d][cat].append(found / len(evidence))

                deepest = set(_retrieved_dia_ids(hits[:max_d]))
                missing = [e for e in evidence if e not in deepest]
                if missing:
                    residual.append(
                        {
                            "sample_id": sample_id,
                            "category": CATEGORY_NAMES[cat],
                            "question": question,
                            "n_evidence": len(evidence),
                            "n_missing": len(missing),
                        }
                    )

    def _pack(store: dict[int, dict[int, list[Any]]], as_rate: bool) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for d in depths:
            row: dict[str, Any] = {}
            for cat in ANSWERABLE_CATEGORIES:
                vals = store[d].get(cat)
                if not vals:
                    continue
                row[CATEGORY_NAMES[cat]] = (
                    _rate(list(vals)) if as_rate else round(sum(vals) / len(vals), 4)
                )
            pooled = [v for cat in ANSWERABLE_CATEGORIES for v in store[d].get(cat, [])]
            if pooled:
                row["OVERALL"] = _rate(pooled) if as_rate else round(sum(pooled) / len(pooled), 4)
            out[str(d)] = row
        return out

    return {
        "table": table,
        "candidate_k": candidate_k,
        "embedder": embedder_name,
        "depths": depths,
        "mean_evidence_turns": {
            CATEGORY_NAMES[c]: round(sum(v) / len(v), 2) for c, v in n_evidence.items() if v
        },
        "any_evidence": _pack(any_by, as_rate=True),
        "all_evidence": _pack(all_by, as_rate=True),
        "mean_fraction_covered": _pack(frac_by, as_rate=False),
        "residual_incomplete_at_max_depth": {
            "n_questions": len(residual),
            "by_category": {
                name: sum(1 for r in residual if r["category"] == name)
                for name in sorted({r["category"] for r in residual})
            },
            "examples": residual[:15],
        },
    }


def verify_against(report: dict[str, Any], harness_path: Path) -> list[str]:
    """Assert this module's ``any`` column reproduces the harness's ``depth_curve``.

    Returns the list of mismatches; empty means the reimplementation agrees cell for cell.
    """
    harness = json.loads(Path(harness_path).read_text(encoding="utf-8"))
    curve = harness.get("depth_curve") or {}
    problems: list[str] = []
    for depth_key, row in report["any_evidence"].items():
        harness_row = curve.get(depth_key) or curve.get(int(depth_key))
        if harness_row is None:
            problems.append(f"depth {depth_key}: absent from the harness report")
            continue
        for name, cell in row.items():
            expected = harness_row.get(name)
            if expected is None:
                continue
            got = cell["rate"] if isinstance(cell, dict) else cell
            expected_rate = expected.get("rate") if isinstance(expected, dict) else expected
            if expected_rate is None:
                continue
            if abs(float(got) - float(expected_rate)) > 5e-4:
                problems.append(
                    f"depth {depth_key} {name}: this module {got} vs harness {expected_rate}"
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m benchmarks.covgate_multievidence")
    p.add_argument("--data", type=Path, default=Path("locomo10.json"))
    p.add_argument("--dsn", default=DEFAULT_DSN)
    p.add_argument("--table", required=True)
    p.add_argument("--candidate-k", type=int, required=True)
    p.add_argument("--depths", required=True, help="comma-separated, e.g. 1,3,5,10,20")
    p.add_argument("--embedder", default="fastembed")
    p.add_argument("--verify", type=Path, default=None, help="a recall.eval.locomo report to check against")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    depths = sorted({int(x) for x in args.depths.split(",") if x.strip()})
    report = score(
        args.data,
        dsn=args.dsn,
        table=args.table,
        candidate_k=args.candidate_k,
        depths=depths,
        embedder_name=args.embedder,
    )

    if args.verify:
        problems = verify_against(report, args.verify)
        report["verified_against"] = str(args.verify)
        report["verification_problems"] = problems
        if problems:
            print("APPARATUS CHECK FAILED — the any-evidence column does not reproduce the harness:")
            for line in problems[:10]:
                print("  " + line)
            print("Refusing to report all-evidence numbers from an instrument that disagrees.")
            return 1
        print(f"apparatus check PASSED: any-evidence reproduces {args.verify} cell for cell")

    print(f"\nmean evidence turns per category: {report['mean_evidence_turns']}")
    for label, key in (("ANY evidence turn", "any_evidence"), ("ALL evidence turns", "all_evidence")):
        print(f"\n{label} — hit@d")
        cats = ["cat1", "cat2-temporal", "cat3", "cat4", "OVERALL"]
        print("  d    " + "".join(f"{c:<16}" for c in cats))
        for d in depths:
            row = report[key][str(d)]
            cells = "".join(
                f"{(row[c]['rate'] if c in row else float('nan')):<16.3f}" for c in cats
            )
            print(f"  {d:<5}{cells}")

    res = report["residual_incomplete_at_max_depth"]
    print(f"\nstill missing >=1 evidence turn at depth {max(depths)}: "
          f"{res['n_questions']} questions {res['by_category']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"full report -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
