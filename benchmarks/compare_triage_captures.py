"""How much does a triage result move when the same retrieval is run twice?

Written because a known-answer check turned out to have the wrong known answer.
`PREREGISTRATION-retrieval-triage.md` registers R1 as *"retrieval of the same question twice →
byte-identical ranked list … retrieval has no sampling, so this must hold"*, and never reported a
result. Measured 2026-08-16: **42.5% of query-embedding call pairs differ** (17 of 40, lowest
cosine 0.998545), so two captures of one benchmark are not expected to match and never were.

⚠️ That makes "are they identical?" the wrong question. The right one is how far the published
quantities move, so this reports:

- how many pools and top-k prefixes survive unchanged, which is the churn;
- **how many `missed_any` labels flip**, which is the only movement that changes what is being
  predicted;
- how far `ratio_8_over_1` shifts per row;
- **the headline AUC computed on each capture**, whose difference is the reproducibility interval
  that every number in this line of work has so far been quoted without.

Nothing here is a hypothesis test. It measures an instrument.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.console import use_utf8_output
from benchmarks.analyse_triage import auc
from benchmarks.explore_triage_signal import _labels
from benchmarks.probe_triage_mechanism import published_ratio


def _scores(row: Mapping[str, Any]) -> list[float]:
    return [float(hit["score"]) for hit in row["ranked"]]


def _docs(row: Mapping[str, Any], k: int | None = None) -> list[str]:
    hits = row["ranked"] if k is None else row["ranked"][:k]
    return [str(hit["doc_id"]) for hit in hits]


def _ranked_list(row: Mapping[str, Any], k: int | None = None) -> list[tuple[str, float]]:
    """The ranked list as (document, score) pairs.

    🔑 Scores included, and this is the difference between measuring the phenomenon and missing
    it. What moves between two runs is the QUERY VECTOR, whose first-order effect is on every
    score; a churn statistic built from `doc_id` alone reports "pools identical" on exactly the
    rows where every score, and the feature computed from them, moved. It also collapses two
    distinct chunks of one document, which the pool may legitimately contain.
    """
    hits = row["ranked"] if k is None else row["ranked"][:k]
    return [(str(hit["doc_id"]), float(hit["score"])) for hit in hits]


def check_comparable(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, list[str]]:
    """Refuse two captures that are not two draws of the same measurement.

    Compares the **shared** keys of the two retrieval fingerprints, and reports the rest.

    ⚠️ Shared keys, not equality of the whole fingerprint. Capture 2 records four keys capture 1
    never had (`capture_schema`, `questions`, `limit`, `hnsw_ef_search_multiplier`), so a blanket
    equality check would refuse the exact comparison this module exists to make. Verified against
    the real pair: ten shared keys, all identical, four extra on the candidate.

    Also refuses two captures with the same `retrieval_sha256`. Passing one file as both arguments
    prints a movement of exactly +0.0000 under a heading claiming to measure reproducibility: the
    most attractive-looking result available, produced by the likeliest slip.
    """
    base_prov = baseline.get("_provenance", {})
    cand_prov = candidate.get("_provenance", {})
    base_digest = base_prov.get("retrieval_sha256")
    if base_digest and base_digest == cand_prov.get("retrieval_sha256"):
        raise SystemExit(
            f"both files are the same capture ({base_digest[:12]}…). A movement of zero between "
            "a capture and itself measures nothing; pass two different runs."
        )

    base_fp = json.loads(base_prov.get("retrieval_fingerprint") or "{}")
    cand_fp = json.loads(cand_prov.get("retrieval_fingerprint") or "{}")
    differing = sorted(k for k in set(base_fp) & set(cand_fp) if base_fp[k] != cand_fp[k])
    if differing:
        detail = ", ".join(f"{k}: {base_fp[k]!r} vs {cand_fp[k]!r}" for k in differing)
        raise SystemExit(
            f"the two captures were retrieved under different settings ({detail}). Churn between "
            "them would measure the configuration change, not the run-to-run variation."
        )
    return {
        "only_in_baseline": sorted(set(base_fp) - set(cand_fp)),
        "only_in_candidate": sorted(set(cand_fp) - set(base_fp)),
    }


def compare(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], top_k: int = 8
) -> dict[str, float]:
    """Movement between two captures of the same questions.

    Two denominators, kept separate because they answer different questions. **Churn and feature
    movement are over every shared row** (`n_shared`): pool identity and `ratio_8_over_1` are
    defined without gold, and restricting them would narrow the churn statistic for a reason that
    has nothing to do with churn. **Label flips and the two AUCs are over gold-bearing rows only**
    (`n`), because `missed_any` is undefined without gold and a row that cannot flip must not pad
    the denominator of a flip rate.

    Refuses in three cases: either capture failing to cover the other, a question whose gold set
    disagrees between them, and no gold-bearing row at all.
    """
    missing = sorted(set(baseline) - set(candidate))
    if missing:
        raise SystemExit(
            f"{len(missing)} baseline questions are absent from the candidate capture, e.g. "
            f"{missing[:3]}. Refusing to report movement over an arbitrary overlap."
        )
    # Symmetric. A `--limit 100` baseline against a full 500-question candidate used to be
    # truncated silently, and the report then labelled a 100-row subset "the candidate capture".
    extra = sorted(set(candidate) - set(baseline))
    if extra:
        raise SystemExit(
            f"{len(extra)} candidate questions are absent from the baseline capture, e.g. "
            f"{extra[:3]}. These are not two captures of one question set."
        )

    identical_docs = identical_docs_top_k = 0
    identical_list = identical_list_top_k = flips = 0
    labels_base: list[int] = []
    labels_cand: list[int] = []
    feature_base: list[float] = []
    feature_cand: list[float] = []
    deltas: list[float] = []

    for qid, base in baseline.items():
        cand = candidate[qid]
        # Gold belongs to the benchmark, not to the run. A disagreement would surface downstream
        # as a `missed_any` flip and be attributed to retrieval having moved, which is the one
        # conclusion this module must never manufacture.
        if sorted(base["expected_doc_ids"]) != sorted(cand["expected_doc_ids"]):
            raise SystemExit(
                f"{qid}: expected_doc_ids differ between the two captures "
                f"({sorted(base['expected_doc_ids'])} vs {sorted(cand['expected_doc_ids'])}). "
                "These are not two captures of the same benchmark."
            )

        identical_docs += int(_docs(base) == _docs(cand))
        identical_docs_top_k += int(_docs(base, top_k) == _docs(cand, top_k))
        identical_list += int(_ranked_list(base) == _ranked_list(cand))
        identical_list_top_k += int(_ranked_list(base, top_k) == _ranked_list(cand, top_k))

        ratio_base = published_ratio(_scores(base))
        ratio_cand = published_ratio(_scores(cand))
        deltas.append(abs(ratio_cand - ratio_base))

        label_base = _labels(base, top_k)["missed_any"]
        label_cand = _labels(cand, top_k)["missed_any"]
        if label_base is None or label_cand is None:
            continue  # no gold: undefined label, but the churn above still counted this row

        labels_base.append(label_base)
        labels_cand.append(label_cand)
        flips += int(label_base != label_cand)
        feature_base.append(ratio_base)
        feature_cand.append(ratio_cand)

    n = len(labels_base)
    if not n:
        raise SystemExit("nothing to compare: no question carries gold in both captures")

    # ⚠️ Each AUC is computed against ITS OWN capture's labels. Scoring the candidate's feature
    # against the baseline's labels would hide exactly the movement being measured, because a
    # flipped label is part of how the number moves.
    return {
        "n": float(n),
        "n_shared": float(len(baseline)),
        "identical_doc_order": float(identical_docs),
        "identical_doc_order_top_k": float(identical_docs_top_k),
        "identical_ranked_list": float(identical_list),
        "identical_ranked_list_top_k": float(identical_list_top_k),
        "label_flips": float(flips),
        "label_positive_baseline": float(sum(labels_base)),
        "label_positive_candidate": float(sum(labels_cand)),
        "mean_abs_feature_delta": statistics.fmean(deltas),
        "max_abs_feature_delta": max(deltas),
        "auc_baseline": auc(feature_base, labels_base),
        "auc_candidate": auc(feature_cand, labels_cand),
    }


def main(argv: Sequence[str] | None = None) -> int:
    use_utf8_output()  # argparse prints this module's docstring; cp1252 cannot
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.top_k < 1:
        # `ranked[:0]` labels every row a miss and `ranked[:-1]` redefines the label as "gold
        # missing from all but the last hit". Both print a complete, plausible report.
        parser.error(f"--top-k must be >= 1, got {args.top_k}")
    if args.top_k != 8:
        print(f"⚠️ label depth is top-{args.top_k}, but the feature stays `ratio_8_over_1`, which "
              "is fixed at ranks 8 and 1. The two are no longer at the same depth.")

    base_payload = json.loads(args.baseline.read_text(encoding="utf-8"))
    cand_payload = json.loads(args.candidate.read_text(encoding="utf-8"))
    fingerprint = check_comparable(base_payload, cand_payload)
    result = compare(base_payload["evidence"], cand_payload["evidence"], args.top_k)
    n, shared = result["n"], result["n_shared"]

    for side in ("baseline", "candidate"):
        keys = fingerprint[f"only_in_{side}"]
        if keys:
            print(f"note: fingerprint keys recorded only by the {side}: {', '.join(keys)}")
    print(f"rows compared: {shared:.0f}; of which gold-bearing: {n:.0f}")
    print()
    print("=== churn ===")
    print(f"  ranked lists identical (doc AND score)  {result['identical_ranked_list']:.0f}/"
          f"{shared:.0f} ({result['identical_ranked_list']/shared:.1%})")
    print(f"  top-{args.top_k} identical (doc AND score)         "
          f"{result['identical_ranked_list_top_k']:.0f}/{shared:.0f} "
          f"({result['identical_ranked_list_top_k']/shared:.1%})")
    print(f"  document order identical, scores ignored {result['identical_doc_order']:.0f}/"
          f"{shared:.0f} ({result['identical_doc_order']/shared:.1%})")
    print("  ⚠️ The last line is the WEAKER identity and is not R1's 'byte-identical ranked")
    print("     list'. A perturbed query vector moves every score while leaving order intact.")
    print()
    print("=== what it does to the measurement ===")
    print(f"  missed_any labels flipped  {result['label_flips']:.0f}/{n:.0f} "
          f"({result['label_flips']/n:.1%})")
    print(f"  positives: baseline {result['label_positive_baseline']:.0f}, "
          f"candidate {result['label_positive_candidate']:.0f}")
    print(f"  |delta ratio_8_over_1|     mean {result['mean_abs_feature_delta']:.6f}, "
          f"max {result['max_abs_feature_delta']:.6f}   (over all {shared:.0f} shared rows)")
    print()
    print("=== the reproducibility interval nobody had ===")
    print(f"  AUC on the baseline capture  = {result['auc_baseline']:.4f}")
    print(f"  AUC on the candidate capture = {result['auc_candidate']:.4f}")
    print(f"  movement                     = {result['auc_candidate']-result['auc_baseline']:+.4f}")
    print()
    print("  ⚠️ Two captures give ONE difference, not a distribution. This is the size of a")
    print("     single re-run, and it is a lower bound on what a confidence interval would need.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
