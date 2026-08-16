"""Comparing two captures of the same retrieval must measure movement, not assert identity.

The first pre-registration's R1 check assumed retrieval repeats byte for byte. Measured, 42.5% of
query-embedding call pairs differ, so two captures of the same 500 questions are NOT expected to
match and a comparison that only reports "identical: no" says nothing useful.

What is useful is how far the published quantities move: whether the label flips, how much the
feature shifts, and what happens to the headline AUC. These tests pin that arithmetic, because a
reproducibility number that is itself wrong is worse than not having one.
"""

from __future__ import annotations

import pytest

from benchmarks.compare_triage_captures import check_comparable, compare


def _row(scores: list[float], gold: list[str], docs: list[str] | None = None) -> dict[str, object]:
    docs = docs or [f"d{i}" for i in range(len(scores))]
    return {
        "ranked": [{"doc_id": d, "score": s} for d, s in zip(docs, scores, strict=True)],
        "expected_doc_ids": gold,
        "gap_warning": False,
        "question_type": "basic",
    }


def _pool(top: str, n: int = 12) -> list[str]:
    return [top] + [f"filler{i}" for i in range(n - 1)]


class TestIdentity:
    def test_two_identical_captures_report_no_movement(self) -> None:
        # Both label classes present, or the AUC is legitimately undefined and the assertion
        # below would compare nan to nan and pass for the wrong reason.
        rows = {
            f"q{i}": _row(
                [0.9 - 0.01 * i] * 12,
                ["d0"],
                _pool("d0") if i % 2 else [*[f"f{j}" for j in range(8)], "d0", "x", "y", "z"],
            )
            for i in range(5)
        }

        result = compare(rows, dict(rows), top_k=8)

        assert {result["label_positive_baseline"], result["label_positive_candidate"]} == {3.0}
        assert result["n"] == 5
        assert result["identical_ranked_list"] == 5
        assert result["identical_ranked_list_top_k"] == 5
        assert result["label_flips"] == 0
        assert result["auc_baseline"] == pytest.approx(result["auc_candidate"])
        assert result["max_abs_feature_delta"] == pytest.approx(0.0)

    def test_a_reordered_pool_is_not_identical_but_need_not_flip_a_label(self) -> None:
        """The observed case: the pool churns, the top-8 changes, the label survives."""
        base = {"q0": _row([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                           ["d0"], _pool("d0", 9))}
        # same documents, two of them swapped inside the top-8; d0 is still present
        cand = {"q0": _row([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                           ["d0"], ["d0", "filler1", "filler0"] + [f"filler{i}" for i in range(2, 8)])}

        result = compare(base, cand, top_k=8)

        assert result["identical_ranked_list"] == 0
        assert result["label_flips"] == 0


class TestLabelFlips:
    def test_gold_leaving_the_top_k_counts_as_a_flip(self) -> None:
        base = {"q0": _row([0.9] * 12, ["d0"], _pool("d0"))}
        cand = {"q0": _row([0.9] * 12, ["d0"], [*[f"filler{i}" for i in range(8)], "d0",
                                                *[f"x{i}" for i in range(3)]])}

        result = compare(base, cand, top_k=8)

        assert result["label_flips"] == 1
        assert result["label_positive_baseline"] == 0
        assert result["label_positive_candidate"] == 1

    def test_a_row_without_gold_is_excluded_from_every_denominator(self) -> None:
        """`missed_any` is undefined without gold, and counting those rows would deflate the
        flip rate by padding it with rows that cannot flip."""
        rows = {"q0": _row([0.9] * 12, ["d0"], _pool("d0")), "q1": _row([0.9] * 12, [])}

        result = compare(rows, dict(rows), top_k=8)

        assert result["n"] == 1


class TestFeatureMovement:
    def test_the_feature_delta_is_reported_per_row_and_at_its_maximum(self) -> None:
        base = {"q0": _row([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.5], ["d0"]),
                "q1": _row([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2], ["d0"])}
        cand = {"q0": _row([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.9], ["d0"]),  # 0.5 -> 0.9
                "q1": _row([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2], ["d0"])}  # unchanged

        result = compare(base, cand, top_k=8)

        assert result["max_abs_feature_delta"] == pytest.approx(0.4)
        assert result["mean_abs_feature_delta"] == pytest.approx(0.2)


class TestEachAucUsesItsOwnCapturesLabels:
    """🔑 Mutation-driven. Scoring the candidate's feature against the BASELINE's labels passed
    every other test in this file, and it is the one error that would defeat the purpose: a
    flipped label is part of how the number moves, so pinning the labels to the baseline reports
    the feature drifting while pretending the target stood still."""

    @staticmethod
    def _captures() -> tuple[dict[str, object], dict[str, object]]:
        # q1's gold leaves the top-8 in the candidate, so its label flips 0 -> 1.
        base = {
            "q0": _row([1.0] + [0.9] * 11, ["d0"], _pool("d0")),
            "q1": _row([1.0] + [0.5] * 11, ["d0"], _pool("d0")),
            "q2": _row([1.0] + [0.8] * 11, ["d0"], [*[f"f{i}" for i in range(8)], "d0", "x", "y", "z"]),
            "q3": _row([1.0] + [0.2] * 11, ["d0"], [*[f"g{i}" for i in range(8)], "d0", "x", "y", "z"]),
        }
        cand = dict(base)
        cand["q1"] = _row([1.0] + [0.5] * 11, ["d0"],
                          [*[f"h{i}" for i in range(8)], "d0", "x", "y", "z"])
        return base, cand

    def test_the_flip_is_seen(self) -> None:
        base, cand = self._captures()

        result = compare(base, cand, top_k=8)

        assert result["label_flips"] == 1
        assert result["label_positive_baseline"] == 2
        assert result["label_positive_candidate"] == 3

    def test_the_candidate_auc_is_computed_against_the_candidate_labels(self) -> None:
        from benchmarks.analyse_triage import auc as _auc
        from benchmarks.probe_triage_mechanism import published_ratio

        base, cand = self._captures()
        order = list(base)
        cand_features = [published_ratio([float(h["score"]) for h in cand[q]["ranked"]])
                         for q in order]
        cand_labels = [1 if q in {"q1", "q2", "q3"} else 0 for q in order]

        result = compare(base, cand, top_k=8)

        assert result["auc_candidate"] == pytest.approx(_auc(cand_features, cand_labels))
        # and the two captures must not be reported as agreeing when a label moved under them
        assert result["auc_candidate"] != pytest.approx(result["auc_baseline"])


class TestChurnSeesScoresNotJustDocuments:
    """🔑 The measured phenomenon is a perturbed QUERY VECTOR, whose first-order effect is on the
    scores. A churn statistic computed from doc_ids alone is blind to precisely that, and would
    print "pools identical 100%" on rows where every score, and the feature, moved."""

    @staticmethod
    def _shifted() -> tuple[dict[str, object], dict[str, object]]:
        base = {"q0": _row([0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.50, 0.1, 0.1, 0.1, 0.1],
                           ["d0"], _pool("d0"))}
        # identical document order, every score moved: exactly the embedder-perturbation shape
        cand = {"q0": _row([0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.10, 0.1, 0.1, 0.1, 0.1],
                           ["d0"], _pool("d0"))}
        return base, cand

    def test_document_order_can_be_identical_while_the_ranked_list_is_not(self) -> None:
        base, cand = self._shifted()

        result = compare(base, cand, top_k=8)

        assert result["identical_doc_order"] == 1
        assert result["identical_ranked_list"] == 0
        assert result["max_abs_feature_delta"] > 0.4

    def test_the_top_k_pair_is_reported_the_same_two_ways(self) -> None:
        base, cand = self._shifted()

        result = compare(base, cand, top_k=8)

        assert result["identical_doc_order_top_k"] == 1
        assert result["identical_ranked_list_top_k"] == 0


class TestChurnIsNotGatedOnGold:
    """Pool identity and `ratio_8_over_1` are defined on a row with no gold. Restricting them to
    gold-bearing rows narrows the churn statistic for a reason unrelated to churn."""

    def test_a_gold_free_row_still_counts_toward_churn_but_not_toward_the_flip_rate(self) -> None:
        base = {"q0": _row([0.9] * 12, ["d0"], _pool("d0")), "q1": _row([0.9] * 12, [])}
        cand = {"q0": _row([0.9] * 12, ["d0"], _pool("d0")),
                "q1": _row([0.1] * 12, [])}  # scores moved on the gold-free row

        result = compare(base, cand, top_k=8)

        assert result["n_shared"] == 2      # churn denominator
        assert result["n"] == 1             # label denominator, gold-bearing only
        assert result["identical_ranked_list"] == 1


class TestComparability:
    """Two captures of DIFFERENT configurations are not two draws of one measurement."""

    @staticmethod
    def _payload(fingerprint: dict[str, object], digest: str = "aaa") -> dict[str, object]:
        import json as _json
        return {"_provenance": {"retrieval_fingerprint": _json.dumps(fingerprint, sort_keys=True),
                                "retrieval_sha256": digest}, "evidence": {}}

    def test_extra_keys_on_one_side_are_allowed_because_the_schema_grew(self) -> None:
        """⚠️ Capture 2 records four fingerprint keys capture 1 never had (`capture_schema`,
        `questions`, `limit`, `hnsw_ef_search_multiplier`). A blanket equality check would refuse
        the exact comparison this tool exists to make."""
        old = self._payload({"table": "t", "pool_k": 200})
        new = self._payload({"table": "t", "pool_k": 200, "capture_schema": 2, "limit": None}, "bbb")

        report = check_comparable(old, new)

        assert report["only_in_candidate"] == ["capture_schema", "limit"]

    def test_a_shared_key_that_differs_is_refused(self) -> None:
        old = self._payload({"table": "t", "pool_k": 200})
        new = self._payload({"table": "t", "pool_k": 50}, "bbb")

        with pytest.raises(SystemExit, match="pool_k"):
            check_comparable(old, new)

    def test_the_same_capture_twice_is_refused_rather_than_reporting_zero_movement(self) -> None:
        """The most attractive possible result produced by the most likely operator slip."""
        same = self._payload({"table": "t"}, "digest-1")

        with pytest.raises(SystemExit, match="same capture"):
            check_comparable(same, dict(same))


class TestRefusals:
    def test_a_question_missing_from_the_candidate_is_refused_not_skipped(self) -> None:
        """A partial candidate is the resumable-run failure mode. Comparing the overlap and
        reporting a rate over it would quietly answer a different question than the one asked."""
        base = {"q0": _row([0.9] * 12, ["d0"]), "q1": _row([0.9] * 12, ["d0"])}

        with pytest.raises(SystemExit, match="q1"):
            compare(base, {"q0": _row([0.9] * 12, ["d0"])}, top_k=8)

    def test_extra_questions_in_the_candidate_are_refused_too(self) -> None:
        """Silently truncating to the baseline's keys reports an AUC labelled 'the candidate
        capture' that is really an unnamed subset of it."""
        base = {"q0": _row([0.9] * 12, ["d0"])}

        with pytest.raises(SystemExit, match="q1"):
            compare(base, {"q0": _row([0.9] * 12, ["d0"]), "q1": _row([0.9] * 12, ["d0"])}, top_k=8)

    def test_a_gold_set_that_disagrees_between_captures_is_refused(self) -> None:
        """Gold is a property of the benchmark, not of the run. A gold difference would surface as
        a label flip and be attributed to retrieval having moved."""
        base = {"q0": _row([0.9] * 12, ["d0"], _pool("d0"))}
        cand = {"q0": _row([0.9] * 12, ["d1"], _pool("d0"))}

        with pytest.raises(SystemExit, match="expected_doc_ids"):
            compare(base, cand, top_k=8)

    def test_an_empty_comparison_is_refused_rather_than_dividing_by_zero(self) -> None:
        with pytest.raises(SystemExit, match="nothing to compare"):
            compare({}, {}, top_k=8)
