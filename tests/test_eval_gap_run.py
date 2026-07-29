"""Orchestration for the overnight run.

Nothing here is about retrieval. It is about the two ways an unattended multi-hour job wastes a
night: losing completed work when it dies partway, and skipping a corpus so quietly that the final
table looks complete when it is not.
"""
from __future__ import annotations

import json

import pytest

from recall.eval.gap_run import failure_record, pending_datasets, summarise
from tests.conftest import requires_db


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_pending_datasets_skips_corpora_that_already_have_results(tmp_path):
    """Resume, or lose the night. Twenty corpora at ~20 minutes each is most of a night, and a
    crash at corpus fourteen must not re-embed the first thirteen."""
    _write(tmp_path / "nfcorpus.json", {"status": "ok"})
    _write(tmp_path / "scifact.json", {"status": "ok"})
    assert pending_datasets(["nfcorpus", "scifact", "fiqa"], tmp_path) == ["fiqa"]


def test_pending_datasets_does_not_silently_retry_a_failure(tmp_path):
    """A corpus that failed after forty minutes of embedding would burn the night again on retry.

    Failures are recorded and left alone unless retried on purpose. The manifest is what makes
    them loud; automatic retry is what makes them expensive.
    """
    _write(tmp_path / "fiqa.json", {"status": "failed", "error": "boom"})
    assert pending_datasets(["fiqa", "scidocs"], tmp_path) == ["scidocs"]
    assert pending_datasets(["fiqa", "scidocs"], tmp_path, retry_failed=True) == ["fiqa", "scidocs"]


def test_failure_record_keeps_the_corpus_in_the_table_rather_than_dropping_it(tmp_path):
    """A dropped corpus makes an incomplete run look like a complete one.

    The study reports n. If a corpus vanishes because its download 404'd, n silently falls and
    every power calculation in the spec is wrong without anything having errored.
    """
    record = failure_record("cqadupstack-tex", RuntimeError("corpus.jsonl missing"))
    assert record["status"] == "failed"
    assert record["dataset"] == "cqadupstack-tex"
    assert "corpus.jsonl missing" in record["error"]
    assert "RuntimeError" in record["error_type"]


def test_summarise_reports_failures_separately_from_a_shrunken_n(tmp_path):
    # The number that goes into the writeup is `usable`, and it has to be visible next to what was
    # attempted — never inferred from the length of a results list.
    _write(tmp_path / "a.json", {"status": "ok", "dataset": "a"})
    _write(tmp_path / "b.json", {"status": "ok", "dataset": "b"})
    _write(tmp_path / "c.json", {"status": "failed", "dataset": "c", "error": "404"})
    summary = summarise(["a", "b", "c", "d"], tmp_path)
    assert summary["attempted"] == 4
    assert summary["usable"] == 2
    assert summary["failed"] == ["c"]
    assert summary["missing"] == ["d"]


def test_summarise_flags_when_n_falls_below_the_preregistered_power_floor(tmp_path):
    """n is not a detail here. The spec preregistered n~20 because n=8 needs |partial r| >= 0.85
    to survive Holm — a null at low n means 'underpowered', not 'no effect'. If the run degrades
    to that regime the summary has to say so, not leave it to be noticed."""
    for i in range(6):
        _write(tmp_path / f"c{i}.json", {"status": "ok", "dataset": f"c{i}"})
    summary = summarise([f"c{i}" for i in range(6)], tmp_path)
    assert summary["underpowered"] is True
    assert "0.8" in summary["power_note"] or "underpowered" in summary["power_note"].lower()


def test_summarise_is_not_underpowered_at_the_preregistered_size(tmp_path):
    for i in range(18):
        _write(tmp_path / f"c{i}.json", {"status": "ok", "dataset": f"c{i}"})
    assert summarise([f"c{i}" for i in range(18)], tmp_path)["underpowered"] is False


@requires_db
def test_run_corpus_end_to_end_against_real_postgres(tmp_path):
    """materialise -> evaluate(local) -> evaluate(cloud) -> predictors, wired together for real.

    Every other test here checks one seam. This is the only one that proves the seams meet, and it
    is the difference between finding a wiring bug now and finding it at 2am on a rented box with
    the night already paid for. Uses HashingEmbedder for both arms: the question is whether the
    plumbing fits together, not whether retrieval is any good.
    """
    pytest.importorskip("tokenizers", reason="bge_encoder needs the fastembed extra")
    from recall.embeddings import HashingEmbedder
    from recall.eval.gap_run import run_corpus
    from tests.conftest import TEST_DSN

    src = tmp_path / "beir" / "toy"
    src.mkdir(parents=True)
    docs = {f"d{i}": (f"Title {i}", f"body about topic {i} and some detail {i}") for i in range(12)}
    (src / "corpus.jsonl").write_text(
        "\n".join(json.dumps({"_id": i, "title": t, "text": b}) for i, (t, b) in docs.items()),
        encoding="utf-8")
    (src / "queries.jsonl").write_text(
        "\n".join(json.dumps({"_id": f"q{i}", "text": f"tell me about topic {i}"}) for i in range(6)),
        encoding="utf-8")
    (src / "qrels").mkdir()
    (src / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n" + "\n".join(f"q{i}\td{i}\t1" for i in range(6)) + "\n",
        encoding="utf-8")

    record = run_corpus(
        "toy", tmp_path / "beir", tmp_path / "work", TEST_DSN,
        embedders=(HashingEmbedder(dim=64), HashingEmbedder(dim=64)),
        max_docs=12,
    )

    assert record["status"] == "ok"
    # Both arms scored, on every retrieval arm the analysis might use.
    for side in ("local", "cloud"):
        for arm in ("bm25", "dense", "sparse", "hybrid"):
            assert 0.0 <= record["scores"][side][arm] <= 1.0
    # Predictors computed on the materialised corpus, not left as placeholders.
    predictors = record["predictors"]
    assert 0.0 <= predictors["oov_rate"] <= 1.0
    assert predictors["n_documents"] == 12
    assert predictors["n_overlap_pairs"] > 0
    # The frozen parameters travel with the result, so a record can never be read without them.
    assert record["params"]["seed"] == 20260726
    assert record["primary_arm"] == "hybrid"


# --- artifact + credential guards (CCA audit 2026-07-28) -----------------------------------


def test_write_json_refuses_to_emit_a_bare_nan_token():
    """`NaN` is not JSON, and two committed study artifacts contained it.

    Verified before the fix: `results/gap/arguana.json` and `fiqa.json` both carried
    `"code_density": NaN`. Python round-trips that happily, so it looked fine from inside the
    harness and rejected in `jq`, `JSON.parse`, Go, Rust and Postgres `jsonb` — a reviewer who
    cannot open the artifact cannot check the claim.
    """
    import json
    import tempfile
    from pathlib import Path

    from recall.eval.gap_run import write_json

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rec.json"
        write_json(path, {"predictors": {"code_density": float("nan"), "oov_rate": 0.5}})
        raw = path.read_text(encoding="utf-8")
        assert "NaN" not in raw
        # Parsed with the constant hook armed: a strict parser must not need one.
        def _reject(token):  # pragma: no cover - only runs if the guard regressed
            raise AssertionError(f"non-JSON constant emitted: {token}")

        parsed = json.loads(raw, parse_constant=_reject)
        assert parsed["predictors"]["code_density"] is None, "absence is null, not a number"
        assert parsed["predictors"]["oov_rate"] == 0.5


def test_write_json_replaces_the_target_atomically():
    # Sixteen nohup'd workers on a rented box means a kill mid-write is the expected path. A
    # torn file is worse than a missing one, because `pending_datasets` used to treat existence
    # as completion and would skip that corpus as done forever.
    import tempfile
    from pathlib import Path

    from recall.eval.gap_run import write_json

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rec.json"
        write_json(path, {"status": "ok"})
        write_json(path, {"status": "ok", "again": True})
        assert not list(Path(tmp).glob("*.tmp")), "no temp file left behind"


def test_pending_datasets_re_runs_a_torn_result_instead_of_skipping_it_forever():
    """A file that cannot be parsed is not a finished corpus.

    Verified before the fix: `pending_datasets` only stat'd the path, so a half-written record
    was skipped as complete and then crashed `summarise` — after the corpus had already been
    dropped from the study.
    """
    import tempfile
    from pathlib import Path

    from recall.eval.gap_run import pending_datasets

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "scifact.json").write_text('{"status": "ok", "dat', encoding="utf-8")
        assert pending_datasets(["scifact"], out) == ["scifact"]


def test_pending_datasets_re_runs_a_record_with_no_status():
    import tempfile
    from pathlib import Path

    from recall.eval.gap_run import pending_datasets

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "scifact.json").write_text('{"dataset": "scifact"}', encoding="utf-8")
        assert pending_datasets(["scifact"], out) == ["scifact"]


def test_failure_record_does_not_commit_a_dsn_password_or_an_api_key():
    """`results/gap/` is git-tracked and the run holds both credentials in its environment."""
    from recall.eval.gap_run import failure_record

    exc = RuntimeError(
        "connect failed for postgresql://recall:hunter2@10.0.0.4:5432/recall "
        "with key sk-abcdefghijklmnopqrstuvwxyz012345"
    )
    record = failure_record("scifact", exc)
    blob = record["error"] + record["traceback"]
    assert "hunter2" not in blob
    assert "sk-abcdefghijklmnopqrstuvwxyz012345" not in blob
    assert "scifact" in record["dataset"]


def test_power_floor_and_primary_arm_have_exactly_one_definition():
    # Both were declared independently in the runner AND the analyser, and both drive the same
    # preregistered `underpowered` verdict on the same n. A frozen constant with two definitions
    # is not frozen.
    from recall.eval import gap_run, gap_study

    assert gap_run.POWER_FLOOR is gap_study.POWER_FLOOR
    assert gap_run.PRIMARY_ARM is gap_study.PRIMARY_ARM
