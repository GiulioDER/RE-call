"""Near-miss coverage at calibration time: measured and reported, never fed into the certificate.

Uses `tests/test_lite_calibration.py`'s content-word embedder and memos, so certification is
deterministic and needs no model and no database server.

Invariants and the failure each one catches:
- N1 the probes are near-misses by construction: every unknown-entity probe names a code the
  corpus never contains, and every in-vocabulary probe uses only corpus words while no single
  chunk contains all of its content words. The same seed gives the same probes.
- N2 the unknown-entity class is rejected by the unknown-term gate, not by the score: with the gate
  on every probe is rejected; with it off, fewer are.
- N3 auto-calibration reports the coverage with the calibration (outcome, status report, stored by
  calibration id), and a certificate stays certified when the in-vocabulary class is provisional:
  the measurement is report only.
- N4 coverage stored for one calibration is not reported for another.
- N5 a corpus with no markdown headings still yields in-vocabulary probes (2026-10-09: four
  heading-less documents yielded none, so a store certified at 0.979 and trusting 78 of 90
  unanswerable questions got no warning).
- N6 when too few probes of a class can be built, the class is `unmeasured` and the certificate
  is provisional for it, never silently covered.

Node IDs, all in `tests/test_near_miss.py`:
N1 `test_probes_are_near_misses_by_construction`,
N2 `test_unknown_entity_probes_are_rejected_by_the_gate_not_the_score`,
N3 `test_calibration_reports_coverage_and_keeps_its_certificate`,
N4 `test_coverage_belongs_to_its_calibration`,
N5 `test_a_corpus_without_headings_still_gets_in_vocabulary_probes`,
N6 `test_too_few_probes_is_unmeasured_and_provisional`.

Red proof, 2026-10-09, run on a Linux host (not this workstation), each mutation alone, failing in
the named assertion, then restored byte for byte and all four green:
- A1 (N2) `near_miss_coverage` counting only `score < threshold`: "an unknown-entity probe passed
  the gate".
- A2 (N3) `auto_calibrate` rewriting the artifact as uncertified when the in-vocabulary class is
  provisional: the outcome "did not certify".
- A3 (N1) in-vocabulary probes drawing the term from the same chunk: "no in-vocabulary probe could
  be built".
- A4 (N3) `status_report` without the coverage: "the status report does not carry the near-miss
  coverage". Its first run failed with a KeyError instead; the test was changed to assert the key.
- A5 (N4) `LiteStore.near_miss_coverage` ignoring the calibration id: the other id got a report.
On this corpus the in-vocabulary class was rejected 0 of 40 times: score alone does not separate it.

Added the same day, after the pre-registered measurement found A blind on a heading-less corpus:
- B1 (N5) `_base_words` without the fallback to distinctive terms: "a heading-less corpus gave 0
  in-vocabulary probes".
- B2 (N6) the unmeasured state ignored when deciding provisional: "an untested class was reported
  as covered". The first N6 stayed green under B2, because its fake store scored every probe above
  the threshold, so the class was provisional by its rate anyway; the store now scores below it,
  and the test asserts the rate is 1.0 so the fixture keeps isolating the rule.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from recall.calibration_v2 import CalibrationStatus
from recall.eval.vocab import word_tokens
from recall.lite.calibration import auto_calibrate, status_report
from recall.near_miss import near_miss_coverage, near_miss_probes
from recall.wizard.queryset import _STOPWORDS, generate_offline
from tests.test_lite_calibration import ContentWordEmbedder, _store


def _texts(store) -> list[str]:  # noqa: ANN001
    return [c.text for c in store.iter_chunks() if c.text.strip()]


def _asked(texts: list[str]) -> list[str]:
    return [e["query"] for e in generate_offline(texts, per_class=20) if e["answerable"]]


def test_probes_are_near_misses_by_construction(tmp_path: Path) -> None:
    texts = _texts(_store(tmp_path, 48))
    probes = near_miss_probes(texts, _asked(texts), per_class=20, seed=3)
    corpus = set(word_tokens(texts))
    assert len(probes["unknown_entity"]) == 20
    for probe in probes["unknown_entity"]:
        code = probe.rstrip("?").split()[-1]
        assert code.lower() not in corpus, f"{code} occurs in the corpus"
    assert probes["in_vocabulary"], "no in-vocabulary probe could be built"
    chunk_words = [set(word_tokens([t])) for t in texts]
    for probe in probes["in_vocabulary"]:
        content = {w for w in word_tokens([probe]) if w not in _STOPWORDS}
        assert not any(content <= words for words in chunk_words), f"a chunk answers {probe!r}"
    assert probes == near_miss_probes(texts, _asked(texts), per_class=20, seed=3)


def test_unknown_entity_probes_are_rejected_by_the_gate_not_the_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, 48)
    embedder = ContentWordEmbedder()
    outcome = auto_calibrate(store, embedder)
    assert outcome.artifact is not None and outcome.artifact.threshold is not None
    texts = _texts(store)
    probes = near_miss_probes(texts, _asked(texts), per_class=20, seed=0)
    monkeypatch.delenv("RECALL_UNKNOWN_TERM_GATE", raising=False)
    gated = near_miss_coverage(store, embedder, outcome.artifact.threshold, probes)
    monkeypatch.setenv("RECALL_UNKNOWN_TERM_GATE", "0")
    ungated = near_miss_coverage(store, embedder, outcome.artifact.threshold, probes)
    assert gated["unknown_term_gate"] == "checked"
    assert gated["unknown_entity"]["rate"] == 1.0, "an unknown-entity probe passed the gate"
    assert ungated["unknown_entity"]["rate"] < 1.0, "the score alone rejected them all: N2 proves nothing here"


def test_calibration_reports_coverage_and_keeps_its_certificate(tmp_path: Path) -> None:
    store = _store(tmp_path, 48)
    outcome = auto_calibrate(store, ContentWordEmbedder())
    assert outcome.status is CalibrationStatus.CERTIFIED, outcome.reason
    assert outcome.artifact is not None
    assert "near-miss coverage:" in outcome.reason
    report = status_report(store)
    coverage = report.get("near_miss_coverage")
    assert isinstance(coverage, dict), "the status report does not carry the near-miss coverage"
    assert coverage["calibration_id"] == outcome.artifact.calibration_id
    assert coverage["provisional_in_vocabulary"] is True, "this corpus was expected to be provisional"
    assert report["certified"] is True, "a provisional in-vocabulary coverage took the certificate away"
    assert "provisional for in-vocabulary near-misses" in str(report["message"])


_PLAIN = [
    f"The {w1} pump feeds the {w2} tank through a {w3} valve, checked by the {w4} crew."
    for w1, w2, w3, w4 in zip(
        ["alder", "birch", "cedar", "dogwood", "elm", "fir", "ginkgo", "hazel", "juniper", "larch",
         "maple", "nettle", "oak", "pine", "quince", "rowan"],
        ["amber", "bronze", "copper", "denim", "ebony", "flax", "garnet", "henna", "indigo", "jade",
         "khaki", "lilac", "mauve", "navy", "ochre", "plum"],
        ["anvil", "bolt", "chisel", "drill", "easel", "file", "gouge", "hammer", "iron", "jack",
         "knife", "lathe", "mallet", "nail", "oiler", "pliers"],
        ["arctic", "boreal", "coastal", "desert", "estuary", "fjord", "glacial", "heath", "island",
         "jungle", "karst", "lagoon", "moor", "northern", "oasis", "prairie"],
        strict=True,
    )
]


def test_a_corpus_without_headings_still_gets_in_vocabulary_probes() -> None:
    probes = near_miss_probes(_PLAIN, [], per_class=12, seed=1)["in_vocabulary"]
    assert len(probes) >= 10, f"a heading-less corpus gave {len(probes)} in-vocabulary probes"
    chunk_words = [set(word_tokens([t])) for t in _PLAIN]
    for probe in probes:
        content = {w for w in word_tokens([probe]) if w not in _STOPWORDS}
        assert not any(content <= words for words in chunk_words), f"a chunk answers {probe!r}"


class _FakeStore:
    """Every probe scores below the threshold, so a rate alone would call the class covered."""

    def top_cosine(self, vector: list[float]) -> float:
        return 0.10

    def unknown_terms(self, tokens: list[str]) -> list[str]:
        return []


def test_too_few_probes_is_unmeasured_and_provisional() -> None:
    probes = {"unknown_entity": [f"what about x in QK{1000 + i}?" for i in range(12)],
              "in_vocabulary": ["what is the behaviour of alder copper", "where is birch drill described"]}
    report = near_miss_coverage(_FakeStore(), ContentWordEmbedder(), 0.5, probes)
    assert report["in_vocabulary"]["state"] == "unmeasured"
    assert report["in_vocabulary"]["rate"] == 1.0, "the fixture no longer isolates the unmeasured rule"
    assert report["provisional_in_vocabulary"] is True, "an untested class was reported as covered"


def test_coverage_belongs_to_its_calibration(tmp_path: Path) -> None:
    store = _store(tmp_path, 48)
    outcome = auto_calibrate(store, ContentWordEmbedder())
    assert outcome.artifact is not None
    assert store.near_miss_coverage(outcome.artifact.calibration_id) is not None
    assert store.near_miss_coverage("cal_someone_else") is None
