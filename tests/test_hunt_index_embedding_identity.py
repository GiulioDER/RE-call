"""Regression tests for the 2026-10-31 CCA hunt over indexing, embedders and schema.

Each test names the finding it pins, the pre-fix baseline or mutation it was proved red against,
and the failure it produced. None needs a database or a network.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from recall.embedding_core import embedding_profile
from recall.embeddings import HashingEmbedder
from recall.index import Indexer, candidate_files
from recall.security_policy import AccessContext, SourceRule, SourceSecurityPolicy


class _Store:
    def __init__(self) -> None:
        self.chunks: list[object] = []

    def source_content_hashes(self) -> dict[str, str]:
        return {}

    def replace_sources(self, sources, chunks, embeddings):  # type: ignore[no-untyped-def]
        self.chunks.extend(chunks)
        return len(chunks)

    def analyze_if_stale(self, modified):  # type: ignore[no-untyped-def]
        return True

    def delete_sources(self, sources):  # type: ignore[no-untyped-def]
        return 0


def _policy() -> SourceSecurityPolicy:
    return SourceSecurityPolicy(
        (
            SourceRule(
                "finance", classification="confidential", principals=frozenset({"alice"}),
                redactions=("email", "secret"),
            ),
            SourceRule("public", classification="public"),
        )
    )


def test_a_link_to_an_excluded_file_is_not_a_candidate(tmp_path: Path) -> None:
    """CCA SEC-001: the default scan judges the file a link points at, not only the link's name.

    Red against `6eee13d87` (pre-fix `recall.index._directory_candidates`): node
    tests/test_hunt_index_embedding_identity.py::test_a_link_to_an_excluded_file_is_not_a_candidate
    fails on "a link renamed .env into the corpus".
    """
    (tmp_path / ".env").write_text("API_KEY=do-not-index", encoding="utf-8")
    (tmp_path / "real.md").write_text("# a real memo", encoding="utf-8")
    try:
        (tmp_path / "notes.md").symlink_to(tmp_path / ".env")
    except OSError:
        pytest.skip("this filesystem cannot create a symlink")
    names = [path.name for path in candidate_files(tmp_path)]
    assert "notes.md" not in names, "a link renamed .env into the corpus"
    assert "real.md" in names


def _stub_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    class SentenceTransformer:
        def __init__(self, model: str, **_kwargs: object) -> None:
            self.model = model

        def get_sentence_embedding_dimension(self) -> int:
            return 4

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = SentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_retraining_an_st_model_in_place_changes_its_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CCA DAT-002: `st:<dir>` identity follows the weights in the directory, not its name.

    The fingerprint keys the embedding cache and the index skip guard, so a model retrained into
    the same directory must not keep it. Red against `6eee13d87` (pre-fix
    `recall.embedding_core.legacy_embedding_profile`, constant digest): node
    tests/test_hunt_index_embedding_identity.py::test_retraining_an_st_model_in_place_changes_its_fingerprint
    fails on "retrained weights kept the old fingerprint".
    """
    from recall.embedding_providers.local import SentenceTransformerEmbedder

    _stub_sentence_transformers(monkeypatch)
    model = tmp_path / "finetune-model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"first training run")
    before = embedding_profile(SentenceTransformerEmbedder(str(model))).fingerprint()
    again = embedding_profile(SentenceTransformerEmbedder(str(model))).fingerprint()
    (model / "weights.bin").write_bytes(b"second training run, longer")
    after = embedding_profile(SentenceTransformerEmbedder(str(model))).fingerprint()
    assert before == again, "the same weights gave two fingerprints"
    assert after != before, "retrained weights kept the old fingerprint"


def test_generation_context_policy_follows_the_registry_derivation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CCA CODE-001: a generation pipeline's context version is checked by the registry's rule.

    The registry bumping its context policy version must not make every context build fail.
    Red against `6eee13d87` (pre-fix `recall.generations._context_policy_for_pipeline`, inline
    `-v1`): node
    tests/test_hunt_index_embedding_identity.py::test_generation_context_policy_follows_the_registry_derivation
    fails with GenerationError "does not match context mode".
    """
    import recall.generations as generations
    from recall.lineage import ChunkerIdentity, EmbedderIdentity, PipelineIdentity

    def bumped(mode: str, policy_version: str = "v2") -> str:
        return "raw-v1" if mode == "none" else f"context-{mode}-v2"

    monkeypatch.setattr(generations, "context_version_for", bumped, raising=False)
    pipeline = PipelineIdentity(
        EmbedderIdentity(
            "fixture", "model-a", 64, revision="model-a-commit",
            context_mode="document", context_version="context-document-v2",
        ),
        ChunkerIdentity("paragraph-pack", 1, {"max_chars": 800, "overlap": 80}),
        fts_configuration={"language": "english", "schema_version": 1},
    )
    assert generations._context_policy_for_pipeline(pipeline).mode == "document"


def test_a_repeated_profile_id_is_refused_rather_than_replaced() -> None:
    """CCA CODE-005: the registry refuses two entries with one profile id.

    Red by mutation of `recall.embedding_registry._unique_profiles` back to the dict comprehension
    it replaced (`return {entry.profile_id: entry for entry in entries}`): node
    tests/test_hunt_index_embedding_identity.py::test_a_repeated_profile_id_is_refused_rather_than_replaced
    fails on "DID NOT RAISE ValueError".
    """
    from dataclasses import replace

    from recall.embedding_registry import REGISTERED_PROFILES, _unique_profiles

    original = next(iter(REGISTERED_PROFILES.values()))
    copy = replace(original, model_name=original.model_name + "-copy")
    with pytest.raises(ValueError, match="registered twice"):
        _unique_profiles((original, copy))
    assert len(REGISTERED_PROFILES) == len(_unique_profiles(tuple(REGISTERED_PROFILES.values())))


def _link(link, target) -> None:
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("this filesystem cannot create a symlink")


def test_indexer_refuses_a_link_to_a_policy_denied_file(tmp_path) -> None:
    """CCA SEC-001: the policy judges the file a link READS, not only the link's own path.

    `public/notes.md` links to `.vault/payroll.md`, which no rule allows; the walk skips the
    hidden folder, so the link is the only path the policy saw. Red against `6eee13d87` (pre-fix
    `Indexer.index_path`, policy decided on the link path alone): node
    tests/test_hunt_index_embedding_identity.py::test_indexer_refuses_a_link_to_a_policy_denied_file fails on
    "DID NOT RAISE PermissionError", the payroll text indexed under the public rule.
    """
    root = tmp_path / "corpus"
    (root / "public").mkdir(parents=True)
    (root / ".vault").mkdir()
    (root / ".vault" / "payroll.md").write_text("salary of bob is 90000", encoding="utf-8")
    _link(root / "public" / "notes.md", root / ".vault" / "payroll.md")
    store = _Store()
    indexer = Indexer(
        store, HashingEmbedder(dim=64), security_policy=_policy(),
        security_context=AccessContext("alice", "tenant", clearance="confidential"),
    )
    with pytest.raises(PermissionError, match="which is denied"):
        indexer.index_path(root)
    assert not store.chunks


def test_indexer_refuses_a_link_that_relabels_a_file_under_another_rule(tmp_path) -> None:
    """CCA SEC-001: a link may not move a file from one policy rule to another.

    `public/budget.md` links to `finance/budget.md`, which alice may read, but under the
    confidential rule and its redactions. Red against `6eee13d87`: node
    tests/test_hunt_index_embedding_identity.py::test_indexer_refuses_a_link_that_relabels_a_file_under_another_rule
    fails on "DID NOT RAISE PermissionError", the file indexed as public with no redaction.
    """
    root = tmp_path / "corpus"
    (root / "public").mkdir(parents=True)
    (root / "finance").mkdir()
    target = root / "finance" / "budget.md"
    target.write_text("contact alice@example.com", encoding="utf-8")
    _link(root / "public" / "budget.md", target)
    store = _Store()
    indexer = Indexer(
        store, HashingEmbedder(dim=64), security_policy=_policy(),
        security_context=AccessContext("alice", "tenant", clearance="confidential"),
    )
    with pytest.raises(PermissionError, match="different source policy rule"):
        indexer.index_path(root)
    assert not store.chunks


def test_indexer_refuses_a_link_between_rules_that_agree_on_classification(tmp_path: Path) -> None:
    """CCA SEC-001, the gap the commit security review found in the first fix.

    Two rules with the same classification and no redactions, readable by different principals:
    a link under `team-b/` to a file under `team-a/` would be stored and served under team B's
    rule. Red against `d8179b678` (the first fix, which compared classification and redactions
    only): node
    tests/test_hunt_index_embedding_identity.py::test_indexer_refuses_a_link_between_rules_that_agree_on_classification
    fails on "DID NOT RAISE PermissionError".
    """
    root = tmp_path / "corpus"
    (root / "team-a").mkdir(parents=True)
    (root / "team-b").mkdir()
    target = root / "team-a" / "plan.md"
    target.write_text("team a roadmap", encoding="utf-8")
    _link(root / "team-b" / "plan.md", target)
    policy = SourceSecurityPolicy(
        (
            SourceRule("team-a", classification="internal", principals=frozenset({"alice"})),
            SourceRule("team-b", classification="internal", principals=frozenset({"alice", "bob"})),
        )
    )
    store = _Store()
    indexer = Indexer(
        store, HashingEmbedder(dim=64), security_policy=policy,
        security_context=AccessContext("alice", "tenant", clearance="internal"),
    )
    with pytest.raises(PermissionError, match="different source policy rule"):
        indexer.index_path(root)
    assert not store.chunks


def test_a_link_within_one_rule_still_indexes(tmp_path: Path) -> None:
    """CCA STAKES-001 (gap-fix run): only a link between rules is refused, not every link.

    `public/latest.md` links to `public/2026-10-31.md`; one rule governs both, so the link is
    indexed as before. Red by mutation of `Indexer._index_path` to the blanket refusal this run
    replaced (the rule comparison forced true, so any link to another path raises): node
    tests/test_hunt_index_embedding_identity.py::test_a_link_within_one_rule_still_indexes fails
    with PermissionError "which a different source policy rule governs".
    """
    root = tmp_path / "corpus"
    (root / "public").mkdir(parents=True)
    target = root / "public" / "2026-10-31.md"
    target.write_text("the current plan", encoding="utf-8")
    _link(root / "public" / "latest.md", target)
    store = _Store()
    Indexer(
        store, HashingEmbedder(dim=64), security_policy=_policy(),
        security_context=AccessContext("alice", "tenant", clearance="confidential"),
    ).index_path(root)
    assert {Path(chunk.source).name for chunk in store.chunks} == {"2026-10-31.md", "latest.md"}
