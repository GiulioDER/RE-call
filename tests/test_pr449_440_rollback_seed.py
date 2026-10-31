"""Carry-forward must continue pending builds without undoing an incident rollback."""

from __future__ import annotations

from recall.generations import GenerationManager
from recall.lineage import IndexManifestV1
from tests.conftest import requires_db
from tests.test_generations import (  # noqa: F401 - shared DB fixture
    _Embedder,
    _manifest,
    _pipeline,
    _reader,
    _ready,
    manager,
)


@requires_db
def test_rollback_seed_excludes_previously_active_ready_but_keeps_pending(
    manager: GenerationManager,  # noqa: F811
) -> None:
    """DAT-001, baseline 160a23a3f: servable_manifest resurrects the rolled-away build.

    Build and promote A and then B, roll back to A, and ask the real carry-forward consumer for
    its manifest. The intended red assertion compares that manifest with A's; the baseline returns
    B's because rollback makes B READY while retaining its newer creation time. Controls in this
    same node cover repeated rollback and continuation of never-activated wizard and desktop builds.
    No calibration or model download is needed for this test-environment lifecycle.
    """
    def ready(version: str, body: bytes) -> tuple[str, IndexManifestV1]:
        manifest = _manifest(
            manager.tenant_id, body, version=version, corpus_version=version
        )
        generation_id = _ready(
            manager, manifest, _pipeline("model-a"), _reader(manifest, body), _Embedder(1)
        )
        return generation_id, manifest

    first, first_manifest = ready("wizard-initial", b"the original working corpus")
    assert manager.servable_manifest() == first_manifest, (
        "a never-activated pending build must remain usable before the first promotion"
    )
    manager.promote(first, unsafe_development=True)

    second, second_manifest = ready("desktop-later", b"a newer corpus rolled away after an incident")
    assert manager.servable_manifest() == second_manifest, (
        "a pending desktop build must be continued even when an older generation is active"
    )
    manager.promote(second, unsafe_development=True)
    assert manager.rollback() == first
    assert manager.active_generation_id() == first

    assert manager.servable_manifest() == first_manifest, (
        "carry-forward must use the active corpus after rollback, not the newer READY generation "
        "that already served and was rolled away"
    )

    # Rollback is reversible. Whichever generation is now active supplies the base; the formerly
    # active READY generation must never win only because it was created later.
    assert manager.rollback() == second
    assert manager.servable_manifest() == second_manifest
    assert manager.rollback() == first
    assert manager.servable_manifest() == first_manifest

    # PR446 intentionally continues a pending wizard build too. Restricting the query to a
    # desktop corpus-version prefix would fix the rollback case while breaking this contract.
    _wizard, wizard_manifest = ready("wizard-pending", b"a pending wizard corpus")
    assert manager.servable_manifest() == wizard_manifest
    _desktop, desktop_manifest = ready("desktop-pending", b"a pending desktop upload")
    assert manager.servable_manifest() == desktop_manifest
