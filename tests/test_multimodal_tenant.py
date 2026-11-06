"""Focused proof for the opt in multimodal tenant contract."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from recall.embedding_registry import registered_profile
from recall.embeddings import resolve_embedder
from recall.multimodal import (
    DEFAULT_MAX_MEDIA_BYTES,
    MediaAccessDenied,
    MediaBudgetExceeded,
    MediaValidationError,
    MultimodalQuery,
    MultimodalTenantConfig,
    VoyageMultimodalEmbedder,
    build_media_ref,
    project_media_evidence,
)
from recall_mcp.settings import Settings
from recall_mcp.service import make_profile_embedder


PAYLOAD = b"\x89PNG\r\nsmall image fixture"
OBJECT_ROOT = "s3://recall-multimodal/media/"


def test_media_digest_and_sidecar_are_bounded_and_linked() -> None:
    """Red proof receipt: a mutation to ``media_digest`` broke the digest assertion at this node.

    Production symbol: ``recall.multimodal.media_digest``. The deliberate mutation appended a
    NUL before hashing, and the failing assertion was ``ref.content_digest == media_digest(PAYLOAD)``.
    Restored production behavior is the SHA256 over the original bytes.
    """
    ref = build_media_ref(
        PAYLOAD,
        media_type="image/png",
        object_uri="s3://recall-multimodal/media/example.png",
        object_root=OBJECT_ROOT,
        source_uri="s3://source-bucket/source.json",
        caption="a small image",
    )

    expected_digest = hashlib.sha256(PAYLOAD).hexdigest()
    assert ref.content_digest == expected_digest
    metadata = ref.metadata()
    assert metadata["media_id"] == f"re-call-multimodal:{expected_digest}"
    assert metadata["source_uri"] == "s3://source-bucket/source.json"
    assert "data_url" not in metadata
    assert "payload" not in metadata
    assert PAYLOAD not in repr(metadata).encode()


def test_media_admission_and_response_budgets_fail_closed() -> None:
    """Red proof receipt: mutating the ``len(payload) > max_bytes`` guard made this test admit it.

    Production symbols: ``build_media_ref`` and ``project_media_evidence``. The baseline mutation
    changed the admission comparison to ``>=`` for the exact limit case, and the response mutation
    removed the remaining byte check. Both were restored before the green run.
    """
    with pytest.raises(MediaBudgetExceeded, match="admission budget"):
        build_media_ref(
            PAYLOAD,
            media_type="image/png",
            object_uri="file:///srv/media/image.png",
            object_root="file:///srv/media/",
            max_bytes=3,
        )
    admitted = build_media_ref(
        PAYLOAD,
        media_type="image/png",
        object_uri="file:///srv/media/exact.png",
        object_root="file:///srv/media/",
        max_bytes=len(PAYLOAD),
    )
    assert admitted.byte_size == len(PAYLOAD)
    ref = build_media_ref(
        PAYLOAD,
        media_type="image/png",
        object_uri="file:///srv/media/image.png",
        object_root="file:///srv/media/",
    )
    with pytest.raises(MediaBudgetExceeded, match="response byte budget"):
        project_media_evidence(
            ref,
            requester_tenant="re-call-multimodal",
            principal="tester",
            max_response_bytes=ref.byte_size - 1,
        )


def test_media_reference_must_stay_under_configured_object_root() -> None:
    with pytest.raises(MediaValidationError, match="object root"):
        build_media_ref(
            PAYLOAD,
            media_type="image/png",
            object_uri="s3://other-bucket/media/image.png",
            object_root=OBJECT_ROOT,
        )
    with pytest.raises(MediaValidationError, match="OBJECT_ROOT"):
        build_media_ref(
            PAYLOAD,
            media_type="image/png",
            object_uri="s3://recall-multimodal/media/image.png",
        )
    with pytest.raises(MediaValidationError, match="object root"):
        build_media_ref(
            PAYLOAD,
            media_type="image/png",
            object_uri="s3://recall-multimodal/media/../private.png",
            object_root=OBJECT_ROOT,
        )


def test_multimodal_query_cannot_raise_the_default_byte_budget() -> None:
    with pytest.raises(MediaValidationError, match="max_bytes"):
        MultimodalQuery(
            image_bytes=PAYLOAD,
            media_type="image/png",
            max_bytes=DEFAULT_MAX_MEDIA_BYTES + 1,
        )


def test_media_access_is_tenant_scoped_and_original_is_reference_only() -> None:
    """Red proof receipt: removing the tenant equality guard made this cross tenant access green.

    Production symbol: ``project_media_evidence``. The failed assertion was the expected
    ``MediaAccessDenied`` for requester tenant ``re-call-code-gen``; the restored guard protects
    both the specialist boundary and the original object reference.
    """
    ref = build_media_ref(
        PAYLOAD,
        media_type="image/png",
        object_uri="s3://recall-multimodal/media/image.png",
        object_root=OBJECT_ROOT,
    )
    with pytest.raises(MediaAccessDenied, match="owning tenant"):
        project_media_evidence(
            ref,
            requester_tenant="re-call-code-gen",
            principal="tester",
        )
    evidence = project_media_evidence(
        ref,
        requester_tenant="re-call-multimodal",
        principal="tester",
    )
    assert evidence["original_media"] == {
        "object_uri": ref.object_uri,
        "content_digest": ref.content_digest,
    }


def test_multimodal_is_disabled_without_explicit_configuration() -> None:
    """Red proof receipt: changing the default enabled flag to true violated this assertion.

    Production symbols: ``Settings.from_env`` and ``MultimodalTenantConfig.from_env``. The
    baseline mutation enabled the tenant on an empty environment; restoration keeps all existing
    text defaults unchanged and requires an object root for explicit activation.
    """
    settings = Settings.from_env({})
    assert settings.multimodal.enabled is False
    assert settings.multimodal.tenant_id == "re-call-multimodal"
    assert settings.embedder == "fastembed"
    assert Settings.from_env({}).multimodal.physical_identity()["embedding_profile"] == (
        "voyage-multimodal-3.5-v1"
    )
    with pytest.raises(ValueError, match="disabled"):
        resolve_embedder("voyage-multimodal", {})
    with pytest.raises(ValueError, match="OBJECT_ROOT"):
        MultimodalTenantConfig.from_env({"RECALL_MULTIMODAL_ENABLED": "1"})
    with pytest.raises(ValueError, match="disabled"):
        make_profile_embedder("voyage-multimodal-3.5-v1", env={"VOYAGE_API_KEY": "test"})


def test_registered_profile_is_carried_by_multimodal_provider() -> None:
    """Red proof receipt: a mutation dropping ``identity`` from the adapter failed this profile.

    Production symbols: ``RegisteredProfile.identity`` and ``VoyageMultimodalEmbedder.profile``.
    The fake client avoids a provider call while proving the runtime carries the same generation
    binding and declared dimension as the registered profile.
    """
    entry = registered_profile("voyage-multimodal-3.5-v1")
    identity = entry.identity()

    class FakeClient:
        def multimodal_embed(self, **kwargs: object) -> object:
            inputs = kwargs["inputs"]
            return SimpleNamespace(embeddings=[[0.0] * 1024 for _ in inputs])

    embedder = VoyageMultimodalEmbedder(identity=identity, client=FakeClient())
    assert embedder.profile is identity
    assert embedder.dim == entry.dimension
    assert embedder.embed_query(MultimodalQuery(text="find the image")) == [0.0] * 1024


@pytest.mark.parametrize(
    "object_uri",
    [
        "s3://recall-multimodal/media/..%5Cprivate.png",
        "s3://recall-multimodal/media/a/..%5C..%5Cprivate.png",
        "s3://recall-multimodal/media/..\\private.png",
        "file:///C:/srv/media/..%5Csecret.png",
    ],
)
def test_a_backslash_traversal_is_refused(object_uri: str) -> None:
    """A backslash is a separator to a Windows consumer (CCA hunt SEC-001).

    Invariant: an admitted object_uri resolves under the object root on every platform.
    Failure mode: the containment checks split on '/' only, so a raw backslash or '..%5C' passed and
    resolved outside the root through url2pathname on Windows.

    Red proof, 2026-11-06, against the pre-fix ``_validate_object_root``: every parameter
    failed with ``Failed: DID NOT RAISE``.
    """
    root = "file:///C:/srv/media/" if object_uri.startswith("file:") else OBJECT_ROOT
    with pytest.raises(MediaValidationError, match="object root"):
        build_media_ref(PAYLOAD, media_type="image/png", object_uri=object_uri, object_root=root)


def test_a_uri_still_encoded_after_the_decode_passes_is_refused() -> None:
    """Running out of decode passes refuses rather than admits (CCA hunt DAT-001).

    Invariant: containment is judged on a path decoded to its fixpoint. Failure mode: after
    three passes a four times encoded '..' was still '%2e%2e', passed every check, and was
    admitted. Red proof, 2026-11-06, against the pre-fix ``decode_path``: the four times encoded
    URI failed with ``Failed: DID NOT RAISE``; the three times encoded control was refused before
    and after.
    """
    for encoded_dots in ("%25252e%25252e", "%2525252e%2525252e", "%252525252e%252525252e"):
        with pytest.raises(MediaValidationError, match="object root"):
            build_media_ref(
                PAYLOAD,
                media_type="image/png",
                object_uri=f"s3://recall-multimodal/media/{encoded_dots}/private.png",
                object_root=OBJECT_ROOT,
            )
    ref = build_media_ref(
        PAYLOAD,
        media_type="image/png",
        object_uri="s3://recall-multimodal/media/100%25.png",
        object_root=OBJECT_ROOT,
    )
    assert ref.object_uri == "s3://recall-multimodal/media/100%25.png"


def test_an_enabled_tenant_refuses_a_root_no_admission_can_use() -> None:
    """The root is checked at startup, not at the first upload (CCA hunt ENV-002).

    Invariant: an enabled tenant starts only with a root ``build_media_ref`` can contain under.
    Failure mode: a bare path or a bucket with no path started cleanly, then every admission was
    refused with a message blaming the object_uri. Red proof, 2026-11-06, against the pre-fix
    ``MultimodalTenantConfig.__post_init__``: both refused roots failed with
    ``Failed: DID NOT RAISE``.
    """
    for root, message in (("s3://bucket", "absolute path prefix"), ("/srv/media", "s3://")):
        with pytest.raises(MediaValidationError, match=message):
            MultimodalTenantConfig.from_env(
                {"RECALL_MULTIMODAL_ENABLED": "1", "RECALL_MULTIMODAL_OBJECT_ROOT": root}
            )
    config = MultimodalTenantConfig.from_env(
        {"RECALL_MULTIMODAL_ENABLED": "1", "RECALL_MULTIMODAL_OBJECT_ROOT": OBJECT_ROOT}
    )
    assert config.object_root == OBJECT_ROOT
