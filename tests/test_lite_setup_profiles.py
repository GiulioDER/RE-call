"""PR900 profile handoff and repeat model regressions.

Original449aa0a1 failed intended assertions for registered section context,
dotenv-only serving identity and a changed same-width model. Raw profile,
same-model repeat and first-model trusted serving are positive controls. Only
provider construction is replaced; registry, CLI, registration and MCP factory
remain real. Initial invalid probes were excluded from audit evidence.
"""

from __future__ import annotations

import importlib
import json

import pytest

from recall.embeddings import HashingEmbedder, embedding_profile, LEGACY_UNVERIFIED_DIGEST
from recall.embedding_registry import registered_profile
from recall.lite import LiteStore
from recall.lite.setup import run_lite_setup
from recall.trust_policy import TrustRefusal, TrustFailureCode
from tests.test_lite_calibration import ContentWordEmbedder, _write_memos, _strict
from tests.test_lite_setup_boundaries import isolated_lite_environment  # noqa: F401


class ProfileHashing(HashingEmbedder):
    def __init__(self, entry):
        super().__init__(dim=entry.dimension)
        self.profile = entry.identity(artifact_digest="0" * 64)

    @property
    def name(self):
        return self.profile.profile_id


def test_supported_section_profile_can_be_indexed(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    _write_memos(root / "notes", 6)
    monkeypatch.setenv("RECALL_EMBED_CACHE", str(tmp_path / "cache.sqlite"))
    raw = ProfileHashing(registered_profile("bge-small-symmetric-v1"))
    control = run_lite_setup(
        project_root=root,
        memory=root / "notes",
        embedder=raw,
        embedder_name="fastembed",
        store_path=root / "raw.db",
        register=False,
    )
    assert control.chunks == 6, "invalid registry raw-profile control"
    section = ProfileHashing(registered_profile("bge-small-context-section-v1"))
    assert section.dim == 384 and section.profile.context_version.startswith("context-section"), (
        "invalid registry section identity control"
    )
    try:
        result = run_lite_setup(
            project_root=root,
            memory=root / "notes",
            embedder=section,
            embedder_name="fastembed",
            store_path=root / "section.db",
            register=False,
        )
    except ValueError as exc:
        raise AssertionError("supported contextual setup profile was rejected") from exc
    assert result.chunks == 6


class OfflineFastEmbed(HashingEmbedder):
    built = []

    def __init__(self, *, identity=None, **kwargs):
        super().__init__(dim=384 if identity is None else identity.dimension)
        self.profile = identity or registered_profile("bge-small-symmetric-v1").identity(
            artifact_digest=LEGACY_UNVERIFIED_DIGEST
        )
        self.built.append(self)

    @property
    def name(self):
        return (
            "fastembed:BAAI/bge-small-en-v1.5"
            if self.profile is None
            else f"fastembed:{self.profile.model_name}"
        )


def test_dotenv_profile_reaches_registered_server(tmp_path, monkeypatch):
    import recall.embeddings as embeddings
    import recall.embedding_registry as registry
    import recall.wizard.wiring as wiring
    import recall_mcp.factories as factories
    from recall_mcp.factories import make_embedder

    for key in (
        "RECALL_EMBED_PROFILE",
        "RECALL_MODEL_CACHE",
        "RECALL_MODEL_SHA256",
        "RECALL_DSN",
        "RECALL_SERVING_DSN",
        "RECALL_MIGRATION_DSN",
        "RECALL_ENV",
        "RECALL_INDEX_MODE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(embeddings, "FastEmbedEmbedder", OfflineFastEmbed)
    monkeypatch.setattr(registry, "FastEmbedEmbedder", OfflineFastEmbed)
    monkeypatch.setattr(factories, "FastEmbedEmbedder", OfflineFastEmbed)
    raw = make_embedder("fastembed", env={})
    assert raw.dim == 384 and raw.profile.profile_id == "bge-small-symmetric-v1", (
        "invalid raw provider control"
    )
    OfflineFastEmbed.built.clear()
    root = tmp_path / "project"
    root.mkdir()
    _write_memos(root / "notes", 6)
    model = root / "model"
    model.mkdir()
    (model / "weights").write_bytes(b"offline provider fixture")
    config = root / "client.json"
    config.write_text(json.dumps({"projects": {}}))
    monkeypatch.setattr(wiring, "claude_config_path", lambda: config)
    (root / ".env").write_text(
        f"RECALL_EMBED_PROFILE=bge-base-symmetric-v1\nRECALL_MODEL_CACHE={model}\nRECALL_MODEL_SHA256={'0' * 64}\n"
    )
    monkeypatch.chdir(root)
    monkeypatch.setenv("RECALL_EMBED_CACHE", str(tmp_path / "cache.sqlite"))
    import recall.cli as cli

    cli = importlib.reload(cli)
    cli.main(["--embedder", "fastembed", "setup", "--lite", "--memory", str(root / "notes")])
    indexed = OfflineFastEmbed.built[0]
    assert indexed.dim == 768 and indexed.profile.profile_id == "bge-base-symmetric-v1", (
        "invalid actual dotenv/registry producer control"
    )
    doc = json.loads(config.read_text())
    envs = [
        entry["env"]
        for project in doc["projects"].values()
        for entry in project.get("mcpServers", {}).values()
    ]
    assert len(envs) == 1, "invalid persisted registration control"
    runtime = make_embedder(envs[0]["RECALL_EMBEDDER"], env=dict(envs[0]))
    assert (
        runtime.dim == indexed.dim
        and embedding_profile(runtime).fingerprint() == embedding_profile(indexed).fingerprint()
    ), "registered server lost the dotenv embedding profile identity"


class PermutedWords(ContentWordEmbedder):
    def __init__(self):
        super().__init__("model-b")
        self.calls = 0

    def embed(self, texts):
        self.calls += len(texts)
        return [v[1:] + v[:1] for v in super().embed(texts)]


def test_repeat_model_setup_preserves_calibration_identity(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    _write_memos(root / "notes", 48)
    config = root / "client.json"
    config.write_text(json.dumps({"projects": {}}))
    monkeypatch.setenv("RECALL_EMBED_CACHE", str(tmp_path / "cache.sqlite"))
    first = ContentWordEmbedder("model-a")
    kwargs = dict(project_root=root, memory=root / "notes", config_path=config)
    report = run_lite_setup(**kwargs, embedder=first, embedder_name=first.name)
    assert report.calibration == "certified", "invalid first-model calibration control"
    with LiteStore(report.store, dim=first.dim) as store:
        assert _strict(store, first).trust_state == "trusted", "invalid first-model serving control"
        original = store.latest_calibration_json()
    repeat = run_lite_setup(**kwargs, embedder=first, embedder_name=first.name)
    assert repeat.skipped == 48 and repeat.calibration == "certified", (
        "invalid same-model repeat control"
    )
    prior_config = config.read_bytes()
    second = PermutedWords()
    try:
        changed = run_lite_setup(**kwargs, embedder=second, embedder_name=second.name)
    except ValueError:
        assert second.calls == 0 and config.read_bytes() == prior_config, (
            "unsafe refusal after mutation"
        )
        with LiteStore(report.store, dim=first.dim) as store:
            assert store.latest_calibration_json() == original, "unsafe refusal changed calibration"
        return
    assert second.calls > 0, "invalid changed-model indexing control"
    with LiteStore(changed.store, dim=second.dim) as store:
        assert store.resolve_calibration().status.value == "certified", (
            "changed model not calibrated"
        )
        try:
            _strict(store, second)
        except TrustRefusal as refusal:
            assert refusal.code is TrustFailureCode.LINEAGE_MISMATCH, (
                "invalid lineage-failure control"
            )
            raise AssertionError(
                "repeat setup retained an incompatible certified model binding"
            ) from refusal


class HostedProfileHashing(HashingEmbedder):
    """Offline vectors with the registry's hosted identity, never a provider request."""

    def __init__(self, entry):
        super().__init__(dim=entry.dimension)
        self.profile = entry.identity()


@pytest.mark.parametrize("case", ["mismatch", "missing-artifact", "missing-credential"])
def test_index_only_model_ignores_unrelated_server_environment(tmp_path, monkeypatch, case):
    """ENV003 repair tree69b15ada: intended refusal assertion red before correction.

    An explicit model can index without registration. Ambient settings are only
    needed for reproducible followups. Each case retains an indexing control.
    """
    root = tmp_path / "project"
    root.mkdir()
    _write_memos(root / "notes", 6)
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    if case == "mismatch":
        model = ContentWordEmbedder()
        selected = "bge-base-symmetric-v1"
    elif case == "missing-artifact":
        entry = registered_profile("bge-small-symmetric-v1")
        model = ProfileHashing(entry)
        selected = entry.profile_id
    else:
        entry = registered_profile("voyage-3-v1")
        model = HostedProfileHashing(entry)
        selected = entry.profile_id
        monkeypatch.delenv(entry.api_key_env, raising=False)
    control = run_lite_setup(
        project_root=root,
        memory=root / "notes",
        embedder=model,
        embedder_name=model.name,
        store_path=root / "control.db",
        register=False,
    )
    assert control.chunks == 6, "invalid explicit model indexing control"
    monkeypatch.setenv("RECALL_EMBED_PROFILE", selected)
    try:
        result = run_lite_setup(
            project_root=root,
            memory=root / "notes",
            embedder=model,
            embedder_name=model.name,
            store_path=root / "target.db",
            register=False,
        )
    except ValueError as exc:
        raise AssertionError("index-only model rejected for unrelated serving environment") from exc
    assert result.chunks == 6 and not result.followup_env


def test_unreproducible_multimodal_registration_refuses_before_writes(tmp_path, monkeypatch):
    """ENV003 repair tree69b15ada admitted a server lacking its required opt-in.

    Lite setup explicitly refuses this unsupported registration before writes.
    An offline supplied-model indexing control proves the fixture is usable.
    """
    root = tmp_path / "project"
    root.mkdir()
    _write_memos(root / "notes", 6)
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    entry = registered_profile("voyage-multimodal-3.5-v1")
    model = HostedProfileHashing(entry)
    control = run_lite_setup(
        project_root=root,
        memory=root / "notes",
        embedder=model,
        embedder_name="voyage",
        store_path=root / "control.db",
        register=False,
    )
    assert control.chunks == 6, "invalid offline model indexing control"
    monkeypatch.setenv("RECALL_EMBED_PROFILE", entry.profile_id)
    monkeypatch.setenv("RECALL_MULTIMODAL_ENABLED", "1")
    monkeypatch.setenv(entry.api_key_env, "offline-provider-fixture")
    config = root / "client.json"
    config.write_text(json.dumps({"projects": {}}))
    original = config.read_bytes()
    store = root / "target.db"
    try:
        run_lite_setup(
            project_root=root,
            memory=root / "notes",
            embedder=model,
            embedder_name="voyage",
            store_path=store,
            config_path=config,
        )
    except ValueError as exc:
        assert "multimodal" in str(exc), "unrelated refusal is not proof"
        assert not store.exists() and config.read_bytes() == original
        return
    raise AssertionError("unreproducible multimodal registration accepted before writes")


@pytest.mark.parametrize(
    "alias, profile_id, refusal",
    [
        ("voyage-multimodal", "voyage-multimodal-3.5-v1", "multimodal"),
        ("voyage-context", "voyage-context-4-v1", "RECALL_EMBED_PROFILE"),
    ],
)
def test_direct_hosted_alias_refuses_unreproducible_registration(
    tmp_path, monkeypatch, alias, profile_id, refusal
):
    """ENV003 repair tree680282af: direct aliases bypassed the empty-profile guard.

    Resolve the real CLI alias and registry identity with only the provider
    constructor replaced. Index-only CLI is a control. Registered CLI must
    refuse before store/config writes when those settings are unsupported.
    """
    import os
    import recall.embeddings as embeddings
    import recall.embedding_registry as registry
    import recall.multimodal as multimodal
    import recall.wizard.wiring as wiring

    class OfflineHosted(HashingEmbedder):
        def __init__(self, *, identity, **kwargs):
            super().__init__(dim=identity.dimension)
            self.profile = identity

    monkeypatch.setattr(multimodal, "VoyageMultimodalEmbedder", OfflineHosted)
    monkeypatch.setattr(registry, "VoyageContextualizedEmbedder", OfflineHosted)
    monkeypatch.setenv("RECALL_MULTIMODAL_ENABLED", "1")
    monkeypatch.setenv("VOYAGE_API_KEY", "offline-provider-fixture")
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    root = tmp_path / "project"
    root.mkdir()
    _write_memos(root / "notes", 6)
    config = root / "client.json"
    config.write_text(json.dumps({"projects": {}}))
    monkeypatch.setattr(wiring, "claude_config_path", lambda: config)
    monkeypatch.chdir(root)
    model = embeddings.resolve_embedder(alias)
    assert model.profile.profile_id == profile_id and not os.environ.get("RECALL_EMBED_PROFILE"), (
        "invalid actual alias producer control"
    )
    import recall.cli as cli

    cli = importlib.reload(cli)
    cli.main(
        [
            "--embedder",
            alias,
            "setup",
            "--lite",
            "--memory",
            str(root / "notes"),
            "--store",
            str(root / "control.db"),
            "--no-register",
        ]
    )
    assert (root / "control.db").exists(), "invalid index-only CLI control"
    before = config.read_bytes()
    try:
        cli.main(["--embedder", alias, "setup", "--lite", "--memory", str(root / "notes")])
    except SystemExit as exc:
        assert refusal in str(exc), "unrelated CLI refusal is not proof"
        assert not (root / ".recall").exists() and config.read_bytes() == before
        return
    raise AssertionError("direct hosted alias registered without reproducible server settings")


def test_cli_omits_guide_advice_when_model_settings_are_unavailable(tmp_path, monkeypatch, capsys):
    """ENV002/ENV003: no-register may index without printable model settings.

    Repair tree680282af printed advice referring to commands it did not render.
    CLI indexes with an explicit offline model; only construction is replaced.
    """
    from recall.cli_commands import setup_wizard

    root = tmp_path / "project"
    root.mkdir()
    _write_memos(root / "notes", 6)
    monkeypatch.chdir(root)
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    monkeypatch.setenv("RECALL_EMBED_PROFILE", "bge-base-symmetric-v1")
    monkeypatch.setattr(
        setup_wizard, "_make_embedder", lambda *args, **kwargs: ContentWordEmbedder()
    )
    import recall.cli as cli

    cli = importlib.reload(cli)
    cli.main(
        [
            "--embedder",
            "hashing",
            "setup",
            "--lite",
            "--memory",
            str(root / "notes"),
            "--no-register",
        ]
    )
    printed = capsys.readouterr().out
    assert (root / ".recall" / "memory.db").exists() and "6 chunk(s) from 6 file(s)" in printed, (
        "invalid index-only CLI control"
    )
    assert not any(line.strip().startswith("env ") for line in printed.splitlines()), (
        "invalid missing-guide control"
    )
    assert "Use the configured commands above" not in printed, (
        "CLI points to nonexistent followup commands"
    )


@pytest.mark.parametrize("alias", ["openai", "openrouter"])
def test_legacy_hosted_dotenv_alias_refuses_before_writes(tmp_path, monkeypatch, alias):
    """ENV003 treeaecd7885: actual legacy resolver loses dotenv key and width.

    Only the provider constructor is offline. Real dotenv/CLI resolution,
    registration and MCP factory are retained. Explicit factory configuration
    and index-only CLI are controls; automatic registration must refuse safely.
    """
    import recall.embeddings as embeddings
    import recall.wizard.wiring as wiring
    from recall_mcp.factories import make_embedder

    class OfflineCompat(HashingEmbedder):
        built = []

        def __init__(self, *, api_key=None, dimensions=None, identity=None, **kwargs):
            if not api_key:
                raise ValueError("offline fixture requires configured provider credential")
            super().__init__(dim=1536 if dimensions is None else dimensions)
            self.profile = identity
            self.credential = api_key
            self.built.append(self)

    monkeypatch.setattr(embeddings, "OpenAICompatEmbedder", OfflineCompat)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("RECALL_EMBED_DIMENSIONS", raising=False)
    root = tmp_path / "project"
    root.mkdir()
    _write_memos(root / "notes", 6)
    config = root / "client.json"
    config.write_text(json.dumps({"projects": {}}))
    monkeypatch.setattr(wiring, "claude_config_path", lambda: config)
    (root / ".env").write_text(
        "OPENROUTER_API_KEY=offline-provider-fixture\nRECALL_EMBED_DIMENSIONS=512\n"
    )
    monkeypatch.chdir(root)
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    control = make_embedder(
        alias,
        env={"OPENROUTER_API_KEY": "offline-provider-fixture", "RECALL_EMBED_DIMENSIONS": "512"},
    )
    assert control.dim == 512 and control.profile is None, "invalid explicit factory control"
    import recall.cli as cli

    cli = importlib.reload(cli)
    cli.main(
        [
            "--embedder",
            alias,
            "setup",
            "--lite",
            "--memory",
            str(root / "notes"),
            "--store",
            str(root / "control.db"),
            "--no-register",
        ]
    )
    indexed = OfflineCompat.built[-1]
    assert indexed.dim == 512 and indexed.credential == "offline-provider-fixture", (
        "invalid real dotenv producer control"
    )
    assert (root / "control.db").exists(), "invalid index-only CLI control"
    before = config.read_bytes()
    try:
        cli.main(["--embedder", alias, "setup", "--lite", "--memory", str(root / "notes")])
    except SystemExit as exc:
        assert "RECALL_EMBED_PROFILE" in str(exc), "unrelated refusal is not proof"
        assert not (root / ".recall").exists() and config.read_bytes() == before
        return
    doc = json.loads(config.read_text())
    envs = [
        entry["env"]
        for project in doc["projects"].values()
        for entry in project.get("mcpServers", {}).values()
    ]
    assert len(envs) == 1, "invalid registration control"
    try:
        runtime = make_embedder(alias, env=dict(envs[0]))
    except ValueError:
        runtime = None
    assert runtime is None or runtime.dim != indexed.dim, "consumer gap was not reproduced"
    raise AssertionError("legacy hosted alias registered without reproducible dotenv settings")
