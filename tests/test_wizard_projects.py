"""Adding a project to an install that already exists.

The hazard this module is built around: `compose_document` builds the whole document from the
tenants it is handed and `write_compose` replaces the file, so the obvious implementation of "add a
project" — run the generator with the new project's tenants — does not add a project. It DELETES
every existing one, orphaning their corpora in the database with nothing erroring.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall.wizard.projects import ProjectRefusal, add_project, compose_path_for, stack_embedder
from recall.wizard.stack import StackSpec, compose_document, existing_tenants, write_compose

INSTALLED = ("default-docs", "default-code", "default-memory")


def _install(root: Path, trust: str = "production") -> Path:
    env = {
        tenant: {
            "RECALL_DSN": "postgresql://recall:recall@db:5432/recall",
            "RECALL_EMBEDDER": "fastembed",
            "RECALL_TENANT": tenant,
            "RECALL_TRUST_MODE": trust,
        }
        for tenant in INSTALLED
    }
    path = compose_path_for(root)
    write_compose(
        path, compose_document(StackSpec(data_root=root, port=5487, tenants=INSTALLED, env=env))
    )
    return path


def test_the_generated_stack_can_build_its_own_image(tmp_path: Path) -> None:
    """`image:` alone makes Compose PULL a tag published to no registry.

    The only other `build:` for it lives in `docker-compose.desktop.yml`, whose context is the
    source checkout, which a user who installed recall from PyPI does not have. So the wizard
    writes its own Dockerfile beside the compose file, and every tenant service must reference it:
    a compose document naming a Dockerfile that is not there is broken by construction, which is
    why `write_compose` writes both.
    """
    from recall.wizard.stack import DOCKERFILE_NAME

    root = (tmp_path / "install").resolve()
    root.mkdir()
    path = _install(root)

    assert (root / DOCKERFILE_NAME).exists(), "the build stanza points at a file that must exist"
    services = json.loads(path.read_text(encoding="utf-8"))["services"]
    for name, service in services.items():
        if name == "db":
            continue
        assert service["build"] == {"context": ".", "dockerfile": DOCKERFILE_NAME}, name
        assert service["image"], f"{name} must still tag the build rather than build anonymously"


def test_the_generated_dockerfile_fails_the_build_on_a_missing_extra() -> None:
    """pip only WARNS about an extra a release does not provide, so the check must be explicit.

    Measured against the real 0.9.5 wheel in a real container:

        WARNING: recall-rag 0.9.5 does not provide the extra 'documents'
        --- pip exit: 0 ---
        ModuleNotFoundError: No module named 'pypdf'

    So a pin whose version predates an extra produces an image that builds clean and then silently
    extracts nothing from every .docx, .pdf, .xlsx and .pptx the user feeds it. These imports move
    that failure to build time. The test pins their presence, not their contents, because losing
    the assertion is what would make the defect silent again.
    """
    from recall.wizard.stack import dockerfile_text

    text = dockerfile_text(version="9.9.9")

    assert '"recall-rag[mcp,fastembed,documents]==9.9.9"' in text, "the pin must be explicit"
    assert "import pypdf, docx, openpyxl, pptx, bs4" in text, (
        "without a post-install import the missing document extra is only a pip WARNING"
    )
    assert "import fastembed" in text
    assert "import recall_mcp.server" in text
    assert "COPY" not in text, (
        "the wizard runs from an installed wheel and has no source tree to copy; copying is what "
        "made the shipped Dockerfile unusable for a generated stack"
    )


def test_regenerating_the_stack_would_delete_the_existing_projects(tmp_path: Path) -> None:
    """The hazard, pinned. This is what `add_project` must not do.

    Less a test of production code than a standing demonstration of why the additive path exists:
    if someone later "simplifies" `add_project` into a regeneration, this test still passes while
    the one below fails, and the pair says exactly what went wrong.
    """
    root = (tmp_path / "install").resolve()
    root.mkdir()
    _install(root)
    assert existing_tenants(compose_path_for(root)) == tuple(sorted(INSTALLED))

    new = ("acme-docs", "acme-code", "acme-memory")
    write_compose(
        compose_path_for(root),
        compose_document(
            StackSpec(
                data_root=root, port=5487, tenants=new, env={t: {"RECALL_DSN": "x"} for t in new}
            )
        ),
    )

    survived = [t for t in existing_tenants(compose_path_for(root)) if t.startswith("default-")]
    assert survived == [], "if this ever passes with survivors, the generator became additive"


def test_add_project_preserves_every_existing_project_and_its_trust(tmp_path: Path) -> None:
    """Adding must be strictly additive, including the trust posture of what is already there.

    A union-then-regenerate would rewrite existing services' environment, and that environment
    carries `RECALL_TRUST_MODE`, which the wiring stage set from what actually CERTIFIED. Resetting
    a certified corpus to development trust is a silent weakening of the safety core, so the
    existing services are asserted byte for byte, not merely present.
    """
    root = (tmp_path / "install").resolve()
    root.mkdir()
    path = _install(root)
    before = json.loads(path.read_text(encoding="utf-8"))["services"]

    added = add_project(root, "acme")

    after = json.loads(path.read_text(encoding="utf-8"))["services"]
    assert added.tenants == ("acme-code", "acme-docs", "acme-memory")
    assert set(existing_tenants(path)) == set(INSTALLED) | set(added.tenants)
    for name, service in before.items():
        assert after[name] == service, f"{name} was modified by an add"
    assert after["recall-acme-docs"]["environment"]["RECALL_TRUST_MODE"] == "development", (
        "a project with no corpus, no generation and no calibration is not production-trusted"
    )


def test_a_new_project_inherits_the_stacks_embedder(tmp_path: Path) -> None:
    """The embedder is welded to the table's vector dimension, so it is not the caller's choice.

    A project added with a different one would build a corpus this database cannot hold, and the
    failure would arrive at ingest rather than here.
    """
    root = (tmp_path / "install").resolve()
    root.mkdir()
    path = _install(root)
    assert stack_embedder(path) == "fastembed"

    add_project(root, "acme")

    services = json.loads(path.read_text(encoding="utf-8"))["services"]
    assert services["recall-acme-docs"]["environment"]["RECALL_EMBEDDER"] == "fastembed"


def test_adding_the_same_project_twice_is_a_no_op(tmp_path: Path) -> None:
    """A repeat must not reset the project's trust posture back to development."""
    root = (tmp_path / "install").resolve()
    root.mkdir()
    path = _install(root)
    add_project(root, "acme")
    once = json.loads(path.read_text(encoding="utf-8"))

    again = add_project(root, "acme")

    assert again.tenants == ()
    assert not again.created_anything, "the caller must not report this as created"
    assert len(again.already_present) == 3
    assert json.loads(path.read_text(encoding="utf-8")) == once


def test_add_tenant_services_never_overwrites_a_service_that_exists(tmp_path: Path) -> None:
    """The guard inside the writer, exercised directly.

    `add_project` filters an already-present tenant out before calling this, so the skip is
    unreachable through that path and a mutation removing it stayed GREEN across the whole suite.
    That is exactly the shape of an untested guard: present, believed, and load-bearing for any
    future caller. Overwriting here would rebuild the service from the passed environment and reset
    a certified corpus's `RECALL_TRUST_MODE`, which is the thing this module exists to prevent.
    """
    from recall.wizard.stack import add_tenant_services

    root = (tmp_path / "install").resolve()
    root.mkdir()
    path = _install(root)
    before = json.loads(path.read_text(encoding="utf-8"))["services"]["recall-default-docs"]
    assert before["environment"]["RECALL_TRUST_MODE"] == "production"

    added = add_tenant_services(
        path,
        {"default-docs": {"RECALL_EMBEDDER": "fastembed", "RECALL_TRUST_MODE": "development"}},
    )

    after = json.loads(path.read_text(encoding="utf-8"))["services"]["recall-default-docs"]
    assert added == (), "a tenant that already has a service was reported as added"
    assert after == before, "an existing service was rebuilt from the caller's environment"


@pytest.mark.parametrize(
    ("name", "because"),
    [
        ("default", "the installer's own project"),
        ("my app!", "a character Docker will not accept in a service name"),
        ("notes-docs", "a name ending in a corpus kind"),
    ],
)
def test_a_name_the_wizard_would_refuse_is_refused_here(
    tmp_path: Path, name: str, because: str
) -> None:
    """Refused at the point of typing, with the wizard's own reason.

    The UI used to accept these and then tell the user to run the wizard, which would have refused
    them — so the guidance itself was a dead end.
    """
    root = (tmp_path / "install").resolve()
    root.mkdir()
    path = _install(root)
    before = path.read_text(encoding="utf-8")

    with pytest.raises(ProjectRefusal):
        add_project(root, name)

    assert path.read_text(encoding="utf-8") == before, f"a refusal ({because}) must not write"


def test_adding_to_a_location_with_no_stack_refuses(tmp_path: Path) -> None:
    """There is nothing to add to, and writing a fresh stack here would be the wrong answer."""
    with pytest.raises(ProjectRefusal, match="no recall stack"):
        add_project((tmp_path / "empty").resolve(), "acme")


def test_an_unreadable_stack_is_never_overwritten(tmp_path: Path) -> None:
    """The file IS the stack. Overwriting one that cannot be parsed strands every corpus it held.

    Deliberately the opposite of `existing_tenants` and `existing_port`, which return empty for an
    unreadable file: those are consulted before writing a fresh document, where nothing is at risk.

    ⚠️ Asserted against `add_tenant_services` DIRECTLY, and that is the whole point of this test.
    Written first against `add_project`, it passed — and passed for the wrong reason: `add_project`
    consults `existing_port` first, which returns None for an unparseable file, so it refused with
    "no recall stack" and never reached this guard. A mutation that made the guard swallow the
    error stayed green. Worse, if `existing_port` DOES succeed the document is parseable by
    construction, so this branch is unreachable from `add_project` at all and exists purely for
    direct callers — which is precisely the kind of guard that rots untested.
    """
    from recall.wizard.stack import add_tenant_services

    root = (tmp_path / "install").resolve()
    root.mkdir()
    path = compose_path_for(root)
    corrupt = '{"services": {"db": {"ports": ["5487:5432"]}}, truncated'
    path.write_text(corrupt, encoding="utf-8")

    with pytest.raises(ValueError, match="not safe to write a new one over it"):
        add_tenant_services(path, {"acme-docs": {"RECALL_EMBEDDER": "fastembed"}})

    assert path.read_text(encoding="utf-8") == corrupt, "the unreadable file must be left alone"


def test_add_project_refuses_an_unparseable_stack_before_it_gets_that_far(tmp_path: Path) -> None:
    """And the path a user actually takes refuses too, just with a different reason."""
    root = (tmp_path / "install").resolve()
    root.mkdir()
    compose_path_for(root).write_text('{"services": {"db": {}}, truncated', encoding="utf-8")

    with pytest.raises(ProjectRefusal, match="no recall stack"):
        add_project(root, "acme")
