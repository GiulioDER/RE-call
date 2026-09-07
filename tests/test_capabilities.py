from __future__ import annotations

from recall.capabilities import diagnose_exception, probe, requirements_for


def test_requirement_planning_does_not_construct_optional_dependencies() -> None:
    requirements = requirements_for("fastembed", documents=True)
    assert [item.target for item in requirements[:2]] == ["fastembed", "onnxruntime"]
    assert all(item.package in {"fastembed", "onnxruntime", "documents", None} for item in requirements)


def test_missing_python_dependency_is_actionable() -> None:
    requirement = requirements_for("fastembed")[0]
    diagnostic = probe(requirement)
    assert diagnostic.code in {"capability.available", "dependency.missing"}
    assert diagnostic.platform


def test_native_loader_failure_has_a_stable_code() -> None:
    diagnostic = diagnose_exception(
        "onnxruntime", "embedder build", OSError("DLL load failed"),
        target="onnxruntime", remediation="install the fastembed extra",
    )
    assert diagnostic.code == "dependency.native_load_failed"
    assert "install the fastembed extra" in diagnostic.render()


def test_missing_executable_has_a_stable_code() -> None:
    diagnostic = diagnose_exception("libreoffice", "document extraction", FileNotFoundError("libreoffice"))
    assert diagnostic.code == "dependency.executable_missing"
