"""The sparse extra must install a usable Transformers PyTorch backend."""

from pathlib import Path
import tomllib

from packaging.requirements import Requirement


def test_sparse_requirements_reject_unsupported_torch_versions() -> None:
    """DEP-001: Transformers 5.16.1 requires torch>=2.5 for its PyTorch backend.

    Red proof target: pre-fix sparse metadata (`torch>=2.0,<3`) admits 2.0 through 2.4;
    the membership assertion below must reject those versions while admitting 2.5.
    Upstream: transformers v5.16.1, src/transformers/utils/import_utils.py.
    """
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = {
        requirement.name: requirement.specifier
        for value in metadata["project"]["optional-dependencies"]["sparse"]
        for requirement in [Requirement(value)]
    }
    for version in ("2.0.1", "2.1.2", "2.2.2", "2.3.1", "2.4.1"):
        assert version not in requirements["torch"], f"unsupported torch {version} is admitted"
    # The floor is 2.13 since 2026-11-07 (advisory fixes), so the lowest admitted version moved.
    assert "2.13.0" in requirements["torch"]
    assert "5.16.1" in requirements["transformers"]
