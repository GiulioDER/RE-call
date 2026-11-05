"""The sdist is built from an allowlist, so the research tree cannot ride along to PyPI.

Invariant: `[tool.hatch.build.targets.sdist]` names what a build needs with `only-include`, and
nothing it names is a research, results or operator directory. Failure mode caught: the section
is deleted or loosened, and hatchling falls back to packing every tracked file. That is not
hypothetical: sdists 0.11.0 to 0.14.0 carried `docs/results` and `results`, and the 0.14.0 sdist
was 91 MB against a 3.5 MB wheel.

Red proof, recorded 2026-09-26, one mutation of `pyproject.toml` per test, each restored after:

- whole `[tool.hatch.build.targets.sdist]` section removed: `test_the_sdist_is_an_allowlist`
  fails on its assertion ("has no ... only-include list"). The third test errors with
  `KeyError: 'sdist'` under the same mutation, which is not counted as its proof.
- `"docs"` appended to `only-include`: `test_the_allowlist_names_no_research_or_operator_tree`
  fails with `['docs'] == []`.
- `"recall_agent"` removed from `only-include`: `test_every_wheel_package_is_in_the_sdist` fails
  with `['recall_agent'] == []`.

Restored, all three pass. The byte-level check that a wheel built from this sdist equals a wheel built
from the tree was run by hand the same day (324 files, no difference) and is not repeated here,
because building twice costs a minute of every suite run.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

#: Top-level trees that hold research artefacts or operator tooling, never package code.
NOT_FOR_PYPI = ("docs", "results", "benchmarks", "scripts", "launch", "infra", "tests", "site")


def _sdist_config() -> dict[str, Any]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    targets: dict[str, dict[str, Any]] = data["tool"]["hatch"]["build"]["targets"]
    return targets.get("sdist", {})


def test_the_sdist_is_an_allowlist() -> None:
    config = _sdist_config()
    assert config.get("only-include"), (
        "pyproject.toml has no [tool.hatch.build.targets.sdist] only-include list, so the sdist "
        "would pack every tracked file, research results included"
    )


def test_the_allowlist_names_no_research_or_operator_tree() -> None:
    included = _sdist_config().get("only-include", [])
    leaked = sorted(
        entry for entry in included if entry.strip("/").split("/", 1)[0] in NOT_FOR_PYPI
    )
    assert leaked == [], f"sdist allowlist includes non-package trees: {leaked}"


def test_every_wheel_package_is_in_the_sdist() -> None:
    """A package the wheel ships but the sdist omits would build an empty wheel from the sdist."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    targets = data["tool"]["hatch"]["build"]["targets"]
    missing = sorted(set(targets["wheel"]["packages"]) - set(targets["sdist"]["only-include"]))
    assert missing == [], f"wheel packages missing from the sdist allowlist: {missing}"


def test_the_aml_benchmark_service_is_not_distributed() -> None:
    """`recall_aml` stays in the repository but leaves both distributions in 0.15.0.

    Invariant: no wheel package, no sdist entry, no optional dependency set and no console script
    names the AML benchmark service. Failure mode caught: any one of them comes back, which puts
    a benchmark endpoint and its dependency set into every install (owner decision 2026-11-04).

    Red proof, recorded 2026-11-05: run against master's `pyproject.toml` at 21261c353 (which
    still lists `recall_aml` in the wheel `packages` and the sdist `only-include`, declares the
    `hosted` extra and the `recall-hosted` script), this test fails on its assertion, listing all
    four. Restored to this branch's `pyproject.toml`, it passes.
    """
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    targets = data["tool"]["hatch"]["build"]["targets"]
    found = [
        *(f"wheel package {name}" for name in targets["wheel"]["packages"] if name == "recall_aml"),
        *(
            f"sdist entry {entry}"
            for entry in targets["sdist"]["only-include"]
            if entry.strip("/").split("/", 1)[0] == "recall_aml"
        ),
        *(f"extra {name}" for name in data["project"]["optional-dependencies"] if name == "hosted"),
        *(
            f"script {name}"
            for name, target in data["project"]["scripts"].items()
            if name == "recall-hosted" or target.startswith("recall_aml")
        ),
    ]
    assert found == [], f"the AML benchmark service is distributed again: {found}"
