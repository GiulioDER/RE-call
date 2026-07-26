import re
from pathlib import Path

import recall


def _declared_version() -> str:
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    m = re.search(r'^version = "([^"]+)"', pyproject.read_text(encoding="utf-8"), re.M)
    assert m, "no version in pyproject.toml"
    return m.group(1)


def test_version_matches_pyproject():
    # single-source check: recall.__version__ and pyproject.toml must never drift again
    assert recall.__version__ == _declared_version()


def test_citation_version_matches_pyproject():
    # CITATION.cff is the third place the version is written, and the only one nothing checked —
    # so it silently sat at 0.5.1 across the whole 0.5.2 release. A stale citation misattributes
    # which version a result was produced with, which is the one job the file has.
    citation = Path(__file__).parent.parent / "CITATION.cff"
    m = re.search(r"^version:\s*(\S+)", citation.read_text(encoding="utf-8"), re.M)
    assert m, "no version in CITATION.cff"
    assert m.group(1) == _declared_version()
