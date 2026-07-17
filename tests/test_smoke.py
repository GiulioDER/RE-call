import re
from pathlib import Path

import recall


def test_version_matches_pyproject():
    # single-source check: recall.__version__ and pyproject.toml must never drift again
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    m = re.search(r'^version = "([^"]+)"', pyproject.read_text(encoding="utf-8"), re.M)
    assert m, "no version in pyproject.toml"
    assert recall.__version__ == m.group(1)
