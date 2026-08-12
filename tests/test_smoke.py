import json
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


def test_server_json_versions_match_pyproject():
    """The MCP registry manifest writes the version THREE times and nothing checked any of them.

    One of the three is a hard `==` pin inside the `uvx --from` argument. A stale pin there does
    not fail loudly the way a bad classifier does: the registry entry keeps resolving and keeps
    installing, it just installs a version nobody asked for. CITATION.cff already earned its own
    test by sitting at 0.5.1 through a whole release; this file writes the number three times as
    often.
    """
    server = json.loads(
        (Path(__file__).parent.parent / "server.json").read_text(encoding="utf-8")
    )
    declared = _declared_version()

    assert server["version"] == declared
    package = server["packages"][0]
    assert package["version"] == declared

    pins = [
        arg["value"]
        for arg in package["runtimeArguments"]
        if arg.get("name") == "--from"
    ]
    assert pins, "no --from argument in server.json to pin the version"
    assert all(pin.endswith(f"=={declared}") for pin in pins), pins
