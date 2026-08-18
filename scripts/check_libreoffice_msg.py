#!/usr/bin/env python
"""Re-measure whether LibreOffice can read an Outlook ``.msg``.

``recall/extraction.py`` deliberately gives ``.msg`` no LibreOffice fallback, because LibreOffice
ships no MAPI import filter. That is a claim about another program's capabilities, so it can rot
when LibreOffice gains a filter. This script re-measures it rather than leaving the claim to be
believed on the strength of being specific.

Measured 2026-08-18, LibreOffice 25.8 on Windows 11: exit 1, "source file could not be loaded",
no output file. Re-run with::

    python scripts/check_libreoffice_msg.py

It needs a genuine ``.msg`` and downloads one from the python-oxmsg test corpus unless you pass a
path to your own. It converts nothing in the repository and writes only to a temporary directory.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recall.extraction import LIBREOFFICE_EXTENSIONS, _libreoffice_executable  # noqa: E402

FIXTURE_URL = (
    "https://raw.githubusercontent.com/scanny/python-oxmsg/main/tests/test_files/message.msg"
)
CFBF_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def _fixture(directory: Path, supplied: str | None) -> Path:
    target = directory / "sample.msg"
    if supplied:
        shutil.copy(supplied, target)
    else:
        print(f"fetching a genuine .msg from {FIXTURE_URL}")
        with urllib.request.urlopen(FIXTURE_URL, timeout=60) as response:
            target.write_bytes(response.read())
    magic = target.read_bytes()[:8]
    if magic != CFBF_MAGIC:
        raise SystemExit(
            f"{target} is not a compound file (magic {magic.hex()}); the measurement would test "
            "the fixture rather than LibreOffice"
        )
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("msg", nargs="?", help="a genuine .msg file; downloaded if omitted")
    arguments = parser.parse_args()

    executable = _libreoffice_executable()
    if executable is None:
        print("LibreOffice is not installed; nothing to measure")
        return 2

    with tempfile.TemporaryDirectory(prefix="recall-msg-check-") as name:
        directory = Path(name)
        profile = directory / "profile"
        profile.mkdir()
        source = _fixture(directory, arguments.msg)
        completed = subprocess.run(
            [
                executable,
                f"-env:UserInstallation={profile.as_uri()}",
                "--headless",
                "--convert-to",
                "txt:Text",
                "--outdir",
                str(directory),
                str(source),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        produced = source.with_suffix(".txt")
        readable = completed.returncode == 0 and produced.exists()
        print(f"executable   {executable}")
        print(f"exit code    {completed.returncode}")
        print(f"stdout       {completed.stdout.strip()!r}")
        print(f"stderr       {completed.stderr.strip()!r}")
        print(f"output file  {'produced' if produced.exists() else 'none'}")

    if readable:
        print(
            "\nCHANGED: LibreOffice now reads .msg. Reconsider giving MSG a LibreOffice fallback "
            "when python-oxmsg is absent, and update the tests in "
            "tests/test_legacy_document_extraction.py."
        )
        return 1

    print("\nUNCHANGED: LibreOffice cannot read .msg, so the oxmsg only dispatch is still correct.")
    if ".msg" in LIBREOFFICE_EXTENSIONS:
        print("but recall.extraction.LIBREOFFICE_EXTENSIONS lists .msg, which contradicts that")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
