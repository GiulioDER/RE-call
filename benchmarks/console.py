"""Make a benchmark CLI's output survive a Windows console.

These modules document themselves with `⚠️`, `⛔` and `🔑`, and argparse prints the module
docstring for `--help`. Neither character exists in cp1252, so on a Windows console **`--help`
itself raises `UnicodeEncodeError`** before any work is attempted, and a run that printed a
warning line died at the warning rather than at the thing being warned about.

⚠️ This is not a new hazard and not a new fix. `recall/cli.py` has carried the same
`reconfigure` call for the shipped CLI for some time; the benchmark entry points never got it,
so `python -m benchmarks.analyse_triage --help` crashed on Windows until this landed.

**Wired into every `benchmarks` entry point that passes a docstring to argparse**, verified by
running `--help` for each on a cp1252 console: the five triage modules, plus
`check_generation_parity`, `drop_parity_tables`, `freeze_supersession_evidence` and
`probe_supersession_annotation`. `store_latency_share` needs nothing: it passes no `description`.
A new CLI in this package should import this too.
"""

from __future__ import annotations

import sys


def use_utf8_output() -> None:
    """Reconfigure stdout and stderr to UTF-8, keeping a non-strict error handler.

    ``errors=`` as well as ``encoding=``, because reconfiguring the encoding RESETS errors to
    strict: the reason is recorded at length in `recall/cli.py`, where dropping the handler once
    made every `print` of a non-UTF-8 filename raise and threw away a completed run at its report
    step. Showing a mangled character beats showing nothing.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
