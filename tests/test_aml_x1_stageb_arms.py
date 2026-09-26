"""X-1 Stage B's extra search arms (``--also-search ARM=PORT``) are parsed strictly.

Invariant: each arm names one service port; a malformed pair or a repeated arm name stops the run
before any Add, because two arms under one name would overwrite each other's items in the record.

Red proof, 2026-09-26, against ``extra_arms`` in ``scripts/aml_x1_stageb.py`` with this file
unchanged: dropping ``name in arms`` from the refusal made ``test_a_repeated_arm_is_refused`` fail
on ``pytest.raises(SystemExit)`` (the second pair silently replaced the first).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

SPEC = importlib.util.spec_from_file_location(
    "aml_x1_stageb", Path(__file__).parents[1] / "scripts" / "aml_x1_stageb.py"
)
assert SPEC and SPEC.loader
stageb = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stageb  # @dataclass resolves its module through sys.modules
SPEC.loader.exec_module(stageb)


def test_pairs_become_a_port_per_arm() -> None:
    assert stageb.extra_arms(["D=18035", "Dt=18036"]) == {"D": 18035, "Dt": 18036}
    assert stageb.extra_arms([]) == {}


@pytest.mark.parametrize("value", ["D", "D=", "=18035", "D=port"])
def test_a_malformed_pair_is_refused(value: str) -> None:
    with pytest.raises(SystemExit):
        stageb.extra_arms([value])


def test_a_repeated_arm_is_refused() -> None:
    with pytest.raises(SystemExit):
        stageb.extra_arms(["D=18035", "D=18036"])
