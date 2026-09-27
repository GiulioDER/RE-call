"""The MemEye Stage 1 harness runs MM-1's four arms by default and MM-4's by an arm config.

Invariants: without a config the harness Searches exactly MM-1's arms and checks their scopes; a
config must name the same arms in its order as in its ports, with the ingest arm among them; a run
is refused unless every arm serves one commit and reports each expected ``/version`` field,
including a nested one such as ``image_text.leg``.

Red proof, 2026-09-27, each against ``scripts/aml_mm_scope_stage1.py`` with this file unchanged
(``PYTHONDONTWRITEBYTECODE=1``):
- ``check_versions`` skipping the expected-field comparison:
  ``test_a_wrong_setting_on_any_arm_refuses_the_run`` failed on ``DID NOT RAISE``.
- ``_field`` reading only the top level (``version.get(dotted)``): it then cannot see ``image_text.build``
  either, so ``test_a_nested_field_is_read_through_its_parent`` failed on the ``match`` of its refusal.
- ``load_arms`` without the ingest-arm check:
  ``test_an_arm_config_must_name_its_ingest_arm`` failed on ``DID NOT RAISE``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

SPEC = importlib.util.spec_from_file_location(
    "aml_mm_scope_stage1", Path(__file__).parents[1] / "scripts" / "aml_mm_scope_stage1.py"
)
assert SPEC and SPEC.loader
stage1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage1
SPEC.loader.exec_module(stage1)

MM4 = {"order": ["S", "M4r"], "ports": {"S": 18037, "M4r": 18038}, "ingest": "S",
       "expect": {"S": {"multimodal_scope": "dual", "image_text.build": True, "image_text.leg": False},
                  "M4r": {"multimodal_scope": "dual", "image_text.leg": True}}}


def _version(scope: str, leg: bool, commit: str = "abc") -> dict:
    return {"git_commit": commit, "multimodal_scope": scope,
            "image_text": {"build": True, "leg": leg, "shown": False, "model": "m"}}


def _config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "arms.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_without_a_config_the_arms_are_mm1s() -> None:
    arms = stage1.load_arms(None)
    assert arms.ports == {"B": 18031, "B2": 18031, "P": 18032, "D": 18033}
    assert arms.order == ("B", "P", "D", "B2") and arms.ingest == "B"
    assert {arm: fields["multimodal_scope"] for arm, fields in arms.expect.items()} == {
        "B": "route", "P": "preserve", "D": "dual"}


def test_an_arm_config_must_name_its_ingest_arm(tmp_path: Path) -> None:
    assert stage1.load_arms(_config(tmp_path, MM4)).order == ("S", "M4r")
    with pytest.raises(stage1.Stage1Error, match="ingest"):
        stage1.load_arms(_config(tmp_path, {**MM4, "ingest": "X"}))
    with pytest.raises(stage1.Stage1Error, match="order"):
        stage1.load_arms(_config(tmp_path, {**MM4, "order": ["S"]}))


def test_a_wrong_setting_on_any_arm_refuses_the_run(tmp_path: Path) -> None:
    arms = stage1.load_arms(_config(tmp_path, MM4))
    stage1.check_versions({"S": _version("dual", False), "M4r": _version("dual", True)}, arms)
    with pytest.raises(stage1.Stage1Error, match="multimodal_scope"):
        stage1.check_versions({"S": _version("route", False), "M4r": _version("dual", True)}, arms)
    with pytest.raises(stage1.Stage1Error, match="different commits"):
        stage1.check_versions({"S": _version("dual", False), "M4r": _version("dual", True, "def")}, arms)


def test_a_nested_field_is_read_through_its_parent(tmp_path: Path) -> None:
    arms = stage1.load_arms(_config(tmp_path, MM4))
    with pytest.raises(stage1.Stage1Error, match="image_text.leg"):
        stage1.check_versions({"S": _version("dual", True), "M4r": _version("dual", True)}, arms)
