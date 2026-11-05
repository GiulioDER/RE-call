"""Inputs a gate cannot decide on, which every registered gate must refuse (M4).

CCA's `fail-open` class holds 55 confirmed RE-call findings, 21 of them P1 (measured 2026-11-04):
a guard, gate, budget or trust check that passed when it could not decide. The recurring inputs
were few: a NaN, which compares False with every threshold; an infinity; a value missing or empty;
a boolean spelled as a string that is truthy whatever it says; a misspelled key that fell back to
the permissive default; and a comparison over nothing that reported a match. This module is the one
list, and `tests/test_undecidable_gates.py` runs each registered gate against the entries of the
kinds it takes. "Refuse" means raise, or return the gate's negative answer: never pass.
"""

from __future__ import annotations

NAN = float("nan")

#: Numbers no threshold, score, budget or latency can be compared with.
NUMBERS: dict[str, float] = {"nan": NAN, "inf": float("inf"), "-inf": float("-inf")}

#: Strings a strict boolean parser must not read as either answer. An empty or blank value is not
#: here: `recall._env.strict_bool` reads it as unset, which is false by design.
FLAGS: dict[str, str] = {
    "misspelled-false": "flase",
    "maybe": "maybe",
    "number": "0.5",
}

#: Booleans written as strings into a JSON config, where only a real `false` turns a thing off.
STRING_FALSES: dict[str, str] = {"false": "false", "off": "off", "no": "no", "zero": "0"}

#: Mode names that differ from a strict mode only by case or space, and must never select the
#: permissive one.
MODE_SPELLINGS: dict[str, str] = {"upper": "STRICT", "leading-space": " strict", "empty": ""}

#: A misspelled key, for loaders whose every default is the permissive one.
MISSPELLED_KEYS: dict[str, str] = {"policy": "default_dney", "rule": "clasification"}
