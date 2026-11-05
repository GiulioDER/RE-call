"""Every credential redactor hides every shape it promises, and none goes unregistered (M1).

Invariant: each function in the production packages named like a redactor (`redact*`, `scrub*`,
`sanitize*`, `mask*`) is either in `REDACTORS`, with the shape kinds of `tests/secret_shapes.py`
it promises to hide, or in `NOT_CREDENTIAL` with the reason it handles something else. Every
registered redactor leaves no run of the secret in any shape of its kinds.

The failure modes: a redactor that misses a form (CCA's most frequent `secret-exposure` shape,
three regressions in `recall_mcp/config_report.py` alone), and a new redactor written beside the
corpus so the forms already learned are never run against it.

Red proof (2026-11-04), each restored afterwards. The first two are master's code, live misses the
corpus found on its first run:

- the source-policy secret rule without `pgpassword` in `_CREDENTIAL_WORD`:
  `[security_policy secret rule-pgpassword]` failed, `PGPASSWORD=<secret>` left in stored text.
- `recall_mcp.config_report._scrub_message` without its `scrub_dsn_secrets` call:
  `[config_report._scrub_message-libpq-quotes-the-password]` failed, the password libpq quotes
  from a malformed DSN left in the message.
- `recall.dsn_security.redacted_dsn` returning its input unchanged:
  `test_redactor_hides_every_shape_it_promises[redacted_dsn-dsn-url]` and the other `dsn` shapes
  failed in the `leaked(...) == []` assertion.
- A new module-level `def scrub_tokens(text)` added to `recall/claude_code.py`:
  `test_every_redactor_is_registered` failed naming `recall.claude_code:scrub_tokens`.
"""

from __future__ import annotations

import ast
import shlex
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.secret_shapes import SHAPES, Shape, leaked

PACKAGES = (
    "recall", "recall_mcp", "recall_hooks", "recall_aml", "recall_agent", "recall_interop",
    "recall_consistency",
)
REPO = Path(__file__).resolve().parents[1]
NAMES = ("redact", "scrub", "sanitize", "sanitise", "mask")


def _redacted_dsn(shape: Shape) -> str:
    from recall.dsn_security import redacted_dsn

    return redacted_dsn(shape.text)


def _claude_code_redacted(shape: Shape) -> str:
    from recall.claude_code import _redacted

    return _redacted(shape.text)


def _scrub_dsn_secrets(shape: Shape) -> str:
    from recall.dsn_security import scrub_dsn_secrets

    return scrub_dsn_secrets(shape.text, shape.dsn)


def _install_ui_scrubbed(shape: Shape) -> str:
    from recall.desktop.install_ui import _scrubbed

    return _scrubbed(shape.text, shape.dsn)


def _gap_run_scrub(shape: Shape) -> str:
    from recall.eval.gap_run import _scrub

    return _scrub(shape.text, shape.dsn)


def _redact_value(shape: Shape) -> str:
    from recall_mcp.config_report import redact_value

    return redact_value(shape.text)


def _redact_argv(shape: Shape) -> str:
    from recall_mcp.config_report import redact_argv

    return " ".join(redact_argv(shlex.split(shape.text)))


def _scrub_message(shape: Shape) -> str:
    from recall_mcp.config_report import _scrub_message

    return _scrub_message(shape.text, {"RECALL_DSN": shape.dsn} if shape.dsn else {})


def _general_knowledge_redact(shape: Shape) -> str:
    from recall_hooks.general_knowledge import _redact

    return _redact(shape.text.lower())


def _general_knowledge_pairs(shape: Shape) -> str:
    from recall_hooks.general_knowledge import _redact_credential_pairs

    return _redact_credential_pairs(shape.text.lower())


def _policy_secret_rule(shape: Shape) -> str:
    from recall.security_policy import _REDACTIONS

    return _REDACTIONS["secret"].sub("[REDACTED:secret]", shape.text)


#: name -> (call on a shape, the shape kinds it promises to hide, the functions it covers).
REDACTORS: dict[str, tuple[Callable[[Shape], str], frozenset[str], tuple[str, ...]]] = {
    "redacted_dsn": (_redacted_dsn, frozenset({"dsn"}), ("recall.dsn_security:redacted_dsn",)),
    "claude_code._redacted": (_claude_code_redacted, frozenset({"dsn"}), ("recall.claude_code:_redacted",)),
    "scrub_dsn_secrets": (
        _scrub_dsn_secrets, frozenset({"dsn", "dsn-in-text"}), ("recall.dsn_security:scrub_dsn_secrets",)
    ),
    "install_ui._scrubbed": (
        _install_ui_scrubbed, frozenset({"dsn", "dsn-in-text"}), ("recall.desktop.install_ui:_scrubbed",)
    ),
    "gap_run._scrub": (_gap_run_scrub, frozenset({"dsn", "dsn-in-text"}), ("recall.eval.gap_run:_scrub",)),
    "redact_value": (
        _redact_value, frozenset({"dsn", "url"}), ("recall_mcp.config_report:redact_value",)
    ),
    "redact_argv": (_redact_argv, frozenset({"argv"}), ("recall_mcp.config_report:redact_argv",)),
    "config_report._scrub_message": (
        _scrub_message, frozenset({"dsn", "dsn-in-text", "url"}),
        ("recall_mcp.config_report:_scrub_message",),
    ),
    "general_knowledge._redact": (
        _general_knowledge_redact, frozenset({"assignment", "header", "url"}), ("recall_hooks.general_knowledge:_redact",)
    ),
    "general_knowledge._redact_credential_pairs": (
        _general_knowledge_pairs, frozenset({"assignment"}),
        ("recall_hooks.general_knowledge:_redact_credential_pairs",),
    ),
    "security_policy secret rule": (_policy_secret_rule, frozenset({"assignment", "header"}), ()),
}

#: Functions named like a redactor that do not redact credentials, with what they do instead.
NOT_CREDENTIAL: dict[str, str] = {
    "recall.control_plane:ControlPlane._scrub_locked": "erases a forgotten source from outbox events",
    "recall.fact_redaction:redacted_fact": "builds the redacted form of an erased fact",
    "recall.fact_redaction:redacted_card": "builds the redacted form of an erased card",
    "recall.fact_redaction:LiteFactRedactor.redact": "erases forgotten memo values from the fact ledger",
    "recall.fact_redaction:PostgresFactRedactor.redact": "erases forgotten memo values from the fact ledger",
    "recall.generations:GenerationManager._scrub_sparse_rows": "deletes sparse rows of erased chunks",
    "recall.store:PgVectorStore._scrub_sparse_rows": "deletes sparse rows of erased chunks",
    "recall.security_policy:SourceSecurityPolicy.redact": "applies a policy's rules; the secret rule is registered",
    "recall.security_policy:SourceSecurityPolicy.redact_with_decision": (
        "applies a policy's rules; the secret rule is registered"
    ),
    "recall_mcp.indexing:_scrub_paths": "hides filesystem paths, not credentials",
    "recall_hooks.general_knowledge:_redact_emails": "hides mailboxes; credentials are `_redact`'s",
    "recall_aml.app:_scrub_surrogates": "replaces lone surrogates so text can be encoded",
}


def _redactor_like(name: str) -> bool:
    bare = name.lstrip("_")
    return any(bare.startswith(prefix) for prefix in NAMES)


def _discovered() -> set[str]:
    """`module:qualname` of every module-level function and method named like a redactor."""
    found: set[str] = set()
    for package in PACKAGES:
        for path in sorted((REPO / package).rglob("*.py")):
            module = ".".join(path.relative_to(REPO).with_suffix("").parts).removesuffix(".__init__")
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _redactor_like(node.name):
                    found.add(f"{module}:{node.name}")
                elif isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and _redactor_like(item.name):
                            found.add(f"{module}:{node.name}.{item.name}")
    return found


def test_every_redactor_is_registered() -> None:
    covered = {name for _call, _kinds, names in REDACTORS.values() for name in names}
    unregistered = sorted(_discovered() - covered - set(NOT_CREDENTIAL))
    assert unregistered == [], (
        "register each in tests/test_secret_redactors.py: REDACTORS with the shape kinds it "
        "promises, or NOT_CREDENTIAL with what it does instead"
    )


def test_no_registration_is_stale() -> None:
    covered = {name for _call, _kinds, names in REDACTORS.values() for name in names}
    assert sorted((covered | set(NOT_CREDENTIAL)) - _discovered()) == []


CASES = [
    pytest.param(redactor, shape_name, id=f"{redactor}-{shape_name}")
    for redactor, (_call, kinds, _names) in REDACTORS.items()
    for shape_name, shape in SHAPES.items()
    if shape.kind in kinds
]


@pytest.mark.parametrize(("redactor", "shape_name"), CASES)
def test_redactor_hides_every_shape_it_promises(redactor: str, shape_name: str) -> None:
    call = REDACTORS[redactor][0]
    shape = SHAPES[shape_name]
    assert leaked(call(shape), shape) == []
