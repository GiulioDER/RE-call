"""The MCP surface proposes. A human applies at the CLI.

The MCP client IS the model. Letting it supply `reviewer_id` and `audit_note` would make the
named human gate a formality it satisfies by typing a string: the gate becomes a field, not a
person. So `recall_rewrite_plan` ships read only and `recall_rewrite_apply` deliberately does
not ship at all.

`recall_mcp/` contained zero file write calls before this work, and this must not be the change
that makes it the first thing to mutate a user's documents.

Properties, one test each:

1. No apply tool is registered, under any spelling.
2. The plan tool is registered and annotated read only.
3. The MCP package makes no file write call at all.
4. `include_extracted` defaults to False on the service, so existing behaviour is unchanged.
5. It defaults to False on the tool too.
6. Asking for extracted proposals refuses rather than returning a misleading empty list.
7. The CLI flag defaults to off.
8. Nothing on this path constructs an extraction engine, which would put a model on the query
   path.
"""
import ast
import inspect
import re
from pathlib import Path

import pytest

import recall_mcp.server as server
from recall_mcp.service import _stored_extracted_proposals, reasoning_proposals

REPO_ROOT = Path(__file__).resolve().parents[1]
MCP_DIR = REPO_ROOT / "recall_mcp"


def _registered_tools() -> dict[str, dict[str, bool]]:
    """Every `@mcp.tool(name=..., annotations=ToolAnnotations(...))` in the server, by name.

    Parsed rather than string-searched. A raw `"recall_rewrite_apply" not in source` check is
    fooled by PROSE: the plan tool's own docstring explains why no apply tool exists, and that
    sentence made the assertion fail. Registration is the property; mentioning a name is not.
    """
    tree = ast.parse(Path(inspect.getfile(server)).read_text(encoding="utf-8"))
    tools: dict[str, dict[str, bool]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "tool"):
            continue
        name = next(
            (k.value.value for k in node.keywords if k.arg == "name" and isinstance(k.value, ast.Constant)),
            None,
        )
        if name is None:
            continue
        hints: dict[str, bool] = {}
        annotations = next((k.value for k in node.keywords if k.arg == "annotations"), None)
        if isinstance(annotations, ast.Call):
            hints = {
                k.arg: k.value.value
                for k in annotations.keywords
                if k.arg and isinstance(k.value, ast.Constant) and isinstance(k.value.value, bool)
            }
        tools[name] = hints
    return tools


def test_no_apply_tool_is_registered():
    assert "recall_rewrite_apply" not in _registered_tools()


def test_the_plan_tool_is_registered_and_annotated_read_only():
    tools = _registered_tools()
    assert "recall_rewrite_plan" in tools
    assert tools["recall_rewrite_plan"]["read_only_hint"] is True
    assert tools["recall_rewrite_plan"]["destructive_hint"] is False
    assert tools["recall_rewrite_plan"]["idempotent_hint"] is True


def test_every_registered_tool_name_is_unique():
    """A duplicate name would silently shadow one tool with another."""
    tree = ast.parse(Path(inspect.getfile(server)).read_text(encoding="utf-8"))
    names = [
        k.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "tool"
        for k in node.keywords
        if k.arg == "name" and isinstance(k.value, ast.Constant)
    ]
    assert len(names) == len(set(names))


WRITERS = frozenset(
    {"write_text", "write_bytes", "atomic_write_bytes", "apply_rewrite", "mkdir", "unlink"}
)


def test_the_mcp_package_makes_no_file_write_call():
    """Walks every module with `ast`, so an indented or aliased call cannot hide."""
    offending: list[str] = []
    for path in sorted(MCP_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                if name in WRITERS:
                    offending.append(f"{path.name}:{node.lineno} {name}()")
    assert not offending, f"recall_mcp writes files: {offending}"


def test_the_mcp_package_does_not_even_import_a_writer():
    """Checked separately, because an IMPORT is not a Call.

    The call check alone stayed green when `apply_rewrite` was added to an import statement.
    Importing without calling writes nothing today, but it puts the writer one keystroke from
    the query path in the package documented as making zero file writes.
    """
    offending: list[str] = []
    for path in sorted(MCP_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                offending += [
                    f"{path.name}:{node.lineno} from {node.module} import {a.name}"
                    for a in node.names
                    if a.name in WRITERS
                ]
    assert not offending, f"recall_mcp imports a writer: {offending}"


def test_include_extracted_defaults_to_false_on_the_service():
    """Existing behaviour must be byte identical by default, mirroring `include_text`."""
    assert inspect.signature(reasoning_proposals).parameters["include_extracted"].default is False


def test_include_extracted_defaults_to_false_on_the_tool():
    source = inspect.getsource(server)
    start = source.index("async def recall_reasoning_proposals")
    window = source[start : start + 400]
    assert "include_extracted: bool = False" in window


def test_asking_for_extracted_proposals_refuses_rather_than_returning_an_empty_list():
    """No extraction is persisted where the query path can read it.

    Returning `()` would print "0 proposals", which a caller reads as "the extractor found
    nothing" when the truth is "nothing was ever recorded". Those call for opposite responses.
    """
    with pytest.raises(ValueError, match="no extraction record"):
        _stored_extracted_proposals(object())


def test_the_cli_flag_exists(capsys):
    """Word-bounded. A plain `in` check passes for `--include-extractedX`, since the real flag
    name is a substring of the typo, so renaming the flag left this green."""
    from recall.cli import main

    with pytest.raises(SystemExit):
        main(["reasoning", "proposals", "--help"])
    assert re.search(r"--include-extracted\b", capsys.readouterr().out)


def test_the_cli_flag_defaults_to_off():
    """`store_true` is what makes existing behaviour byte identical for anyone who does not ask.

    Read off the parser definition with `ast`, because `recall reasoning proposals` opens the
    database and this property does not need one to be true.
    """
    tree = ast.parse((REPO_ROOT / "recall" / "cli.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"):
            continue
        first = node.args[0] if node.args else None
        if isinstance(first, ast.Constant) and first.value == "--include-extracted":
            action = next(
                (k.value.value for k in node.keywords if k.arg == "action"), None
            )
            default = next((k.value for k in node.keywords if k.arg == "default"), None)
            assert action == "store_true", f"action is {action!r}, so the default is not False"
            assert default is None, "an explicit default would override store_true's False"
            return
    raise AssertionError("--include-extracted is not defined in recall/cli.py")


def test_nothing_on_this_path_builds_an_extraction_engine():
    """`max_model_calls` is 0 on the query path. Extraction is an ingest concern."""
    offending: list[str] = []
    for path in sorted(MCP_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                if name in {"resolve_extraction_engine", "extract_corpus_claims"}:
                    offending.append(f"{path.name}:{node.lineno} {name}()")
    assert not offending, f"recall_mcp reaches the extractor: {offending}"
