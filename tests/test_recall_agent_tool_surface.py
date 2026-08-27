"""The model-facing tool surface may not be wider than the MCP server's.

`tests/test_recall_agent_descriptions.py` pins the model-facing DESCRIPTIONS to the server's
docstrings. Nothing pinned the SCHEMAS, and that gap shipped a real defect: `recall_index` was
given a `glob` parameter the MCP server deliberately withholds, and any non-default glob switches
off `recall.index._safe_default_file`, the exclusion list that keeps `tokens.json`,
`credentials.json` and `secrets.json` out of a searchable corpus. `SECURITY.md` names reading
"outside the allowed glob" as a reportable vulnerability, and `recall/index.py`'s own docstring
calls that filter "a security boundary rather than a consistency nicety" because it was added
after exactly this exposure.

A parameter the MODEL controls is not the same risk as one an operator controls: the CLI's
`--glob` is a human choosing a scan, while a tool argument is reachable by any text the corpus or
a web page can get in front of the model. So the rule pinned here is narrow and mechanical: the
in-process tools accept no argument the server's tool does not. Narrower is fine; wider is not.
"""
from __future__ import annotations

import inspect
import re

import pytest

from recall_agent import _sdk


def _server_tool_parameters(name: str) -> set[str]:
    """Parameter names of a `recall_mcp.server` tool, read from its source.

    Read from source rather than by importing the tool object, because the tools are defined
    inside `build_server`'s closure and are not module attributes.
    """
    pytest.importorskip("mcp")
    import recall_mcp.server as server

    source = inspect.getsource(server)
    match = re.search(rf"async def {name}\((.*?)\) -> str:", source, re.DOTALL)
    assert match, f"no tool named {name} found in recall_mcp.server"
    parameters = set()
    for line in match.group(1).split(","):
        candidate = line.strip().split(":")[0].strip()
        if candidate and candidate not in {"ctx", "self"}:
            parameters.add(candidate)
    return parameters


@pytest.mark.parametrize(
    ("schema_name", "tool_name"),
    [
        ("SEARCH_SCHEMA", "recall_search"),
        ("EVIDENCE_SCHEMA", "recall_evidence"),
        ("INDEX_SCHEMA", "recall_index"),
        ("FORGET_SCHEMA", "recall_forget"),
    ],
)
def test_no_in_process_tool_accepts_an_argument_the_server_withholds(
    schema_name: str, tool_name: str
) -> None:
    schema = getattr(_sdk, schema_name)
    extra = set(schema["properties"]) - _server_tool_parameters(tool_name)
    assert not extra, (
        f"{tool_name} exposes {sorted(extra)} to the model, which recall_mcp.server's "
        f"{tool_name} does not accept. A model-facing parameter the server withholds can widen "
        f"what the corpus admits; see this module's docstring."
    )


def test_the_index_tool_does_not_let_the_model_choose_the_glob() -> None:
    """The specific regression: `glob` disables the secret-file exclusions wholesale.

    Pinned separately from the parity test above so the reason survives even if the server's
    signature changes: `recall.index._safe_default_file` returns True unconditionally for any
    glob other than the default, so a model-chosen glob admits `tokens.json`.
    """
    assert "glob" not in _sdk.INDEX_SCHEMA["properties"]


def test_the_index_handler_cannot_be_passed_a_glob_by_the_model() -> None:
    """The schema constrains a well-behaved model; the handler must constrain the rest.

    A tool schema is advisory: it shapes what the model is told to send, not what the runtime
    accepts. So the handler is pinned too, by source, to forward no caller-supplied glob.
    """
    from recall_agent import memory as memory_module

    source = inspect.getsource(memory_module)
    handler = source[
        source.index("async def _recall_index") : source.index("async def _recall_forget")
    ]
    # Comments are stripped first: the handler carries a comment explaining WHY there is no glob,
    # and a check that cannot tell an explanation from an implementation would forbid documenting
    # the fix it exists to protect.
    code = "\n".join(
        line for line in handler.splitlines() if not line.strip().startswith("#")
    )
    assert "glob" not in code, (
        "_recall_index still reads or forwards a glob; the model must not be able to widen the "
        "exclusion list even by sending an argument the schema does not advertise."
    )
