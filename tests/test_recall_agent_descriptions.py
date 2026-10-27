"""Pin the in-process tool descriptions to the MCP server's docstrings.

The model-facing guidance ("only `ok` hits", "treat `abstained: true` as no supported answer",
the evidence DATA-not-instruction rule) was written once, in `recall_mcp/server.py`, and every
measured search-rate result was produced against that wording. `recall_agent._descriptions`
copies the lead paragraphs; this test reads the server SOURCE (no `mcp` import needed) and fails
if either surface drifts from the other.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from recall_agent._descriptions import (
    RECALL_EVIDENCE_DESCRIPTION,
    RECALL_SEARCH_DESCRIPTION,
)

_SERVER_SOURCE = Path(__file__).resolve().parents[1] / "recall_mcp" / "server.py"


def _normalized(text: str) -> str:
    return " ".join(text.split())


@pytest.mark.parametrize(
    "description",
    [RECALL_SEARCH_DESCRIPTION, RECALL_EVIDENCE_DESCRIPTION],
    ids=["recall_search", "recall_evidence"],
)
def test_the_description_is_verbatim_from_the_server_docstring(description: str) -> None:
    source = _normalized(_SERVER_SOURCE.read_text(encoding="utf-8"))
    assert _normalized(description) in source, (
        "recall_agent._descriptions has drifted from the recall_mcp/server.py docstring; "
        "update whichever surface changed so both say the same thing"
    )


def test_search_description_qualifies_historical_citations() -> None:
    """DOC-001: models must receive the same conditional historical contract as the response.

    Red proof: PR HEAD 0367a335's exposed search description unconditionally forbids historical
    citations, failing the assertion 'tool description omits historical eligibility'.
    The existing parity test independently pins the canonical MCP source to this description.
    """
    assert "historical.candidates" in RECALL_SEARCH_DESCRIPTION, (
        "tool description omits historical eligibility")
    assert "valid_from" in RECALL_SEARCH_DESCRIPTION and "valid_until" in RECALL_SEARCH_DESCRIPTION
