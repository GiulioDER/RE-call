"""What the AGENT actually receives — asserted against LangChain's own machinery, not our belief.

The adapter fix (only verdict-`ok` hits by default, in-band warning when `include_untrusted` is
set) rested on a claim about a *third party*: that a standard LangChain consumer renders
`page_content` alone and drops `metadata`, so a `recall_verdict` key is not a safety control. The
adapter's own unit tests cannot check that — they assert on `Document` objects, which is the layer
above the one where the claim lives.

So this pins it at the real boundary. `create_retriever_tool` is `langchain_core`'s own helper for
handing a retriever to an agent — the primary way RE-call is meant to be consumed — and it does:

    document_prompt_ = document_prompt or PromptTemplate.from_template("{page_content}")
    content = document_separator.join(format_document(doc, document_prompt_) for doc in docs)

`{page_content}` only. Every `recall_*` metadata key is dropped before the model ever sees it.
That is not a quirk of one chain type; it is the default of the core agent path, in a package this
project already depends on.

These tests therefore fail if LangChain ever changes that default (in which case the reasoning
behind the fix needs revisiting) *and* if the adapter regresses. Both are things we want to hear
about.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from recall.types import (
    Chunk,
    Provenance,
    StalenessReport,
    TrustedHit,
    TrustedResult,
    Validity,
)

pytest.importorskip("langchain_core")

STALE_TEXT = "rate limit is 100 rps"
CURRENT_TEXT = "rate limit is 500 rps"


def _hit(text: str, file: str, *, verdict: str = "ok", superseded_by: str | None = None):
    return TrustedHit(
        chunk=Chunk(id=file, source="mem", text=text, metadata={"file": file}),
        cosine=0.78,
        confidence=1.0,
        verdict=verdict,  # type: ignore[arg-type]
        provenance=Provenance(source="mem", file=file, ord=0,
                              indexed_at=datetime(2026, 1, 1, tzinfo=timezone.utc)),
        validity=Validity(valid_from=None, valid_until=None, superseded_by=superseded_by),
    )


def _mixed_result() -> TrustedResult:
    return TrustedResult(
        query="how many rps?",
        hits=[
            _hit(CURRENT_TEXT, "rate_v2.md"),
            _hit(STALE_TEXT, "rate_v1.md", verdict="superseded", superseded_by="rate_v2.md"),
        ],
        abstained=False,
        reason="",
        calibrated=True,
        gap_warning=False,
        staleness=StalenessReport(stale=False, newest_indexed_at=None, age=None,
                                  max_age=timedelta(days=1)),
    )


def _agent_tool_output(**retriever_kwargs) -> str:
    """The exact string `create_retriever_tool` hands an agent for our mixed result."""
    from langchain_core.tools import create_retriever_tool

    from recall.integrations.langchain import RecallRetriever

    tool = create_retriever_tool(
        RecallRetriever(search_fn=lambda _q: _mixed_result(), **retriever_kwargs),
        "recall",
        "search agent memory",
    )
    return tool.invoke({"query": "how many rps?"})


def test_the_superseded_memory_never_reaches_the_agent():
    output = _agent_tool_output()

    assert CURRENT_TEXT in output
    assert STALE_TEXT not in output, "the superseded memory reached the agent's context"


def test_opting_in_delivers_the_warning_where_the_model_will_see_it():
    output = _agent_tool_output(include_untrusted=True)

    assert STALE_TEXT in output, "opting in must actually include the hit"
    assert "RE-CALL WARNING" in output, "the warning did not survive into the agent's context"
    assert "superseded" in output.lower()


def test_metadata_is_dropped_by_the_standard_agent_path():
    """The premise of the whole fix, asserted against LangChain rather than assumed.

    If this ever fails, `recall_verdict` in `metadata` HAS become visible to the model by
    default — and the reasoning behind returning ok-only hits should be revisited rather than
    silently kept.
    """
    output = _agent_tool_output(include_untrusted=True)

    assert "recall_verdict" not in output, (
        "metadata now reaches the model; re-examine why the adapter filters"
    )
    assert "rate_v1.md" not in output.replace('"rate_v2.md"', ""), (
        "provenance keys are not rendered by the default document prompt"
    )
