"""Model-facing rendering for the in-process Agent SDK tools.

Everything a tool returns to the model goes through this module, and the module keeps one
invariant: the only free text it emits is library-authored. A `SearchResult`'s `advice` is
assembled from library constants precisely because the tool description instructs the model to
obey it; corpus-controlled strings travel only inside the serialized result's data fields
(`reason`, `source`, `superseded_by`), never templated into instruction-position prose. Rendering
is therefore serialization and nothing else.

A `TrustRefusal` is rendered as its wire form (`to_dict()`: trust state, failure code, advice,
lineage identity, deliberately no hits and no query) with `is_error` left False. The refusal's
`advice` is the instruction channel the model is told to follow, and SDK clients may summarise or
hide error results, so marking the refusal an error would hide exactly the guidance that makes
failing closed actionable. An empty-but-trusted result stays distinguishable by construction: it
serializes with `trust_state: "trusted"` and an empty `hits`, while a refusal carries `code` and
`trust_state: "refused"` and no `hits` key at all.
"""
from __future__ import annotations

import json
from typing import Any

from recall.trust_policy import TrustRefusal
from recall_mcp.service import serving_json


def tool_text(text: str) -> dict[str, Any]:
    """Wrap `text` in the SDK tool-result content shape."""
    return {"content": [{"type": "text", "text": text}]}


def render_result(result: object) -> dict[str, Any]:
    """Serialize a service result byte-identically to the MCP server's tool output."""
    return tool_text(serving_json(result))


def render_refusal(refusal: TrustRefusal) -> dict[str, Any]:
    """Serialize a refusal's wire form; see the module docstring for why `is_error` stays unset."""
    return tool_text(json.dumps(refusal.to_dict(), indent=2))


def render_tool_error(detail: str) -> dict[str, Any]:
    """A bad tool ARGUMENT, rendered rather than raised.

    Distinct from a refusal, which is the trust layer declining to answer: this is the caller
    sending something the tool cannot use (`k: "five"`, a missing `query`, a string where a list
    of sources belongs). It is marked `is_error` because it genuinely is one and the model should
    retry with corrected arguments, where a refusal must NOT be marked so its advice survives.

    `detail` is an exception's type and message from ARGUMENT coercion only, which every caller
    performs before the retrieval starts, so nothing the corpus controls can reach this string.
    Service-layer failures are deliberately not routed here: they are the library's own signals,
    not the caller's mistake, and flattening them into "invalid tool arguments" would tell the
    model to retry an argument when the real answer was a refusal.
    """
    return {
        "content": [{"type": "text", "text": f"invalid tool arguments: {detail}"}],
        "is_error": True,
    }
