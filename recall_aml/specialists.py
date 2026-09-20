"""Deterministic routing for isolated hosted embedding specialists."""

from __future__ import annotations

import re
from typing import Literal

from recall_aml.models import ContentValue, ImageContentPart


SpecialistRoute = Literal["code", "context", "multimodal"]
SPECIALIST_ROUTER_PROFILE = "conservative-specialist-router-v1"
SPECIALIST_FUSION_PROFILE = "routed-rank-fusion-v1"

_CODE_SIGNAL = re.compile(
    r"(?:`[^`]+`|\b(?:bug|code|compile|compiler|function|class|method|module|package|"
    r"repository|repo|test|pytest|exception|traceback|stack|api|schema|database|sql|"
    r"json|yaml|config|dependency|implementation|refactor|fix|build|deploy|endpoint|"
    r"typescript|javascript|python|rust|java|golang|git)\b|"
    r"(?:\.[A-Za-z0-9]{1,8}\b)|(?:[A-Za-z_][A-Za-z0-9_]*\([^\n)]*\)))",
    re.IGNORECASE,
)
_CONTEXT_SIGNAL = re.compile(
    r"\b(?:remember|conversation|meeting|discussion|preference|preferred|favorite|"
    r"yesterday|last week|last month|previously|earlier|we decided|we agreed|"
    r"you told me|i told you|what did|when did|who did|where did)\b",
    re.IGNORECASE,
)
_VISUAL_SIGNAL = re.compile(
    r"\b(?:image|photo|picture|screenshot|diagram|chart|figure|visual|screen|ui)\b",
    re.IGNORECASE,
)


def route_query(value: ContentValue) -> SpecialistRoute:
    """Choose one semantic space without comparing scores across embedding models.

    Multimodal input routes to the visual index. Text routes to Context4 only when it has
    an explicit conversational-memory signal and no code signal. Ambiguous text stays on
    Code4, which protects the measured CAMBench coding baseline.
    """
    if not isinstance(value, str):
        if any(isinstance(part, ImageContentPart) for part in value):
            return "multimodal"
        text = "\n".join(getattr(part, "text", "") for part in value)
    else:
        text = value
    if _VISUAL_SIGNAL.search(text):
        return "multimodal"
    if _CODE_SIGNAL.search(text):
        return "code"
    if _CONTEXT_SIGNAL.search(text):
        return "context"
    return "code"
