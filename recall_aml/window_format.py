"""Two ways to let AML's reader see a memory's date without changing what Coding retrieves.

Pre-registration: docs/preregistrations/2026-09-25-aml-c9-window-format.md

C9 stores windows as bare message content, so a date reaches AML's Answer model only through the
optional ``created_at``, which the platform may not render. Timestamping every message inside the
window won back 18.1 LoCoMo temporal points but cost the Coding screen 0.09 MRR. Both candidates
here are off unless a variant turns them on:

* ``dated_items`` prefixes each returned text item with its own ``created_at`` at Search time.
  Nothing stored, embedded or ranked changes; only the text handed to the reader does.
* ``looks_like_coding`` lets an Add choose its window renderer: a coding trajectory keeps
  content-only windows, anything else gets the timestamp, role and content renderer.

The detector reads message content only. It never reads ``request_id``, ``user_id`` or
``session_id``, which carry AML's benchmark names: branching on those would be dataset-specific
behaviour, not memory.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timezone
import re
from typing import Any

from recall_aml.models import SearchItem

#: A token naming a source, config or data file by a code-ish extension.
_CODE_FILE = re.compile(
    r"\b[\w.-]+\.(?:py|pyi|ipynb|js|jsx|ts|tsx|mjs|go|rs|java|kt|rb|php|c|cc|cpp|h|hpp|cs|swift"
    r"|sh|bash|ps1|sql|json|ya?ml|toml|ini|cfg|lock|md|html|css)\b",
    re.IGNORECASE,
)
#: Syntax that ordinary conversation rarely produces. A message needs two distinct ones.
_SYNTAX = (
    re.compile(r"\bdef \w+\s*\("),
    re.compile(r"\bclass \w+\s*[:(]"),
    re.compile(r"\bimport \w"),
    re.compile(r"\bfrom [\w.]+ import\b"),
    re.compile(r"\breturn\b"),
    re.compile(r"=>|==|!="),
    re.compile(r"\w\(\)"),
    re.compile(r"[{}]"),
    re.compile(r";\s*$", re.MULTILINE),
    re.compile(r"```"),
)
#: The share of non-empty messages that must look like code for the Add to be a trajectory.
CODING_MESSAGE_SHARE = 0.25
DATE_HEADER_FORMAT = "[%Y-%m-%d %H:%M UTC]"


def message_looks_like_code(content: str) -> bool:
    """A code file name, or two distinct kinds of code syntax."""
    if _CODE_FILE.search(content):
        return True
    return sum(1 for pattern in _SYNTAX if pattern.search(content)) >= 2


def looks_like_coding(messages: Sequence[Any]) -> bool:
    """Whether an Add is a coding trajectory, from its message contents alone."""
    contents = [
        message.content
        for message in messages
        if isinstance(message.content, str) and message.content.strip()
    ]
    if not contents:
        return False
    code_like = sum(message_looks_like_code(content) for content in contents)
    return code_like / len(contents) >= CODING_MESSAGE_SHARE


def dated_items(items: Sequence[SearchItem]) -> list[SearchItem]:
    """Prefix each text item with its own ``created_at``, in UTC to the minute.

    An item with no ``created_at``, or with multimodal content parts, is returned unchanged.
    """
    output: list[SearchItem] = []
    for item in items:
        if item.created_at is None or not isinstance(item.content, str):
            output.append(item)
            continue
        created = item.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        stamp = created.astimezone(timezone.utc).strftime(DATE_HEADER_FORMAT)
        output.append(item.model_copy(update={"content": f"{stamp} {item.content}"}))
    return output
