"""Remove private memory-store content from committed experiment traces, keeping every number.

Some committed result and pool files captured whole memos from the private memory store as
retrieval-candidate text: infrastructure addresses, key fingerprints, personal emails and another
project's operational notes, in a public repository. This rewrites such a file so that:

* every content field (``CONTENT_KEYS``) becomes a placeholder carrying the SHA-256 prefix and the
  length of what it replaced, so a holder of the original can still match it;
* every memo path from the other project (``sentiment-agent/...``) becomes a stable pseudonym, the
  same pseudonym wherever the same path occurs, so joins between fields still hold;
* any IPv4 address other than loopback or unspecified, any SSH key fingerprint, any hosting server
  ID and any personal mailbox left in other string fields is replaced in place.

Numbers, ids, ranks, scores, labels and structure are unchanged.

    python scripts/redact_memo_traces.py FILE.json [FILE.json ...]
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

CONTENT_KEYS = frozenset({"text", "payload", "answer", "answer_span", "original_prompt", "label_note"})
OTHER_PROJECT = re.compile(r"sentiment-agent/[^\s\"'`,;)\]}]+")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
SAFE_IPV4 = frozenset({"127.0.0.1", "0.0.0.0"})  # noqa: S104, values to keep, not an address to bind
FINGERPRINT = re.compile(r"SHA256:[A-Za-z0-9+/]{20,}={0,2}")
SERVER_ID = re.compile(r"\bvmi\d{5,}\b")
PERSONAL_MAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@(?:gmail|hotmail|outlook|yahoo|icloud|proton(?:mail)?)\.[a-z]{2,}\b", re.I)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def placeholder(value: str) -> str:
    return f"[redacted 2026-09-26: private memo content, sha256 {_digest(value)[:16]}, {len(value)} chars]"


def pseudonym(match: re.Match[str]) -> str:
    if match.group(0).startswith("sentiment-agent/redacted-"):
        return match.group(0)  # already a pseudonym: a second pass must not re-hash it
    return f"sentiment-agent/redacted-{_digest(match.group(0))[:12]}"


def scrub_value(value: str) -> str:
    value = OTHER_PROJECT.sub(pseudonym, value)
    value = IPV4.sub(lambda m: m.group(0) if m.group(0) in SAFE_IPV4 else "[redacted-ip]", value)
    value = FINGERPRINT.sub("[redacted-fingerprint]", value)
    value = SERVER_ID.sub("[redacted-server-id]", value)
    return PERSONAL_MAIL.sub("[redacted-email]", value)


def scrub(node: Any, key: str = "") -> Any:
    if isinstance(node, dict):
        # Keys are scrubbed too: some traces key their per-source audits by memo path.
        return {scrub_value(k): scrub(v, k) for k, v in node.items()}
    if isinstance(node, list):
        return [scrub(v, key) for v in node]
    if isinstance(node, str):
        if key in CONTENT_KEYS and node and not node.startswith("[redacted 2026-09-26: "):
            return placeholder(node)
        return scrub_value(node)
    return node


def main() -> None:
    for name in sys.argv[1:]:
        path = Path(name)
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            out = json.dumps(scrub(json.loads(text)), indent=2, ensure_ascii=False) + "\n"
        elif path.suffix == ".jsonl":
            out = "".join(
                json.dumps(scrub(json.loads(line)), ensure_ascii=False) + "\n"
                for line in text.splitlines()
                if line.strip()
            )
        else:
            # Prose (Markdown, text): only the in-place redactions apply; there is no field to blank.
            out = scrub_value(text)
        path.write_text(out, encoding="utf-8")
        print(f"scrubbed {name}")


if __name__ == "__main__":
    main()
