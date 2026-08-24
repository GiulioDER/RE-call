"""Exact reader token accounting for public benchmark artifacts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any
from typing import Protocol


class TokenCounter(Protocol):
    """Reader tokenizer contract used for exact evidence and total input counts."""

    tokenizer_id: str
    tokenizer_revision: str

    def count_tokens(self, text: str) -> int:
        """Return the nonnegative token count for one rendered text payload."""



@dataclass(frozen=True)
class PinnedReaderTokenizer:
    """A pinned tokenizer adapter used for comparable evidence cost curves."""

    tokenizer_id: str = "cl100k_base"
    tokenizer_revision: str = "tiktoken-0.13.0"

    def _encoding(self) -> Any:
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "exact benchmark token accounting needs tiktoken; install recall-rag[bench]"
            ) from exc
        return tiktoken.get_encoding(self.tokenizer_id)

    def count_tokens(self, text: str) -> int:
        return len(self._encoding().encode(text))

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Return the UTF 8 text prefix whose reader token count is at most ``max_tokens``."""
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0:
            raise ValueError("max_tokens must be a nonnegative int")
        encoding = self._encoding()
        return str(encoding.decode(encoding.encode(text)[:max_tokens]))

    def encoding_hash(self) -> str:
        """Hash the mergeable ranks and special tokens, not just the package version."""
        encoding = self._encoding()
        digest = hashlib.sha256()
        for token, rank in sorted(encoding._mergeable_ranks.items()):
            digest.update(len(token).to_bytes(4, "big"))
            digest.update(token)
            digest.update(int(rank).to_bytes(4, "big"))
        for token, rank in sorted(encoding._special_tokens.items()):
            encoded = token.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "big"))
            digest.update(encoded)
            digest.update(int(rank).to_bytes(4, "big"))
        return digest.hexdigest()

    def metadata(self) -> dict[str, str]:
        return {
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_hash": self.encoding_hash(),
            "basis": "reader_tokenizer",
        }


def prompt_token_cost(
    system_prompt: str,
    user_message: str,
    tokenizer: TokenCounter,
    evidence_text: str | None = None,
) -> dict[str, int]:
    """Return rendered evidence payload and total reader input token counts.

    ``evidence_text`` is normally the exact evidence block embedded in ``user_message``.  The
    fallback keeps the helper compatible with older callers that supplied only one payload.
    """
    evidence_tokens = tokenizer.count_tokens(user_message if evidence_text is None else evidence_text)
    total_tokens = tokenizer.count_tokens(system_prompt + "\n" + user_message)
    return {
        "evidence_tokens_exact": evidence_tokens,
        "input_tokens_exact": total_tokens,
    }


def truncate_evidence_context(
    context: str,
    budget: int,
    tokenizer: TokenCounter,
    *,
    prefix: str = "<memories>\n",
    suffix: str = "\n</memories>",
) -> str:
    """Fit a context into an exact rendered evidence token budget.

    The returned text is a prefix in reader token space. The wrapper is counted as evidence, so
    the function never confuses a raw context count with the bytes actually sent to the reader.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("evidence budget must be a positive int")
    def rendered(value: str) -> str:
        return f"{prefix}{value}{suffix}"
    if tokenizer.count_tokens(rendered(context)) <= budget:
        return context
    truncator = getattr(tokenizer, "truncate_text", None)
    if not callable(truncator):
        raise TypeError("exact evidence budgeting requires a tokenizer with truncate_text")
    high = tokenizer.count_tokens(context)
    low = 0
    while low < high:
        candidate_tokens = (low + high + 1) // 2
        candidate = truncator(context, candidate_tokens)
        if tokenizer.count_tokens(rendered(candidate)) <= budget:
            low = candidate_tokens
        else:
            high = candidate_tokens - 1
    return str(truncator(context, low))


__all__ = [
    "PinnedReaderTokenizer",
    "TokenCounter",
    "prompt_token_cost",
    "truncate_evidence_context",
]
