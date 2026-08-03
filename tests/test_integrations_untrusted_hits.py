"""An adapter must not hand a chain a memory the trust layer refused.

Both framework adapters returned `result.hits` wholesale. `trust.evaluate` builds that list as
`ok + rest`, so whenever at least ONE hit is `ok` the result does not abstain and every superseded,
expired, not-yet-valid or invalid-metadata hit rides along with it.

The verdict travels in `metadata`, which sounds sufficient and is not: LangChain's stock
`create_retrieval_chain` / `stuff_documents_chain` renders `page_content` alone into the prompt,
and LlamaIndex's default node parser does the same. So the payload went into the channel the
consumer reads and the safety signal went into a channel the consumer drops — the same shape as
the `advice` injection, one layer out.

The tell that this is a defect rather than a design choice is that the adapter is inconsistent
with ITSELF: when every hit is untrustworthy it abstains and returns nothing, but when one
unrelated hit happens to be `ok` those same untrustworthy memories are served. The identical
superseded memo is filtered or forwarded depending on what else matched — which is not a rule
anyone chose.

RE-call's whole premise is that the stale memory is often the highest-cosine hit, so this quietly
voided the headline promise at the integration boundary.
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

_INDEXED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _hit(text: str, file: str, *, verdict: str = "ok", superseded_by: str | None = None) -> TrustedHit:
    return TrustedHit(
        chunk=Chunk(id=f"{file}#0", source="memory", text=text, metadata={"file": file}),
        cosine=0.78,
        confidence=1.0,
        verdict=verdict,  # type: ignore[arg-type]
        provenance=Provenance(source="memory", file=file, ord=0, indexed_at=_INDEXED_AT),
        validity=Validity(valid_from=None, valid_until=None, superseded_by=superseded_by),
    )


def _mixed_result() -> TrustedResult:
    """One valid hit plus a superseded one — exactly `evaluate`'s `ok + rest` ordering."""
    return TrustedResult(
        query="how many rps?",
        hits=[
            _hit("rate limit is 500 rps", "rate_v2.md"),
            _hit("rate limit is 100 rps", "rate_v1.md",
                 verdict="superseded", superseded_by="rate_v2.md"),
        ],
        abstained=False,
        reason="",
        gap_warning=False,
        staleness=StalenessReport(
            stale=False, newest_indexed_at=None, age=None, max_age=timedelta(days=1)
        ),
        calibration_id="cal_fixture",
        calibration_status="certified",
    )


class TestLangChain:
    def test_a_superseded_memory_is_not_handed_to_the_chain(self):
        from recall.integrations.langchain import RecallRetriever

        docs = RecallRetriever(search_fn=lambda _q: _mixed_result()).invoke("how many rps?")

        contents = [d.page_content for d in docs]
        assert "rate limit is 100 rps" not in contents, "the superseded memory reached the chain"
        assert contents == ["rate limit is 500 rps"]

    def test_opting_in_marks_the_untrusted_text_in_band(self):
        """A caller may still ask for everything — but then the warning rides in `page_content`.

        Out-of-band metadata is what failed here, so the opt-in must not repeat that mistake.
        """
        from recall.integrations.langchain import RecallRetriever

        docs = RecallRetriever(
            search_fn=lambda _q: _mixed_result(), include_untrusted=True
        ).invoke("how many rps?")

        assert len(docs) == 2
        stale = next(d for d in docs if d.metadata["file"] == "rate_v1.md")
        assert "superseded" in stale.page_content.lower(), "no in-band warning on untrusted text"
        assert "rate limit is 100 rps" in stale.page_content, "the memory itself must survive"

    def test_the_valid_hit_is_untouched_when_opting_in(self):
        from recall.integrations.langchain import RecallRetriever

        docs = RecallRetriever(
            search_fn=lambda _q: _mixed_result(), include_untrusted=True
        ).invoke("how many rps?")

        good = next(d for d in docs if d.metadata["file"] == "rate_v2.md")
        assert good.page_content == "rate limit is 500 rps"


class TestLlamaIndex:
    def test_a_superseded_memory_is_not_handed_to_the_chain(self):
        pytest.importorskip("llama_index.core")
        from recall.integrations.llamaindex import RecallRetriever as RecallLlamaRetriever

        nodes = RecallLlamaRetriever(search_fn=lambda _q: _mixed_result()).retrieve("how many rps?")

        contents = [n.node.text for n in nodes]
        assert "rate limit is 100 rps" not in contents, "the superseded memory reached the chain"
        assert contents == ["rate limit is 500 rps"]

    def test_opting_in_marks_the_untrusted_text_in_band(self):
        pytest.importorskip("llama_index.core")
        from recall.integrations.llamaindex import RecallRetriever as RecallLlamaRetriever

        nodes = RecallLlamaRetriever(
            search_fn=lambda _q: _mixed_result(), include_untrusted=True
        ).retrieve("how many rps?")

        assert len(nodes) == 2
        stale = next(n for n in nodes if n.node.metadata["file"] == "rate_v1.md")
        assert "superseded" in stale.node.text.lower()
