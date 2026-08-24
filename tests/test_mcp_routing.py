from recall.profiles import QUALITY_PROFILE
import recall_mcp.service as service


def test_profile_selected_by_active_routing_reaches_reranker_builder(monkeypatch) -> None:
    seen = {}

    def fake_new_reranker(env=None, profile=None):
        del env
        seen["profile"] = profile
        return None

    service._reset_reranker_cache()
    monkeypatch.setattr("recall_mcp.factories._new_reranker", fake_new_reranker)
    assert service._build_reranker(QUALITY_PROFILE) is None
    assert seen["profile"] is QUALITY_PROFILE
