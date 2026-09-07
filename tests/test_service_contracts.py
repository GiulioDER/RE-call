from recall_mcp.models import EvidenceResult, SearchResult
from recall_mcp.service import EvidenceResult as ServiceEvidenceResult
from recall_mcp.service import SearchResult as ServiceSearchResult


def test_service_reexports_canonical_retrieval_models():
    assert ServiceSearchResult is SearchResult
    assert ServiceEvidenceResult is EvidenceResult


def test_missing_retrieval_lineage_is_not_labeled_legacy():
    assert SearchResult.model_fields["embedding_profile"].default is None
    assert SearchResult.model_fields["retrieval_profile"].default is None
    assert SearchResult.model_fields["index_generation"].default is None
    assert EvidenceResult.model_fields["embedding_profile"].default is None
    assert EvidenceResult.model_fields["retrieval_profile"].default is None
    assert EvidenceResult.model_fields["index_generation"].default is None
