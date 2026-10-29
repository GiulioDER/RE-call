import asyncio
import pytest
from recall.pool import SharedPool
from recall.store import PgVectorStore
from recall_aml.storage import PgHostedRepository
from recall_aml.service import HostedService
from recall_aml.retrieval import HostedRetriever
from recall_aml.models import AddRequest, Message
from recall_aml.identity import tenant_for, specialist_tenant
from recall_aml.variants import variant
from tests.conftest import TEST_DSN, requires_db
from tests.test_aml_atomic_views import C9, _Embedder, _Compiler, _Reranker, _session_messages


@requires_db
@pytest.mark.parametrize('erased_scope', ['specialist', 'specialist_views'])
def test_delete_after_disabling_specialist_removes_prior_evidence(make_store, erased_scope):
    """FIX-005: acknowledged Delete removes all previously supported specialist scopes.

    Red proof: baseline8c296fd31 leaves9 specialist rows and their atomic views after
    deleting through A0_raw. The exact final count assertion must fail before the repair.
    """
    fixture_store = make_store(3)
    pool = SharedPool(TEST_DSN, min_size=1, max_size=8)
    base = PgVectorStore(TEST_DSN, 3, table=fixture_store.table,
                         tenant='aml_service_readiness', shared_pool=pool, owns_pool=True)
    profile = C9.context_embedding_profile
    try:
        enabled = PgHostedRepository(base, _Embedder('code'),
                    specialist_embedders={profile: _Embedder('context')})
        writer = HostedService(enabled, _Compiler(),
                    HostedRetriever(_Embedder('code'), _Reranker()), behavior=C9,
                    multimodal_embedder=object(),
                    specialist_retrievers={profile: HostedRetriever(_Embedder('context'), _Reranker())})
        user = 'scope-switch-' + fixture_store.table
        tenant = tenant_for(user)
        asyncio.run(writer.add(AddRequest(request_id='scope-switch-add', user_id=user,
                    session_id='scope-switch-session', messages=[Message(role='user', content=x)
                    for x in _session_messages(12, needle_in=10)])))
        raw = enabled.tenant_store(tenant)
        specialist = enabled.specialist_store(tenant, profile)
        views = enabled.atomic_view_store(specialist_tenant(tenant, profile))
        assert raw.count() > 0
        assert specialist.count() > 0
        assert views.count() > 0
        disabled = PgHostedRepository(base, _Embedder('code'))
        deleter = HostedService(disabled, None,
                    HostedRetriever(_Embedder('code'), _Reranker()), behavior=variant('A0_raw'))
        assert asyncio.run(deleter.delete_user(user)) > 0
        assert raw.count() == 0
        reopened = PgHostedRepository(base, _Embedder('code'),
                    specialist_embedders={profile: _Embedder('context')})
        remaining = (reopened.specialist_store(tenant, profile) if erased_scope == 'specialist'
                     else reopened.atomic_view_store(specialist_tenant(tenant, profile)))
        assert remaining.count() == 0, 'Delete acknowledged but previously configured specialist evidence survived'
    finally:
        base.close()
