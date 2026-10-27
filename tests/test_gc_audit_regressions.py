"""Behavioral regressions from the independently verified fact audit.

The original combined nodes failed their intended assertions before repairs.
Splitting by feature ownership preserves the exact test bodies for the PR stack.
"""
from __future__ import annotations
import json
from dataclasses import replace
from recall.provenance_controller import DecisionCode, FactApplicationRequest, ProvenanceController
from tests.test_fact_retraction import DOOR
from tests.test_gc_retraction import manager as manager, three as three, _gc, _bare_current, _ledger
from tests.fact_fixtures import NOW, evidence_card

def test_stakes001_gc_keeps_fact_supported_by_real_legacy_row(three, make_store, tmp_path):
    from recall.embeddings import HashingEmbedder
    from recall.index import Indexer
    from recall.provenance_cards import PostgresEvidenceCardStore
    from recall.store import PgVectorStore
    from tests.conftest import TEST_DSN
    manager, first = three
    tenant = manager.tenant_id
    table = make_store(64).table
    root = tmp_path / 'legacy-corpus'
    root.mkdir()
    memo = root / 'legacy-support.md'
    memo.write_text('---\nrecall_graph: ' + json.dumps({'facts':[DOOR.to_payload()]}) + '\n---\nThe same door fact has independent legacy evidence.\n',encoding='utf-8')
    with PgVectorStore(TEST_DSN,dim=64,table=table,tenant=tenant) as legacy:
        stats = Indexer(legacy,HashingEmbedder(dim=64)).index_path(root)
        assert stats.files == 1 and legacy.count() > 0
        chunk = next(legacy.iter_chunks())
        assert chunk.metadata['recall_graph']['facts'][0]['object'] == DOOR.object
        digest = chunk.metadata['content_hash']
        card = replace(evidence_card(DOOR,tenant=tenant,generation=legacy.generation_id,chunk=chunk.id,source=str(chunk.metadata.get('file') or chunk.source)),card_id='',source_digest=digest)
        cards = PostgresEvidenceCardStore(TEST_DSN,tenant_id=tenant)
        cards.put((card,))
        def actual_digest(c):
            row = legacy.chunk_by_id(c.chunk_id)
            return None if row is None else row.metadata['content_hash']
        decision = ProvenanceController(tenant_id=tenant,generation_id=legacy.generation_id,cards=cards,ledger=_ledger(manager),source_digest_for=actual_digest,now=lambda:NOW,writer='gc-legacy-support-test').apply_fact(FactApplicationRequest(DOOR,(card.card_id,),'gc-real-legacy-support'))
        assert decision.code == DecisionCode.DUPLICATE, decision
        assert _gc(manager) == (first,)
        assert legacy.chunk_by_id(chunk.id) is not None, 'legacy support was actually removed'
        assert DOOR.object in _bare_current(manager), 'gc permanently retracted a fact with surviving indexed legacy support'
