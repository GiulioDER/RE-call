from examples.self_recall_agent import decide
from recall.types import Chunk

from tests.conftest import requires_db


class DictEmbedder:
    dim = 3
    name = "dict"

    def __init__(self, mapping, default):
        self._mapping, self._default = mapping, default

    def embed(self, texts):
        return [self._mapping.get(t, self._default) for t in texts]


@requires_db
def test_agent_backs_off_on_known_closed_decision(make_store):
    store = make_store(3)
    proposal = "inject retrieved context into the prompt"
    store.upsert(
        [Chunk("h", "hypotheses.md", "prompt injection of retrieved context was falsified closed")],
        [[1.0, 0.0, 0.0]],
    )
    # The proposal embeds onto the stored memory's vector -> strong match, no gap -> back off.
    emb = DictEmbedder({proposal: [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 1.0])
    d = decide(store, emb, proposal)
    assert d["proceed"] is False
    assert "memory" in d["reason"].lower()


@requires_db
def test_agent_proceeds_when_memory_has_no_match(make_store):
    store = make_store(3)
    store.upsert([Chunk("x", "notes.md", "unrelated note about deployment")], [[1.0, 0.0, 0.0]])
    # The proposal is orthogonal to everything stored -> gap_warning -> safe to proceed.
    emb = DictEmbedder({}, default=[0.0, 0.0, 1.0])
    d = decide(store, emb, "should we add a brand new telemetry dashboard")
    assert d["proceed"] is True
