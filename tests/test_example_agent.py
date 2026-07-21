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


@requires_db
def test_agent_cites_the_successor_not_the_superseded_memory(make_store):
    store = make_store(3)
    proposal = "raise the rate limit"
    store.upsert(
        [
            Chunk("old", "rate_v1.md", "rate limit is one hundred per second",
                  metadata={"file": "rate_v1.md", "ord": 0}),
            Chunk("new", "rate_v2.md", "rate limit is twenty per second",
                  metadata={"file": "rate_v2.md", "ord": 0, "supersedes": "rate_v1.md"}),
        ],
        [[1.0, 0.0, 0.0], [0.9, 0.44, 0.0]],
    )
    emb = DictEmbedder({proposal: [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 1.0])
    d = decide(store, emb, proposal)
    assert d["proceed"] is False
    assert "rate_v2.md" in d["reason"]      # the current version
    assert "rate_v1.md" not in d["reason"]  # never cite the superseded one as the blocker


@requires_db
def test_agent_does_not_treat_an_untrustworthy_hit_as_a_blocker(make_store):
    # the only match is expired: there is no trustworthy prior record, so the agent must not
    # block on it — and must say the memory abstained rather than quote the stale text
    store = make_store(3)
    proposal = "run the winter deploy freeze again"
    store.upsert(
        [Chunk("f", "freeze.md", "temporary deploy freeze for the winter release",
               metadata={"file": "freeze.md", "ord": 0, "valid_until": "2020-01-01"})],
        [[1.0, 0.0, 0.0]],
    )
    emb = DictEmbedder({proposal: [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 1.0])
    d = decide(store, emb, proposal)
    assert d["proceed"] is True
    assert "winter release" not in d["reason"]
