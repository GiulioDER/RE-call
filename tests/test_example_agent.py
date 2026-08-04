from examples.self_recall_agent import decide
from recall.types import Chunk

from recall.calibration import Calibration
from recall.guards import DEFAULT_GAP_THRESHOLD
from recall.trust_policy import TrustPolicy
from tests.conftest import requires_db

#: These stores have no published calibration, so the example's production default (strict)
#: would refuse. The example's refusal branch is covered by its own test; here we exercise the
#: decision logic behind a working gate.
_DEV = {"policy": TrustPolicy.development(),
        "calibration": Calibration(embedder="test-development",
                                   threshold=DEFAULT_GAP_THRESHOLD)}


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
    d = decide(store, emb, proposal, **_DEV)
    assert d["proceed"] is False
    assert "memory" in d["reason"].lower()


@requires_db
def test_agent_proceeds_when_memory_has_no_match(make_store):
    store = make_store(3)
    store.upsert([Chunk("x", "notes.md", "unrelated note about deployment")], [[1.0, 0.0, 0.0]])
    # The proposal is orthogonal to everything stored -> gap_warning -> safe to proceed.
    emb = DictEmbedder({}, default=[0.0, 0.0, 1.0])
    d = decide(store, emb, "should we add a brand new telemetry dashboard", **_DEV)
    assert d["proceed"] is True


@requires_db
def test_agent_refuses_to_proceed_when_the_trust_system_is_unavailable(make_store):
    """The branch that matters: an outage must NOT read as 'no prior memory'.

    Without it the example fails OPEN — the guard goes down, the search yields nothing, and the
    agent proceeds to re-litigate a decision that was in fact closed. Strict mode is the default,
    and these stores have no published calibration, so no injection is needed to reach it.
    """
    store = make_store(3)
    store.upsert([Chunk("x", "notes.md", "we already decided against this")], [[1.0, 0.0, 0.0]])
    emb = DictEmbedder({}, default=[0.0, 0.0, 1.0])

    d = decide(store, emb, "should we add a brand new telemetry dashboard")

    assert d["proceed"] is False, "an unavailable guard must never read as permission to proceed"
    assert d["failure_code"] in {
        "INDEX_NOT_READY", "CALIBRATION_MISSING", "CALIBRATION_UNCERTIFIED",
        "CALIBRATION_STALE", "LINEAGE_MISMATCH", "DEPENDENCY_UNAVAILABLE",
    }
    # And it must be distinguishable from the working-gate-found-nothing answer.
    assert "no relevant prior memory" not in d["reason"].lower()
