from __future__ import annotations

import json

from benchmarks.structural_edges import atm_structural_metadata, locomo_structural_metadata
from recall.eval.locomo import _turn_document, write_conversation_corpus
from recall.frontmatter import parse_frontmatter
from recall.semantic_graph import build_semantic_graph
from recall.types import Chunk


def test_atm_edges_use_only_explicit_identifiers_and_resolve_media() -> None:
    """The invariant is explicit links only, and the failure mode is guessed cross-modal edges.

    This test is intended to go red if the producer starts matching free text or emits an
    unresolved attachment. The baseline mutation is to replace the target cardinality check with
    an unconditional ``next(iter(targets), "")`` in ``atm_structural_metadata``.
    Node ``test_atm_edges_use_only_explicit_identifiers_and_resolve_media`` went red during the
    implementation mutation that omitted modality specific IDs, because the attachment edge was
    absent at the intended assertion.
    """
    records = [
        {"id": "email-1", "modality": "email", "thread_id": "thread-a", "attachments": ["img-1"]},
        {"id": "email-2", "modality": "email", "thread_id": "thread-a", "detail": "img-1"},
        {"image_id": "img-1", "modality": "image", "image_path": "img-1.jpg", "event_id": "event-a"},
        {"image_id": "img-2", "modality": "image", "image_path": "img-2.jpg", "event_id": "event-a"},
    ]

    metadata = atm_structural_metadata(records)
    edges = [edge for value in metadata.values() for edge in value["relations"]]

    assert {(edge["structural_type"], edge["object"]) for edge in edges} >= {
        ("same_email_thread", "email-2"),
        ("attachment_link", "img-1"),
    }
    assert not any(edge["object"] == "img-1" and edge["structural_type"] == "media_reference" for edge in edges)
    assert all(edge["relation"] == "references" for edge in edges)


def test_atm_preparation_attaches_graph_metadata_to_index_items(tmp_path) -> None:
    from benchmarks.atm_structural import build_memory_items_with_structural_edges

    email = tmp_path / "emails.json"
    image = tmp_path / "images.json"
    video = tmp_path / "videos.json"
    email.write_text(
        json.dumps([{"id": "email-1", "attachments": ["img-1"]}]), encoding="utf-8"
    )
    image.write_text(
        json.dumps([{"image_path": "img-1.jpg", "image_id": "img-1"}]), encoding="utf-8"
    )
    video.write_text("[]", encoding="utf-8")

    items = build_memory_items_with_structural_edges(image, video, email)
    email_item = next(item for item in items if item[0] == "email-1")
    assert email_item[3]["recall_graph"]["relations"][0]["object"] == "img-1"


def test_locomo_edges_are_ordered_and_entity_repetition_is_explicit_only() -> None:
    """Order, speaker, date, reply, and entity edges must come from structured fields.

    The failure mode is a relation added because two arbitrary words happen to repeat in turn
    text. The baseline mutation is to feed ``turn["text"].split()`` to the entity group in
    ``locomo_structural_metadata``; this test must fail on the extra relation.
    Node ``test_locomo_edges_are_ordered_and_entity_repetition_is_explicit_only`` was run against
    that mutation and failed with two explicit entity edges instead of one.
    """
    conversation = {
        "session_1_date_time": "1 January 2024",
        "session_1": [
            {"dia_id": "D1:1", "speaker": "Alice", "text": "blue blue", "entities": ["Project A"]},
            {"dia_id": "D1:2", "speaker": "Bob", "text": "blue", "reply_to": "D1:1"},
            {"dia_id": "D1:3", "speaker": "Alice", "text": "blue", "entities": ["Project A"]},
        ],
    }

    metadata = locomo_structural_metadata(conversation)
    edges = [edge for value in metadata.values() for edge in value["relations"]]
    kinds = {edge["structural_type"] for edge in edges}

    assert kinds == {
        "conversation_order",
        "speaker",
        "session_date",
        "reply_continuity",
        "explicit_entity_repetition",
    }
    assert all(edge["relation"] == "references" for edge in edges)
    assert sum(edge["structural_type"] == "explicit_entity_repetition" for edge in edges) == 1


def test_locomo_corpus_writes_graph_metadata_that_projects_as_authored_relations(tmp_path) -> None:
    """The graph metadata must survive corpus materialisation and the semantic graph projection.

    The failure mode is a helper that returns edges but is never reached by the real corpus writer.
    The baseline mutation is to remove ``structural_metadata=`` from ``write_conversation_corpus``.
    Node ``test_locomo_corpus_writes_graph_metadata_that_projects_as_authored_relations`` was run
    against that mutation and failed with a missing ``recall_graph`` key.
    """
    conversation = {
        "session_1_date_time": "1 January 2024",
        "session_1": [
            {"dia_id": "D1:1", "speaker": "Alice", "text": "first"},
            {"dia_id": "D1:2", "speaker": "Bob", "text": "second"},
        ],
    }
    assert write_conversation_corpus(conversation, tmp_path) == 2
    metadata, body = parse_frontmatter((tmp_path / "D1_1.md").read_text(encoding="utf-8"))
    assert body.startswith("# Alice")
    assert metadata["recall_graph"]["relations"][0]["object"] == "D1_2.md"

    chunks = [
        Chunk("c1", "D1_1.md", "first", {"file": "D1_1.md", **metadata}),
        Chunk(
            "c2",
            "D1_2.md",
            "second",
            {"file": "D1_2.md", **parse_frontmatter((tmp_path / "D1_2.md").read_text(encoding="utf-8"))[0]},
        ),
    ]
    graph = build_semantic_graph(chunks, tenant_id="t", generation_id="g")
    assert len(graph.relations) >= 1
    assert graph.relations[0].status == "authored"
    assert {relation.metadata["structural_type"] for relation in graph.relations} >= {
        "conversation_order",
        "session_date",
    }


def test_turn_document_keeps_graph_json_on_one_frontmatter_line() -> None:
    graph = {"schema_version": 1, "relations": [{"relation": "references", "object": "D1:2"}]}
    document = _turn_document(
        {"speaker": "Alice", "text": "hello", "_recall_structural_metadata": graph},
        "unknown date",
    )
    metadata, body = parse_frontmatter(document)
    assert json.dumps(metadata["recall_graph"], sort_keys=True, separators=(",", ":")) == json.dumps(graph, sort_keys=True, separators=(",", ":"))
    assert body.startswith("# Alice")
