"""Conservative structural relations for benchmark corpora.

These helpers only use identifiers that the benchmark supplies explicitly. They never infer a
relationship from free text, and they connect members of a group as an ordered chain rather than
materialising a quadratic clique. The returned shape is directly consumable as ``recall_graph``
frontmatter metadata.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

STRUCTURAL_GRAPH_VERSION = 1
STRUCTURAL_RELATION = "references"

ATM_EDGE_TYPES = frozenset(
    {
        "same_email_thread",
        "attachment_link",
        "media_reference",
        "shared_event",
    }
)
LOCOMO_EDGE_TYPES = frozenset(
    {
        "conversation_order",
        "speaker",
        "session_date",
        "reply_continuity",
        "explicit_entity_repetition",
    }
)

_ID_KEYS = ("id", "evidence_id", "media_id", "image_id", "video_id")
_PATH_KEYS = ("path", "file", "filename", "image_path", "video_path", "uri")
_ENTITY_KEYS = ("entities", "entity_ids", "mentioned_entities", "recall_entities")


def _text(value: Any) -> str:
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return str(value).strip()
    return ""


def _safe_key(value: Any) -> str:
    text = _text(value)
    if not text or "\n" in text or "\r" in text:
        return ""
    return text[:512]


def _record_id(record: Mapping[str, Any]) -> str:
    for key in _ID_KEYS:
        value = _safe_key(record.get(key))
        if value:
            return value
    return ""


def _values(record: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    for key in keys:
        value = record.get(key)
        raw_values = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else (value,)
        for raw in raw_values:
            if isinstance(raw, Mapping):
                nested = []
                for nested_key in (*_ID_KEYS, *_PATH_KEYS, "name"):
                    nested_value = _safe_key(raw.get(nested_key))
                    if nested_value:
                        nested.append(nested_value)
                raw_values_nested = nested
            else:
                raw_values_nested = [_safe_key(raw)]
            for item in raw_values_nested:
                if item and item not in values:
                    values.append(item)
    return tuple(values)


def _aliases(value: str) -> tuple[str, ...]:
    normalized = value.strip().replace("\\", "/")
    normalized = normalized.split("#", 1)[0].split("?", 1)[0]
    basename = PurePosixPath(normalized).name
    stem = basename.rsplit(".", 1)[0] if "." in basename else basename
    return tuple(dict.fromkeys(item.casefold() for item in (value, basename, stem) if item))


def _group_edges(
    groups: Mapping[tuple[str, str], Sequence[str]],
    *,
    edge_type: str,
) -> list[tuple[str, str, str, str]]:
    edges: list[tuple[str, str, str, str]] = []
    for (kind, key), members in sorted(groups.items()):
        ordered = list(dict.fromkeys(members))
        for subject, object_id in zip(ordered, ordered[1:]):
            if subject != object_id:
                edges.append((subject, object_id, edge_type, key))
    return edges


def _relation(subject: str, object_id: str, edge_type: str, key: str) -> dict[str, Any]:
    return {
        "relation": STRUCTURAL_RELATION,
        "subject": subject,
        "object": object_id,
        "confidence": 1.0,
        "structural_type": edge_type,
        "structural_key": key,
    }


def _metadata_for_edges(
    ids: Sequence[str], edges: Sequence[tuple[str, str, str, str]]
) -> dict[str, dict[str, Any]]:
    result = {
        item_id: {"schema_version": STRUCTURAL_GRAPH_VERSION, "relations": []}
        for item_id in ids
    }
    for subject, object_id, edge_type, key in edges:
        if subject not in result or object_id not in result:
            continue
        result[subject]["relations"].append(_relation(subject, object_id, edge_type, key))
    for value in result.values():
        value["relations"] = sorted(
            value["relations"],
            key=lambda item: (item["structural_type"], item["structural_key"], item["object"]),
        )
    return result


def atm_structural_metadata(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build safe structural metadata for ATM email, image, and video records.

    Thread and event links require explicit nonempty identifiers. Attachments and media links
    resolve through exact id, filename, or path stem aliases. An unresolved reference is omitted.
    """
    ids: list[str] = []
    modalities: dict[str, str] = {}
    aliases: dict[str, set[str]] = defaultdict(set)
    for record in records:
        item_id = _record_id(record)
        if not item_id or item_id in modalities:
            continue
        ids.append(item_id)
        modality = _safe_key(record.get("modality") or record.get("type")).casefold()
        modalities[item_id] = modality
        for value in _values(record, (*_ID_KEYS, *_PATH_KEYS)):
            for alias in _aliases(value):
                aliases[alias].add(item_id)

    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    direct_edges: list[tuple[str, str, str, str]] = []
    for record in records:
        subject = _record_id(record)
        if subject not in modalities:
            continue
        for key in ("thread_id", "thread", "threadId", "conversation_id", "conversationId"):
            for value in _values(record, (key,)):
                groups[("thread", value.casefold())].append(subject)
        for key in ("event_id", "event_ids", "event_identifier", "event_identifiers"):
            for value in _values(record, (key,)):
                groups[("event", value.casefold())].append(subject)
        for key in (
            "attachment_ids", "attachments", "attachment_refs", "attached_media",
            "image_ids", "video_ids", "media_ids", "image_refs", "video_refs", "media_refs",
            "references",
        ):
            for value in _values(record, (key,)):
                targets = aliases.get(value.casefold(), set())
                if len(targets) == 1:
                    target = next(iter(targets))
                    if target != subject:
                        direct_edges.append((subject, target, "attachment_link" if "attach" in key else "media_reference", value))
        for key in ("email_id", "email_ids", "source_email_id", "source_email_ids", "attached_to"):
            for value in _values(record, (key,)):
                targets = aliases.get(value.casefold(), set())
                if len(targets) == 1:
                    target = next(iter(targets))
                    if target != subject:
                        direct_edges.append((subject, target, "attachment_link", value))

    edges = _group_edges(
        {key: members for key, members in groups.items() if key[0] == "thread"},
        edge_type="same_email_thread",
    )
    edges.extend(
        _group_edges(
            {key: members for key, members in groups.items() if key[0] == "event"},
            edge_type="shared_event",
        )
    )
    edges.extend(direct_edges)
    unique = list(dict.fromkeys(edges))
    return _metadata_for_edges(ids, unique)


def locomo_structural_metadata(conversation: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Build safe structural metadata for one LoCoMo conversation.

    The walk follows numeric session order and source turn order. Entity repetition is enabled
    only for explicit entity fields if a dataset extension provides them. Names guessed from turn
    prose are intentionally excluded.
    """
    turns: list[tuple[str, Mapping[str, Any], str, str]] = []
    dia_to_file: dict[str, str] = {}
    for session_key in sorted(
        (key for key in conversation if isinstance(key, str) and key.startswith("session_") and key[8:].isdigit()),
        key=lambda key: int(key[8:]),
    ):
        date_value = _safe_key(conversation.get(f"{session_key}_date_time"))
        raw_turns = conversation.get(session_key)
        if not isinstance(raw_turns, Sequence) or isinstance(raw_turns, (str, bytes)):
            continue
        for turn in raw_turns:
            if not isinstance(turn, Mapping):
                continue
            turn_id = _safe_key(turn.get("dia_id"))
            if turn_id:
                file_id = turn_id.replace(":", "_") + ".md"
                dia_to_file[turn_id] = file_id
                turns.append((file_id, turn, session_key, date_value))

    ids = [turn_id for turn_id, _turn, _session, _date in turns]
    edges: list[tuple[str, str, str, str]] = []
    edges.extend(
        (left[0], right[0], "conversation_order", f"{left[2]}:{index}")
        for index, (left, right) in enumerate(zip(turns, turns[1:]))
        if left[0] != right[0]
    )

    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for turn_id, turn, session, date_value in turns:
        speaker = _safe_key(turn.get("speaker"))
        if speaker:
            grouped[("speaker", speaker.casefold())].append(turn_id)
        if date_value:
            grouped[("session_date", f"{session}:{date_value.casefold()}")].append(turn_id)
        for entity in _values(turn, _ENTITY_KEYS):
            grouped[("explicit_entity_repetition", entity.casefold())].append(turn_id)
        reply = _values(turn, ("reply_to", "reply_to_id", "parent_dia_id", "in_reply_to"))
        for target in reply:
            target_file = dia_to_file.get(target, target)
            if target_file in ids and target_file != turn_id:
                edges.append((target_file, turn_id, "reply_continuity", target))

    for edge_type, group_name in (
        ("speaker", "speaker"),
        ("session_date", "session_date"),
        ("explicit_entity_repetition", "explicit_entity_repetition"),
    ):
        edges.extend(_group_edges(
            {key: members for key, members in grouped.items() if key[0] == group_name},
            edge_type=edge_type,
        ))
    return _metadata_for_edges(ids, list(dict.fromkeys(edges)))


__all__ = [
    "ATM_EDGE_TYPES",
    "LOCOMO_EDGE_TYPES",
    "STRUCTURAL_GRAPH_VERSION",
    "atm_structural_metadata",
    "locomo_structural_metadata",
]
