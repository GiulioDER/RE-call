"""The admission gate: proof that the treatment was applied, before anything is measured.

This module exists because of a specific, reproduced failure. On 2026-08-20, a `recall_on` session
was launched with `--mcp-config` pointing at a RE-call server that needed 13.3 s to load its
embedder. The installed CLI was 2.1.220, one version below the release that waits for a pending
MCP server before the first turn. The session therefore ran with **no RE-call tool at all**, the
model answered from its own knowledge, and the run reported:

    "subtype": "success", "is_error": false, "permission_denials": [], "num_turns": 1

Nothing in the exit status, the token counts, the wall time or the response distinguishes that
from a genuine on-arm result. It is a null result manufactured by a wiring fault, and averaged
into a summary it does not look like an outlier: it looks like evidence that RE-call does nothing.
An earlier attempt with a different agent lost a run this way.

So the rule is: **a pair that cannot prove the treatment was applied is discarded, not recorded.**
Discarding is not the same as scoring zero, and the discarded count is published beside the
result, because a gate that quietly drops half the run is its own kind of lie.

## Available is the gate; used is a finding

The gate requires the RE-call tools to be **present in the session's tool list**. It does not
require the agent to have called them. Those are different facts:

- Tool absent  -> wiring failure. The experiment did not happen. Discard.
- Tool present, never called -> a real behavioural result. The agent had the memory layer and
  chose not to reach for it. That is a finding about discoverability, and averaging it in is
  correct.

Conflating the two is exactly the mistake that made the Codex run look like a null.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, asdict
from typing import Any

from .claude_exec import DEFAULT_RECALL_TOOL_PREFIX
from .schema import RECALL_OFF, RECALL_ON, SessionRecord

#: MCP server statuses that mean the server is usable. Anything else, including the `pending`
#: that produced the failure above, is treated as absent.
CONNECTED_STATUSES = frozenset({"connected", "ready"})


@dataclass(frozen=True)
class AdmissionVerdict:
    """Why one session is or is not admissible evidence."""

    task_id: str
    variant: str
    admitted: bool
    reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _recall_tools(record: SessionRecord, prefix: str) -> list[str]:
    available = record.metadata.get("recall_tools_available")
    if isinstance(available, Sequence) and not isinstance(available, (str, bytes)):
        return [str(name) for name in available]
    # Fall back to the raw tool list, so a record written by a different adapter is still gateable.
    tools = record.metadata.get("session_tools")
    if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes)):
        return [str(name) for name in tools if str(name).startswith(prefix)]
    return []


def check_session(
    record: SessionRecord, *, recall_tool_prefix: str = DEFAULT_RECALL_TOOL_PREFIX
) -> AdmissionVerdict:
    """Decide whether one session is admissible evidence for its arm."""

    reasons: list[str] = []
    notes: list[str] = []

    if not record.metadata.get("init_present", True):
        reasons.append(
            "no system/init event: the session's real tool surface was never observed, so "
            "neither arm can be verified"
        )

    tools = _recall_tools(record, recall_tool_prefix)
    servers = record.metadata.get("mcp_servers")
    servers = servers if isinstance(servers, Sequence) and not isinstance(servers, str) else []
    server_errors = record.metadata.get("mcp_server_errors")
    server_errors = (
        server_errors
        if isinstance(server_errors, Sequence) and not isinstance(server_errors, str)
        else []
    )

    if record.variant == RECALL_ON:
        if not tools:
            unhealthy = [
                f"{item.get('name')}={item.get('status')}"
                for item in servers
                if isinstance(item, Mapping)
                and str(item.get("status", "")).lower() not in CONNECTED_STATUSES
            ]
            detail = f"; servers reported {unhealthy}" if unhealthy else ""
            reasons.append(
                f"the on arm had no tool matching {recall_tool_prefix!r} in its session tool "
                f"list, so RE-call was never available to it{detail}"
            )
        if server_errors:
            reasons.append(f"MCP servers were skipped at startup: {list(server_errors)}")
        for item in servers:
            if not isinstance(item, Mapping):
                continue
            status = str(item.get("status", "")).lower()
            if status not in CONNECTED_STATUSES:
                reasons.append(
                    f"MCP server {item.get('name')!r} was {status!r}, not connected"
                )
        if not reasons and record.recall_call_count == 0:
            # Deliberately a note, not a reason. See the module docstring.
            notes.append(
                "RE-call was available and never called: a behavioural result, not a wiring "
                "failure, and it is counted in the summary"
            )
    elif record.variant == RECALL_OFF:
        if tools:
            reasons.append(
                f"the off arm had RE-call tools available ({tools}), so the arms differ by "
                f"less than the experiment claims"
            )
        if record.recall_call_count:
            reasons.append(
                f"the off arm recorded {record.recall_call_count} RE-call call(s)"
            )
    else:  # pragma: no cover - SessionRecord already rejects unknown variants
        reasons.append(f"unknown variant {record.variant!r}")

    if record.error is not None:
        reasons.append(f"the session did not complete: {record.error}")

    return AdmissionVerdict(
        task_id=record.task_id,
        variant=record.variant,
        admitted=not reasons,
        reasons=tuple(reasons),
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class AdmissionReport:
    """The admitted records, and a full account of everything dropped."""

    admitted: tuple[SessionRecord, ...]
    verdicts: tuple[AdmissionVerdict, ...]
    discarded_task_ids: tuple[str, ...]

    @property
    def discarded_pair_count(self) -> int:
        return len(self.discarded_task_ids)

    def summary(self) -> dict[str, Any]:
        """A JSON-safe account, for the run artifact."""

        return {
            "admitted_pairs": len(self.admitted) // 2,
            "discarded_pairs": self.discarded_pair_count,
            "discarded_task_ids": list(self.discarded_task_ids),
            "verdicts": [verdict.to_dict() for verdict in self.verdicts],
        }


def admit_pairs(
    records: Iterable[SessionRecord],
    *,
    recall_tool_prefix: str = DEFAULT_RECALL_TOOL_PREFIX,
) -> AdmissionReport:
    """Keep only the pairs where both arms are admissible evidence.

    A pair is the unit, not a session: an on arm whose treatment was verified is still useless
    without the off arm it is being compared against, and admitting it alone would quietly turn a
    paired design into an unpaired one partway through the task set.
    """

    by_task: dict[str, dict[str, SessionRecord]] = defaultdict(dict)
    ordered_tasks: list[str] = []
    for record in records:
        if record.task_id not in by_task:
            ordered_tasks.append(record.task_id)
        by_task[record.task_id][record.variant] = record

    admitted: list[SessionRecord] = []
    verdicts: list[AdmissionVerdict] = []
    discarded: list[str] = []
    for task_id in ordered_tasks:
        arms = by_task[task_id]
        pair_verdicts = []
        for variant in (RECALL_ON, RECALL_OFF):
            arm = arms.get(variant)
            if arm is None:
                pair_verdicts.append(
                    AdmissionVerdict(
                        task_id=task_id,
                        variant=variant,
                        admitted=False,
                        reasons=(f"no {variant} record for this task",),
                    )
                )
                continue
            pair_verdicts.append(
                check_session(arm, recall_tool_prefix=recall_tool_prefix)
            )
        verdicts.extend(pair_verdicts)
        if all(verdict.admitted for verdict in pair_verdicts):
            admitted.extend(arms[variant] for variant in (RECALL_ON, RECALL_OFF))
        else:
            discarded.append(task_id)

    return AdmissionReport(
        admitted=tuple(admitted),
        verdicts=tuple(verdicts),
        discarded_task_ids=tuple(discarded),
    )
