"""Paired agent benchmark primitives for RE-call enabled and disabled runs."""

from .runner import run_paired
from .schema import RECALL_OFF, RECALL_ON, SessionRecord
from .summarize import summarize_pairs
from .io import read_jsonl, write_jsonl
from .codex_exec import CodexExecConfig, make_codex_runner, parse_codex_jsonl, run_codex_case
from .claude_exec import (
    ClaudeExecConfig,
    make_claude_runner,
    parse_claude_stream_json,
    run_claude_case,
)
from .gate import AdmissionReport, AdmissionVerdict, admit_pairs, check_session

__all__ = [
    "RECALL_OFF",
    "RECALL_ON",
    "AdmissionReport",
    "AdmissionVerdict",
    "ClaudeExecConfig",
    "CodexExecConfig",
    "SessionRecord",
    "admit_pairs",
    "check_session",
    "make_claude_runner",
    "make_codex_runner",
    "parse_claude_stream_json",
    "parse_codex_jsonl",
    "read_jsonl",
    "run_claude_case",
    "run_codex_case",
    "run_paired",
    "summarize_pairs",
    "write_jsonl",
]
