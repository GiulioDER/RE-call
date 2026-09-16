"""Collect one graph precision evaluation arm through the VPS2 TTY MCP transport."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)|[@-_])")


def _command(
    tenant: str,
    embedder: str,
    index_root: str,
    profile: str,
    variant: str,
    control: str,
    control_seed: int,
    hub_threshold: int,
    cosine_margin: float,
    pinned_generation_id: str | None = None,
    benchmark_retrieval_leg_audit: bool = False,
    benchmark_document_expansion_audit: bool = False,
    benchmark_source_admission_audit: bool = False,
    benchmark_source_conditioning_reuse_audit: bool = False,
    source_conditioning_mode: str = "off",
    source_conditioning_artifact: str | None = None,
    source_conditioning_sample_rate: float = 0.0,
    source_conditioning_policy: str = "alpha008",
    atomic_rescue_mode: str = "off",
    atomic_rescue_artifact: str | None = None,
    atomic_rescue_artifact_root: str | None = None,
    atomic_rescue_sample_rate: float = 0.0,
    atomic_rescue_expected: str | None = None,
    benchmark_pin: bool = False,
    source_policy_file: str | None = None,
) -> list[str]:
    ssh = os.environ.get(
        "RECALL_SSH_EXECUTABLE",
        str(Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "OpenSSH" / "ssh.exe"),
    )
    ssh_config = str(Path.home() / ".ssh" / "config").replace("\\", "/")
    code_root = os.environ.get("RECALL_BENCHMARK_REMOTE_CODE_ROOT", "/home/sentiment/recall-repos")
    quoted_code_root = shlex.quote(code_root)
    pin = "RECALL_BENCHMARK_PIN=1 " if pinned_generation_id or benchmark_pin else ""
    if pinned_generation_id:
        pin += f"RECALL_PINNED_GENERATION_ID={pinned_generation_id} "
    leg_audit = "RECALL_BENCHMARK_RETRIEVAL_LEG_AUDIT=1 " if benchmark_retrieval_leg_audit else ""
    document_audit = (
        "RECALL_BENCHMARK_DOCUMENT_EXPANSION_AUDIT=1 " if benchmark_document_expansion_audit else ""
    )
    source_admission_audit = (
        "RECALL_BENCHMARK_SOURCE_ADMISSION_AUDIT=1 " if benchmark_source_admission_audit else ""
    )
    source_conditioning_reuse_audit = (
        "RECALL_BENCHMARK_SOURCE_CONDITIONING_REUSE_AUDIT=1 "
        if benchmark_source_conditioning_reuse_audit
        else ""
    )
    source_conditioning = ""
    if source_conditioning_mode != "off":
        source_conditioning = (
            f"RECALL_SOURCE_CONDITIONING_MODE={shlex.quote(source_conditioning_mode)} "
            "RECALL_SOURCE_CONDITIONING_SHADOW_SAMPLE_RATE="
            f"{source_conditioning_sample_rate:.6f} "
            "RECALL_SOURCE_CONDITIONING_SHADOW_POLICY="
            f"{shlex.quote(source_conditioning_policy)} "
        )
        if source_conditioning_artifact is not None:
            source_conditioning += (
                f"RECALL_SOURCE_CONDITIONING_ARTIFACT={shlex.quote(source_conditioning_artifact)} "
            )
    atomic_rescue = ""
    if atomic_rescue_mode != "off":
        atomic_rescue = (
            "OPENBLAS_NUM_THREADS=1 "
            f"RECALL_ATOMIC_RESCUE_MODE={shlex.quote(atomic_rescue_mode)} "
            "RECALL_ATOMIC_RESCUE_SHADOW_SAMPLE_RATE="
            f"{atomic_rescue_sample_rate:.6f} "
        )
        if atomic_rescue_artifact is not None:
            atomic_rescue += (
                f"RECALL_ATOMIC_RESCUE_ARTIFACT={shlex.quote(atomic_rescue_artifact)} "
            )
        if atomic_rescue_artifact_root is not None:
            atomic_rescue += (
                "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT="
                f"{shlex.quote(atomic_rescue_artifact_root)} "
            )
        if atomic_rescue_expected is not None:
            atomic_rescue += (
                "RECALL_BENCHMARK_ATOMIC_RESCUE_EXPECTED="
                f"{shlex.quote(atomic_rescue_expected)} "
            )
    source_policy = (
        f"RECALL_SOURCE_POLICY_FILE={shlex.quote(source_policy_file)} "
        if source_policy_file is not None
        else ""
    )
    remote = (
        "stty -echo; stty -onlcr -ocrnl 2>/dev/null || true; "
        "stty rows 1000 cols 10000 2>/dev/null || true; "
        f"cd {quoted_code_root} && set -a && . /home/sentiment/recall-repos/.env && set +a && "
        f"PYTHONPATH={quoted_code_root} "
        "RECALL_ENV=production RECALL_TRUST_MODE=production "
        f"RECALL_TENANT={tenant} RECALL_EMBEDDER={embedder} "
        f"RECALL_INDEX_ROOT={index_root} RECALL_RETRIEVAL_PROFILE={profile} "
        f"RECALL_GRAPH_PRECISION_VARIANT={variant} "
        f"RECALL_GRAPH_RELATION_CONTROL={control} "
        f"RECALL_GRAPH_RELATION_CONTROL_SEED={control_seed} "
        f"RECALL_GRAPH_HUB_DEGREE_THRESHOLD={hub_threshold} "
        f"RECALL_GRAPH_COSINE_MARGIN={cosine_margin:.2f} "
        + leg_audit
        + document_audit
        + source_admission_audit
        + source_conditioning_reuse_audit
        + source_conditioning
        + atomic_rescue
        + source_policy
        + pin
        + "exec /home/sentiment/recall-repos/.venv/bin/python -m recall_mcp.server"
    )
    return [ssh, "-tt", "-o", "BatchMode=yes", "-F", ssh_config, "vps2", remote]


class TTYMCP:
    def __init__(self, command: list[str], timeout: float) -> None:
        self.timeout = timeout
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.stdin = self.process.stdin
        self.events: queue.Queue[bytes | None] = queue.Queue()
        self.buffer = ""
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        try:
            while True:
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    break
                self.events.put(chunk)
        finally:
            self.events.put(None)

    @staticmethod
    def _clean(chunk: bytes) -> str:
        text = chunk.decode("utf-8", errors="replace")
        return ANSI_RE.sub("", text).replace("\r", "").replace("\n", "")

    def _response(self, request_id: int) -> dict[str, Any]:
        marker = '{"jsonrpc":"2.0","id":' + str(request_id)
        decoder = json.JSONDecoder()
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            start = self.buffer.find(marker)
            if start >= 0:
                try:
                    value, _ = decoder.raw_decode(self.buffer[start:])
                    if isinstance(value, dict):
                        return value
                except json.JSONDecodeError:
                    pass
            remaining = max(0.05, min(1.0, deadline - time.monotonic()))
            try:
                chunk = self.events.get(timeout=remaining)
            except queue.Empty:
                continue
            if chunk is None:
                break
            self.buffer += self._clean(chunk)
        raise TimeoutError(
            f"timed out waiting for MCP response id {request_id}; "
            f"process_returncode={self.process.poll()} buffer_tail={self.buffer[-400:]}"
        )

    def call(self, request_id: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        self.stdin.write((json.dumps(request, ensure_ascii=False) + "\r").encode("utf-8"))
        self.stdin.flush()
        return self._response(request_id)

    def batch(self, calls: list[tuple[int, str, dict[str, Any]]]) -> list[dict[str, Any]]:
        """Send calls together, then collect responses by id for concurrency probes."""
        for request_id, method, params in calls:
            request = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            self.stdin.write((json.dumps(request, ensure_ascii=False) + "\r").encode("utf-8"))
        self.stdin.flush()
        return [self._response(request_id) for request_id, _, _ in calls]

    def close(self) -> None:
        try:
            self.stdin.write(b"\x03")
            self.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


def _extract_payload(response: dict[str, Any]) -> str:
    if response.get("error"):
        raise RuntimeError(json.dumps(response["error"], ensure_ascii=False))
    content = response.get("result", {}).get("content", [])
    text_blocks = [item.get("text", "") for item in content if item.get("type") == "text"]
    if not text_blocks:
        raise RuntimeError("MCP response contained no text content")
    payload = "".join(text_blocks)
    json.loads(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query-set", default="docs/preregistrations/2026-08-17-memory-queries.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--embedder", default="voyage:voyage-4")
    parser.add_argument("--index-root", default="/home/sentiment/recall-repos/memory")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-graph-nodes", type=int, default=32)
    parser.add_argument("--max-evidence-tokens", type=int, default=2048)
    parser.add_argument(
        "--variant",
        choices=(
            "baseline",
            "directional",
            "corroboration",
            "hub",
            "cosine",
            "selective",
            "combined",
        ),
        required=True,
    )
    parser.add_argument("--control", choices=("none", "shuffled", "removed"), default="none")
    parser.add_argument("--control-seed", type=int, default=20260825)
    parser.add_argument("--hub-threshold", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--cosine-margin", type=float, choices=(0.05, 0.10, 0.15), default=0.10)
    args = parser.parse_args()

    queries = json.loads(Path(args.query_set).read_text(encoding="utf-8"))
    if args.limit:
        queries = queries[: args.limit]
    client = TTYMCP(
        _command(
            args.tenant,
            args.embedder,
            args.index_root,
            args.profile,
            args.variant,
            args.control,
            args.control_seed,
            args.hub_threshold,
            args.cosine_margin,
        ),
        args.timeout,
    )
    rows: list[dict[str, object]] = []
    try:
        client.call(
            1,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "recall-graph-precision", "version": "1.0"},
            },
        )
        client.stdin.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\r')
        client.stdin.flush()
        request_id = 2
        for arm in ("off", "one_hop"):
            for index, query in enumerate(queries, start=1):
                print(
                    f"{args.variant}/{args.control}/{arm} {index}/{len(queries)} {query['query']}",
                    flush=True,
                )
                response = client.call(
                    request_id,
                    "tools/call",
                    {
                        "name": "recall_reasoning_query",
                        "arguments": {
                            "query": query["query"],
                            "k": 5,
                            "mode": "evidence_assembly",
                            "max_steps": args.max_steps,
                            "max_graph_nodes": args.max_graph_nodes,
                            "max_evidence_tokens": args.max_evidence_tokens,
                            "graph_expansion": arm,
                        },
                    },
                )
                rows.append(
                    {
                        "query": query,
                        "arm": arm,
                        "variant": args.variant,
                        "relation_control": args.control,
                        "relation_control_seed": args.control_seed,
                        "hub_threshold": args.hub_threshold,
                        "cosine_margin": args.cosine_margin,
                        "payload": _extract_payload(response),
                    }
                )
                request_id += 1
    finally:
        client.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(output)}))


if __name__ == "__main__":
    main()
