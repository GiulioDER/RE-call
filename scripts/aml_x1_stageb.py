"""Stage B of the X-1 pre-registration: ingest each drawn source into C9 and store every Search.

Pre-registration: docs/preregistrations/2026-09-25-aml-c9-source-coverage-baseline.md (Stage B,
amendment 1). Requests are the ones Stage A built and dry-ran (``scripts/aml_x1_sources.py``); the
runner recomputes each source's Add digest and refuses a source whose digest differs from the
committed dry run, so what is sent is what was checked.

``probe``  Before any source is ingested, sends the first Adds of each text source under a probe
           user, reads the compile token usage the service logs (``compiler_provider_usage``),
           deletes the probe users, and projects each source's compile cost at list price. The
           halving rule (a source over a quarter of the USD 25 cap is halved) is applied from this.
``run``    Per tenant, in parallel across tenants and in order within one: every Add, then every
           question's Search at top_k 100, then delete and verify an empty Search. One JSONL line
           per finished tenant; a restart skips finished tenants and re-ingests the rest from a
           clean delete. Image parts are stored as a digest, not bytes; ``images.json`` maps each
           digest to its materialized file so Stage C can restore them.

    python scripts/aml_x1_stageb.py probe --data-dir DATA --draw draw.json --serve-log serve-B.log
    python scripts/aml_x1_stageb.py run --data-dir DATA --draw draw.json --dryrun dryrun.json \\
        --serve-log serve-B.log --out-dir OUT --sources longmemeval_s ... --workers 4
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import aml_x1_sources as x1  # noqa: E402

#: OpenRouter list price for deepseek/deepseek-v4.1-flash, read 2026-09-26 from /api/v1/models.
PRICE_PROMPT = 0.30 / 1_000_000
PRICE_COMPLETION = 1.20 / 1_000_000
CAP_USD = 25.0
QUARTER = CAP_USD / 4
TEXT_SOURCES = ("longmemeval_s", "clbench", "personamem_v2", "memlens_32k")
RETRYABLE = {408, 429, 500, 502, 503, 504, 520, 522, 524}
_USAGE = re.compile(r"compiler_provider_usage (\{.*\})")


@dataclass
class Reply:
    status: int
    body: Any
    ms: float


def call(port: int, path: str, payload: dict[str, Any], token: str, attempts: int = 6) -> Reply:
    data = json.dumps(payload, separators=(",", ":")).encode()
    last = Reply(0, {"error": "not sent"}, 0.0)
    for attempt in range(1, attempts + 1):
        request = Request(f"http://127.0.0.1:{port}{path}", data=data, method="POST",
                          headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=900) as response:  # noqa: S310, localhost only
                last = Reply(response.status, json.loads(response.read()), (time.perf_counter() - started) * 1e3)
        except HTTPError as exc:
            raw = exc.read()
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = {"error": raw[:300].decode("utf-8", "replace")}
            last = Reply(exc.code, body, (time.perf_counter() - started) * 1e3)
        except (URLError, TimeoutError, OSError) as exc:
            last = Reply(599, {"error": repr(exc)}, (time.perf_counter() - started) * 1e3)
        if last.status not in RETRYABLE and last.status != 599:
            return last
        time.sleep(min(30.0, 2.0**attempt))
    return last


def usage_since(log: Path, offset: int) -> tuple[dict[str, int], int]:
    """Compile token totals logged after byte ``offset``, and the new offset."""
    totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    with log.open("rb") as handle:
        handle.seek(offset)
        chunk = handle.read()
    for line in chunk.decode("utf-8", "replace").splitlines():
        match = _USAGE.search(line)
        if match:
            record = json.loads(match.group(1))
            totals["calls"] += 1
            totals["prompt_tokens"] += int(record.get("prompt_tokens") or 0)
            totals["completion_tokens"] += int(record.get("completion_tokens") or 0)
    return totals, offset + len(chunk)


def usd(totals: dict[str, int]) -> float:
    return totals["prompt_tokens"] * PRICE_PROMPT + totals["completion_tokens"] * PRICE_COMPLETION


def text_chars(add: dict[str, Any]) -> int:
    return sum(x1._text_len(m["content"]) for m in add["messages"])


def image_digest(part: dict[str, Any]) -> str:
    return hashlib.sha256(str(part["image_url"]["url"]).encode()).hexdigest()


def compact(item: dict[str, Any]) -> dict[str, Any]:
    content = item.get("content")
    if isinstance(content, list):
        content = [
            {"type": "image_ref", "sha256": image_digest(part)} if part.get("type") == "image_url" else part
            for part in content
        ]
    return {key: item.get(key) for key in ("id", "session_id", "created_at", "score", "kind", "source")} | {"content": content}


def source_digest(source: str, data_dir: Path, draw: dict[str, Any], run_id: str) -> str:
    stats = x1.SourceStats()
    for tenant in x1.tenants_for(source, data_dir, draw, run_id):
        for add, _ in x1.build_adds(tenant, stats):
            stats.digest.update(hashlib.sha256(x1.canonical(add)).digest())
    return stats.digest.hexdigest()


def token_from_env(serve_env: Path) -> str:
    for line in serve_env.read_text(encoding="utf-8").splitlines():
        if line.startswith("RECALL_AML_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("RECALL_AML_API_KEY not found in the serve env file")


def probe(args: argparse.Namespace) -> None:
    token = token_from_env(args.serve_env)
    draw = json.loads(args.draw.read_text(encoding="utf-8"))
    dry = json.loads(args.dryrun.read_text(encoding="utf-8"))["sources"]
    report: dict[str, Any] = {"price_per_token": {"prompt": PRICE_PROMPT, "completion": PRICE_COMPLETION},
                              "cap_usd": CAP_USD, "quarter_usd": QUARTER, "sources": {}}
    for source in TEXT_SOURCES:
        stats = x1.SourceStats()
        sent = chars = failures = 0
        offset = args.serve_log.stat().st_size
        started = time.perf_counter()
        users: set[str] = set()
        for tenant in x1.tenants_for(source, args.data_dir, draw, "x1probe"):
            users.add(tenant.user_id)
            for add, _ in x1.build_adds(tenant, stats):
                if sum(1 for m in add["messages"] if isinstance(m["content"], list)):
                    continue  # the compile runs on text-only Adds; image Adds cost nothing here
                reply = call(args.port, "/v1/add", add, token)
                failures += reply.status != 200
                sent += 1
                chars += text_chars(add)
                if sent >= args.adds:
                    break
            if sent >= args.adds:
                break
        elapsed = time.perf_counter() - started
        time.sleep(2)
        totals, _ = usage_since(args.serve_log, offset)
        for user in users:
            call(args.port, "/v1/delete", {"user_id": user}, token)
        per_char = usd(totals) / chars if chars else 0.0
        # The dry run records every Add's text, image Adds included, which are not compiled; for a
        # source with image Adds (MemLens) the projection is therefore an upper bound.
        projected = per_char * dry[source]["total_text_chars"]
        report["sources"][source] = {
            "probe_adds": sent, "probe_failures": failures, "probe_text_chars": chars, **totals,
            "probe_usd": usd(totals), "usd_per_add": usd(totals) / sent if sent else None,
            "usd_per_million_chars": per_char * 1e6, "seconds_per_add": elapsed / sent if sent else None,
            "source_text_chars": dry[source].get("total_text_chars"),
            "projected_source_usd": projected, "over_quarter_of_cap": projected > QUARTER,
        }
        print(source, json.dumps(report["sources"][source]), flush=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    token = token_from_env(args.serve_env)
    draw = json.loads(args.draw.read_text(encoding="utf-8"))
    dry = json.loads(args.dryrun.read_text(encoding="utf-8"))["sources"]
    keep = json.loads(args.keep.read_text(encoding="utf-8")) if args.keep else {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    for source in args.sources:
        expected = dry[source]["adds_sha256"]
        observed = source_digest(source, args.data_dir, draw, "x1")
        if observed != expected:
            raise SystemExit(f"{source}: Add digest {observed[:16]} differs from the dry run's {expected[:16]}")
        out = args.out_dir / f"{source}.jsonl"
        done = set()
        if out.exists():
            done = {json.loads(line)["tenant"] for line in out.read_text(encoding="utf-8").splitlines()
                    if json.loads(line).get("status") == "ok"}
        tenants = [t for t in x1.tenants_for(source, args.data_dir, draw, "x1") if t.key not in done]
        if source in keep:
            allowed = set(keep[source])
            tenants = [t for t in tenants if t.key in allowed]
        offset = args.serve_log.stat().st_size
        lock = threading.Lock()
        images: dict[str, str] = {}
        counter = {"tenants": 0, "adds": 0}
        print(json.dumps({"source": source, "digest": "matches dry run", "pending_tenants": len(tenants),
                          "done_tenants": len(done)}), flush=True)

        def one(tenant: x1.Tenant) -> None:
            if stop.is_set():
                return
            call(args.port, "/v1/delete", {"user_id": tenant.user_id}, token)
            stats = x1.SourceStats()
            adds, sessions_ok, failures = [], set(), []
            for add, facts in x1.build_adds(tenant, stats):
                reply = call(args.port, "/v1/add", add, token)
                ok = reply.status == 200
                adds.append({"request_id": add["request_id"], "session_id": facts["session_id"], "status": reply.status,
                             "ms": round(reply.ms), "compiled": reply.body.get("compiled_count") if ok else None,
                             "fallback": reply.body.get("compiler_fallback") if ok else None})
                if ok:
                    sessions_ok.add(facts["session_id"])
                    with lock:
                        for message, name in zip(
                            [p for m in add["messages"] if isinstance(m["content"], list) for p in m["content"]
                             if p.get("type") == "image_url"], facts["image_names"], strict=False):
                            images[image_digest(message)] = name
                else:
                    failures.append({"request_id": add["request_id"], "status": reply.status, "body": str(reply.body)[:300]})
            searches = []
            for question in tenant.questions:
                reply = call(args.port, "/v1/search", question["search"], token)
                items = reply.body.get("data", []) if reply.status == 200 else []
                evidence = (question.get("evidence") or {}).get("sessions") or []
                searches.append({"question_id": question["question_id"], "category": question["category"],
                                 "status": reply.status, "ms": round(reply.ms),
                                 "evidence_present": [s for s in evidence if s in sessions_ok],
                                 "evidence_labelled": evidence, "items": [compact(i) for i in items]})
            deleted = call(args.port, "/v1/delete", {"user_id": tenant.user_id}, token)
            after = call(args.port, "/v1/search", {"query": "cleanup verification", "user_id": tenant.user_id, "top_k": 1}, token)
            clean = deleted.status == 200 and after.status == 200 and not after.body.get("data")
            status = "ok" if not failures and all(s["status"] == 200 for s in searches) and clean else "failed"
            row = {"tenant": tenant.key, "user_id": tenant.user_id, "status": status, "adds": adds,
                   "add_failures": failures, "searches": searches, "deleted_and_empty": clean}
            with lock:
                with out.open("a", encoding="utf-8") as sink:
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                counter["tenants"] += 1
                counter["adds"] += len(adds)
                if counter["tenants"] % args.report_every == 0 or status != "ok":
                    totals, _ = usage_since(args.serve_log, offset)
                    print(json.dumps({"source": source, "tenants": counter["tenants"], "adds": counter["adds"],
                                      "last_status": status, "compile_usd": round(usd(totals), 4)}), flush=True)
                    if usd(totals) > args.max_usd:
                        stop.set()
                        print(f"STOP: {source} compile spend over {args.max_usd} USD", flush=True)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(one, tenants))
        totals, _ = usage_since(args.serve_log, offset)
        image_file = args.out_dir / f"{source}.images.json"
        previous = json.loads(image_file.read_text(encoding="utf-8")) if image_file.exists() else {}
        image_file.write_text(json.dumps(previous | images, indent=0), encoding="utf-8")
        summary = {"source": source, "tenants_this_run": counter["tenants"], "adds_this_run": counter["adds"],
                   **totals, "compile_usd_list_price": round(usd(totals), 4), "stopped": stop.is_set()}
        with (args.out_dir / "spend.jsonl").open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
        if stop.is_set():
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("probe", probe), ("run", run)):
        p = sub.add_parser(name)
        p.add_argument("--data-dir", type=Path, required=True)
        p.add_argument("--draw", type=Path, required=True)
        p.add_argument("--dryrun", type=Path, required=True)
        p.add_argument("--serve-log", type=Path, required=True)
        p.add_argument("--serve-env", type=Path, default=Path(os.path.expanduser("~/mm1-mm3/serve/base.env")))
        p.add_argument("--port", type=int, default=18031)
        p.set_defaults(handler=handler)
        if name == "probe":
            p.add_argument("--adds", type=int, default=30)
            p.add_argument("--out", type=Path, required=True)
        else:
            p.add_argument("--out-dir", type=Path, required=True)
            p.add_argument("--sources", nargs="+", default=list(x1.SOURCES), choices=x1.SOURCES)
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--max-usd", type=float, default=20.0)
            p.add_argument("--report-every", type=int, default=10)
            p.add_argument("--keep", type=Path, default=None, help="JSON {source: [tenant keys]} after halving")
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
