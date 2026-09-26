"""X-1 amendment 3/4: MM-1 and MM-3 held-out arms on MemLens 32K and MobileMem-Omni EN.

Reads X-1 Stage B's stored searches (``out/<source>.jsonl``): arm **B** is the search's ``items``
(C9-raw, scope ``route``), **B2** is B again (the noise floor), **D** and **Dt** are the
``arms`` Stage B recorded through the dual-scope services (amendment 3). Every arm is answered by
DeepSeek V4.1 Flash through one pinned provider with reasoning off, under each source's own
evaluator, fetched at a pinned commit into ``--eval-root`` (amendment 4):

* **MemLens** (``github.com/xrenaf/MEMLENS`` at ``77f3ab9a``): the memory-agent answer prompt
  ``build_mem0_answer_messages`` (``memory-agent/prompt_builders.py``), then the extract-then-match
  scorer ``score_single_item`` (``answer_extraction.py``), its extraction call routed to the same
  pinned reader;
* **MobileMem** (``github.com/zjunlp/MobileMem`` at ``c4d6cc15``): the text or multimodal answer
  prompt and the CORRECT/WRONG judge from ``omni/eval/eval/question_answering_and_judge_prompts.txt``,
  the question written with its options one per line as ``Raw2Locomo.py`` does.

Image items reach the reader as images, capped at the longest ranked prefix with at most 30 (the
provider's limit, MM-1 amendment 4).

    python scripts/aml_x1_mm_answers.py retrieval --out-dir x1b/out
    python scripts/aml_x1_mm_answers.py run --out-dir x1b/out --data-dir x1/data --draw x1/draw.json \\
        --eval-root x1/eval --answers answers.jsonl
    python scripts/aml_x1_mm_answers.py score --answers answers.jsonl --out score.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import threading
import time
from types import ModuleType
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

SEED = 20260925
MODEL = "deepseek/deepseek-v4.1-flash"
PROVIDER = "DeepInfra"
OPENROUTER = "https://openrouter.ai/api/v1"
ARMS = ("B", "B2", "D", "Dt")
SOURCES = ("memlens_32k", "mobilemem_omni")
IMAGE_CAP = 30
TOP = 10
ANSWER_MAX_TOKENS = 400
JUDGE_MAX_TOKENS = 300
MEMLENS_COMMIT = "77f3ab9a52fa2d6a17978e2dffe80438a4ecced2"
MOBILEMEM_COMMIT = "c4d6cc15f72462a0cad036db1245201739f47962"
CREDIT_FLOOR_USD = float(os.environ.get("RECALL_EXPERIMENT_CREDIT_FLOOR_USD", "40"))
RETRYABLE = {408, 429, 500, 502, 503, 504, 520, 522, 524}
_LABEL = re.compile(r'"label"\s*:\s*"?(CORRECT|WRONG)', re.IGNORECASE)


# ---------------------------------------------------------------------------------------------
# Stored searches


def rows(out_dir: Path, source: str) -> list[dict[str, Any]]:
    """Stage B's collected tenants of ``source`` whose collect succeeded, one row per question."""
    out = []
    for line in (out_dir / f"{source}.jsonl").read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") != "ok":
            continue
        for search in record["searches"]:
            out.append({"source": source, "tenant": record["tenant"], **search})
    return out


def items_for(arm: str, search: dict[str, Any]) -> list[dict[str, Any]]:
    """The items an arm hands the reader: B and B2 are Stage B's own, D and Dt the recorded arms."""
    if arm in ("B", "B2"):
        return list(search["items"])
    if arm in ("D", "Dt"):
        return list(search["arms"][arm]["items"])
    raise ValueError(f"unknown arm {arm!r}")


def image_refs(item: dict[str, Any]) -> list[str]:
    content = item.get("content")
    if not isinstance(content, list):
        return []
    return [str(p["sha256"]) for p in content if isinstance(p, dict) and p.get("type") == "image_ref"]


def text_of(item: dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(p.get("text") or "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text").strip()
    return ""


def capped(items: list[dict[str, Any]], limit: int = IMAGE_CAP) -> tuple[list[dict[str, Any]], bool]:
    """The longest ranked prefix of ``items`` carrying at most ``limit`` images, and whether it cut."""
    total = 0
    for index, item in enumerate(items):
        total += len(image_refs(item))
        if total > limit:
            return items[:index], True
    return items, False


def retrieval(args: argparse.Namespace) -> None:
    """Amendment 3's free check: image items in the top 10 per arm, and Dt ranks exactly as D."""
    report: dict[str, Any] = {}
    for source in SOURCES:
        path = args.out_dir / f"{source}.jsonl"
        if not path.exists():
            report[source] = {"collected": 0}
            continue
        found = rows(args.out_dir, source)
        with_image = {arm: 0 for arm in ("B", "D", "Dt")}
        order_mismatch = []
        for row in found:
            for arm in with_image:
                if any(image_refs(item) for item in items_for(arm, row)[:TOP]):
                    with_image[arm] += 1
            d_ids = [item["id"] for item in items_for("D", row)]
            dt_ids = [item["id"] for item in items_for("Dt", row)]
            if d_ids != dt_ids:
                order_mismatch.append(row["question_id"])
        n = len(found)
        report[source] = {
            "questions": n,
            "share_with_an_image_in_top10": {arm: round(v / n, 4) if n else None for arm, v in with_image.items()},
            "dt_order_differs_from_d": len(order_mismatch),
            "dt_order_differs_examples": order_mismatch[:5],
        }
    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------------------------
# Evaluators, from the pinned checkouts


def _checked(repo: Path, commit: str) -> None:
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    if head != commit:
        raise SystemExit(f"{repo} is at {head}, expected {commit}")


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def mobilemem_sections(text: str) -> dict[str, str]:
    """The three prompts of MobileMem's prompt file, keyed by their ``Prompt for ...`` headings."""
    sections: dict[str, list[str]] = {}
    current = None
    for line in text.split("\n"):
        if line.startswith("Prompt for "):
            current = line.strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {title: "\n".join(body).strip("\n") for title, body in sections.items()}


def load_evaluators(eval_root: Path) -> dict[str, Any]:
    memlens, mobilemem = eval_root / "memlens", eval_root / "mobilemem"
    _checked(memlens, MEMLENS_COMMIT)
    _checked(mobilemem, MOBILEMEM_COMMIT)
    prompts = mobilemem_sections(
        (mobilemem / "omni/eval/eval/question_answering_and_judge_prompts.txt").read_text(encoding="utf-8"))
    return {
        "prompt_builders": _load(memlens / "memory-agent/prompt_builders.py", "memlens_prompt_builders"),
        "answer_extraction": _load(memlens / "answer_extraction.py", "memlens_answer_extraction"),
        "mm_text": prompts["Prompt for Question Answering with Text Memory"],
        "mm_multimodal": prompts["Prompt for Question Answering with Mutimodal Memory"],
        "mm_judge": prompts["Prompt for LLM-as-a-Judge Evaluation"],
    }


# ---------------------------------------------------------------------------------------------
# Prompts


def memlens_messages(ev: dict[str, Any], scorer: dict[str, Any], items: list[dict[str, Any]],
                     images: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """MemLens's memory-agent answer messages; an image item is a ``[image k]`` memory line, and the
    images follow the text in the same order."""
    memories, parts = [], []
    for item in items:
        text = text_of(item)
        refs = image_refs(item)
        for ref in refs:
            parts.append(images[ref])
            text = (text + f" [image {len(parts)}]").strip()
        memories.append({"memory": text or "(empty)", "created_at": item.get("created_at")})
    messages = ev["prompt_builders"].build_mem0_answer_messages(
        str(scorer["question"]), memories, scorer.get("question_date"))
    if parts:
        messages[-1] = {"role": "user", "content": [{"type": "text", "text": messages[-1]["content"]}, *parts]}
    return messages


def mobilemem_question(scorer: dict[str, Any], options: list[str] | None) -> str:
    """The question as MobileMem's ``Raw2Locomo.py`` writes it: options one per line after it."""
    question = str(scorer["question"])
    return f"{question}\n" + "\n".join(options) if options else question


def mobilemem_messages(ev: dict[str, Any], scorer: dict[str, Any], options: list[str] | None,
                       items: list[dict[str, Any]], images: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """MobileMem's text prompt, or its multimodal prompt when any image is retrieved."""
    lines, parts, stamps = [], [], []
    for item in items:
        text = text_of(item)
        if text:
            lines.append(f"- {text}")
        for ref in image_refs(item):
            parts.append(images[ref])
            stamps.append(str(item.get("created_at") or "unknown"))
    question = mobilemem_question(scorer, options)
    memories = "\n".join(lines) if lines else "(no text memories retrieved)"
    if not parts:
        prompt = ev["mm_text"].replace("[Question]", question).replace("[Retrieved Memories]", memories)
        return [{"role": "user", "content": prompt}]
    prompt = (ev["mm_multimodal"].replace("[Question]", question)
              .replace("[Retrieved Memories]", memories).replace("[Timestamps]", ", ".join(stamps)))
    before, _, after = prompt.partition("[Images]")
    return [{"role": "user", "content": [{"type": "text", "text": before}, *parts, {"type": "text", "text": after}]}]


def mobilemem_judge_prompt(ev: dict[str, Any], scorer: dict[str, Any], options: list[str] | None,
                           prediction: str) -> str:
    return (ev["mm_judge"].replace("[Question]", mobilemem_question(scorer, options))
            .replace("[Gold Answer]", str(scorer["gold_answer"]))
            .replace("[Evidences]", "\n".join(scorer.get("evidence") or []))
            .replace("[Generated Answer]", prediction))


def judge_label(text: str) -> str | None:
    """CORRECT or WRONG from the judge's JSON ``label``; None when the reply carries neither."""
    match = _LABEL.search(text or "")
    return match.group(1).upper() if match else None


# ---------------------------------------------------------------------------------------------
# The reader


class Reader:
    """DeepSeek V4.1 Flash through one pinned provider, reasoning off, capped spend."""

    def __init__(self, key: str, spent: float, cap: float) -> None:
        self._client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=300)
        self._lock = threading.Lock()
        self.spent, self.cap = spent, cap
        self.stopped = threading.Event()

    def balance(self) -> float:
        body = self._client.get(f"{OPENROUTER}/credits").json()["data"]
        return float(body["total_credits"]) - float(body["total_usage"])

    def complete(self, messages: list[dict[str, Any]], max_tokens: int) -> tuple[str, dict[str, Any]]:
        if self.stopped.is_set():
            raise RuntimeError("stopped")
        payload = {"model": MODEL, "temperature": 0, "max_tokens": max_tokens, "messages": messages,
                   "reasoning": {"enabled": False}, "usage": {"include": True},
                   "provider": {"order": [PROVIDER], "allow_fallbacks": False}}
        for attempt in range(6):
            try:
                response = self._client.post(f"{OPENROUTER}/chat/completions", json=payload)
            except httpx.TransportError:
                time.sleep(2**attempt)
                continue
            if response.status_code == 402:
                self.stopped.set()
                raise RuntimeError("OpenRouter 402: out of credit")
            if response.status_code in RETRYABLE:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            usage = {**(body.get("usage") or {}), "provider": body.get("provider"),
                     "finish_reason": choice.get("finish_reason")}
            self.add(float(usage.get("cost") or 0.0))
            return str(choice["message"].get("content") or "").strip(), usage
        raise RuntimeError("OpenRouter kept failing")

    def add(self, cost: float) -> None:
        with self._lock:
            self.spent += cost
            if self.spent >= self.cap:
                self.stopped.set()


def pin_extraction_client(ae: ModuleType, reader: Reader, key: str) -> None:
    """Route MemLens's extraction calls through the pinned provider, reasoning off, and count their
    cost against the same cap; the extraction prompt and parsing stay MemLens's own."""
    ae._ensure_openai_client(key, OPENROUTER)
    original = ae._openai_client.chat.completions.create

    def create(**kwargs: Any) -> Any:
        extra = dict(kwargs.pop("extra_body", None) or {})
        extra.update({"provider": {"order": [PROVIDER], "allow_fallbacks": False},
                      "reasoning": {"enabled": False}, "usage": {"include": True}})
        response = original(extra_body=extra, **kwargs)
        usage = getattr(response, "usage", None)
        reader.add(float(getattr(usage, "cost", 0.0) or 0.0))
        return response

    ae._openai_client.chat.completions.create = create


# ---------------------------------------------------------------------------------------------
# Run and score


class LazyParts:
    """Digest to image file, encoded as a data-URL part only when a prompt asks for it, so thousands
    of images are never held in memory at once."""

    def __init__(self, refs: dict[str, Any]) -> None:
        self._refs = refs

    def __len__(self) -> int:
        return len(self._refs)

    def __getitem__(self, digest: str) -> dict[str, Any]:
        return self._refs[digest].part()[0]


def image_parts(source: str, data_dir: Path, draw: dict[str, Any], tenants: set[str]) -> LazyParts:
    """Each stored image digest's file, found by rebuilding the Adds Stage B made (same builder)."""
    import aml_x1_sources as x1
    from aml_x1_stageb import image_digest

    refs: dict[str, Any] = {}
    for tenant in x1.tenants_for(source, data_dir, draw, "x1"):
        if tenant.key not in tenants:
            continue
        for _, messages in tenant.sessions():
            for message in messages:
                content = message["content"]
                for ref in content if isinstance(content, list) else []:
                    if isinstance(ref, x1.ImageRef) and ref.path.is_file():
                        refs[image_digest(ref.part()[0])] = ref
    return LazyParts(refs)


def scorers(source: str, data_dir: Path, draw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    import aml_x1_sources as x1

    out = {}
    for tenant in x1.tenants_for(source, data_dir, draw, "x1"):
        for question in tenant.questions:
            out[str(question["question_id"])] = {**question["scorer"],
                                                 "options": question["search"].get("options"),
                                                 "image_evidence": bool(question.get("image_evidence"))}
    return out


def run(args: argparse.Namespace) -> None:
    ev = load_evaluators(args.eval_root)
    draw = json.loads(args.draw.read_text(encoding="utf-8"))
    key = os.environ["OPENROUTER_API_KEY"].strip()
    done: set[tuple[str, str]] = set()
    spent = 0.0
    if args.answers.exists():
        for line in args.answers.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                record = json.loads(line)
                done.add((record["id"], record["arm"]))
                spent += float(record.get("cost") or 0.0)
    reader = Reader(key, spent, args.max_usd)
    pin_extraction_client(ev["answer_extraction"], reader, key)
    balance = reader.balance()
    print(json.dumps({"done_pairs": len(done), "spent_usd": round(spent, 4), "balance_usd": round(balance, 2)}),
          flush=True)
    if balance < CREDIT_FLOOR_USD:
        raise SystemExit(f"balance {balance:.2f} below the {CREDIT_FLOOR_USD:.0f} USD floor")
    lock = threading.Lock()
    for source in args.sources:
        found = rows(args.out_dir, source)
        meta = scorers(source, args.data_dir, draw)
        images = image_parts(source, args.data_dir, draw, {row["tenant"] for row in found})
        print(json.dumps({"source": source, "questions": len(found), "images_resolved": len(images)}), flush=True)

        def work(index_row: tuple[int, dict[str, Any]], source: str = source,
                 meta: dict[str, Any] = meta, images: dict[str, Any] = images) -> None:
            index, row = index_row
            scorer = meta[str(row["question_id"])]
            order = ARMS[index % len(ARMS):] + ARMS[: index % len(ARMS)]
            for arm in order:
                ident = f"{source}:{row['question_id']}"
                if (ident, arm) in done or reader.stopped.is_set():
                    continue
                items, cut = capped(items_for(arm, row))
                if source == "memlens_32k":
                    messages = memlens_messages(ev, scorer, items, images)
                else:
                    messages = mobilemem_messages(ev, scorer, scorer.get("options"), items, images)
                judge_cost = 0.0
                try:
                    prediction, usage = reader.complete(messages, ANSWER_MAX_TOKENS)
                    if source == "memlens_32k":
                        scored = ev["answer_extraction"].score_single_item(
                            {"question_id": row["question_id"], "question": scorer["question"],
                             "question_type": scorer["question_type"],
                             "reference_answer": str(scorer["gold_answer"]), "prediction": prediction},
                            model=MODEL, use_cache=True, api_key=key, base_url=OPENROUTER)
                        correct, detail = bool(scored["correct"]), {k: scored[k] for k in (
                            "subtype", "extracted_answer", "match_method", "extraction_error")}
                    else:
                        verdict, judge_usage = reader.complete(
                            [{"role": "user", "content": mobilemem_judge_prompt(
                                ev, scorer, scorer.get("options"), prediction)}], JUDGE_MAX_TOKENS)
                        judge_cost = float(judge_usage.get("cost") or 0.0)
                        label = judge_label(verdict)
                        correct, detail = (None if label is None else label == "CORRECT"), {"judge": verdict[:600]}
                except RuntimeError:
                    return
                record = {"id": ident, "source": source, "arm": arm, "category": row["category"],
                          "image_evidence": scorer["image_evidence"],
                          "items": len(items), "images": sum(len(image_refs(i)) for i in items),
                          "capped": cut, "prediction": prediction, "correct": correct, **detail,
                          # This record's own billed calls; MemLens's extraction calls are counted
                          # in the reader's running total and cap, not per record.
                          "provider": usage.get("provider"),
                          "cost": round(float(usage.get("cost") or 0.0) + judge_cost, 6)}
                with lock, args.answers.open("a", encoding="utf-8") as sink:
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")

        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(work, enumerate(found)))
        if reader.stopped.is_set():
            break
    print(json.dumps({"spent_usd": round(reader.spent, 4), "stopped": reader.stopped.is_set()}), flush=True)


def paired(scores: dict[str, dict[str, float]], a: str, b: str, keys: list[str]) -> dict[str, Any]:
    keys = [k for k in keys if k in scores[a] and k in scores[b]]
    if not keys:
        return {"n": 0}
    diffs = [scores[a][k] - scores[b][k] for k in keys]
    rng = random.Random(SEED)
    boots = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(10_000))
    return {"n": len(keys), "a": round(sum(scores[a][k] for k in keys) / len(keys), 4),
            "b": round(sum(scores[b][k] for k in keys) / len(keys), 4),
            "diff": round(sum(diffs) / len(diffs), 4), "ci95": [round(boots[249], 4), round(boots[9_749], 4)],
            "wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs)}


def score(args: argparse.Namespace) -> None:
    by_source: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    unscored: dict[str, int] = defaultdict(int)
    image_q: dict[str, set[str]] = defaultdict(set)
    for line in args.answers.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        record = json.loads(line)
        if record["correct"] is None:
            unscored[f"{record['source']}:{record['arm']}"] += 1
            continue
        by_source[record["source"]][record["arm"]][record["id"]] = float(record["correct"])
        if record["image_evidence"]:
            image_q[record["source"]].add(record["id"])
    result: dict[str, Any] = {"unscored_answers": dict(unscored)}
    for source, scores in by_source.items():
        every = sorted(scores["B"])
        result[source] = {
            "answers_per_arm": {arm: len(scores[arm]) for arm in ARMS},
            "D_minus_B": paired(scores, "D", "B", every),
            "Dt_minus_D": paired(scores, "Dt", "D", every),
            "B2_minus_B": paired(scores, "B2", "B", every),
            "D_minus_B_image_evidence": paired(scores, "D", "B", sorted(image_q[source])),
        }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("retrieval")
    r.add_argument("--out-dir", type=Path, required=True)
    r.set_defaults(handler=retrieval)
    a = sub.add_parser("run")
    a.add_argument("--out-dir", type=Path, required=True)
    a.add_argument("--data-dir", type=Path, required=True)
    a.add_argument("--draw", type=Path, required=True)
    a.add_argument("--eval-root", type=Path, required=True)
    a.add_argument("--answers", type=Path, required=True)
    a.add_argument("--sources", nargs="+", default=list(SOURCES), choices=SOURCES)
    a.add_argument("--workers", type=int, default=2)
    a.add_argument("--max-usd", type=float, default=5.0)
    a.set_defaults(handler=run)
    s = sub.add_parser("score")
    s.add_argument("--answers", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(handler=score)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
