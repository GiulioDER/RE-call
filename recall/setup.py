from __future__ import annotations

import importlib.util
import json
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from recall.calibration import Calibration, from_samples, save
from recall.embeddings import resolve_embedder
from recall.eval.calibrate import CalibrationReport

SETUP_BEGIN = "# recall setup begin"
SETUP_END = "# recall setup end"
DEFAULT_ENV_PATH = Path(".env")
DEFAULT_CALIBRATION_PATH = Path("calibration.json")
MODEL_DOWNLOAD_FLOOR_BYTES = 1_500_000_000
CLAUDE_MD_BEGIN = "<!-- recall setup begin -->"
CLAUDE_MD_END = "<!-- recall setup end -->"
DEFAULT_CLAUDE_MD_PATH = Path("CLAUDE.md")
DEFAULT_MEMORY_DIR = Path("memory")


@dataclass(frozen=True)
class HardwareProbe:
    cpu_count: int | None
    gpu: str | None
    cuda_available: bool
    free_bytes: int
    internet: bool
    fastembed_available: bool
    sentence_transformers_available: bool


@dataclass(frozen=True)
class Choice:
    label: str
    value: str
    description: str


@dataclass(frozen=True)
class CalibrationResult:
    path: Path
    calibration: Calibration
    report: CalibrationReport


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _probe_gpu() -> str | None:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return str(torch.cuda.get_device_name(0))
        except Exception:
            pass
        try:
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "apple mps"
        except Exception:
            pass
    if shutil.which("nvidia-smi"):
        return "nvidia-smi"
    return None


def _probe_cuda() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _probe_internet(url: str = "https://pypi.org", timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout):  # noqa: S310 - install-time probe
            return True
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def probe_hardware(path: Path | None = None) -> HardwareProbe:
    root = path or Path.cwd()
    free_bytes = shutil.disk_usage(root).free
    cuda_available = _probe_cuda()
    return HardwareProbe(
        cpu_count=os.cpu_count(),
        gpu=_probe_gpu(),
        cuda_available=cuda_available,
        free_bytes=free_bytes,
        internet=_probe_internet(),
        fastembed_available=_module_available("fastembed"),
        sentence_transformers_available=_module_available("sentence_transformers"),
    )


def _can_download_models(probe: HardwareProbe) -> bool:
    return probe.internet and probe.free_bytes >= MODEL_DOWNLOAD_FLOOR_BYTES


def embedder_choices(
    probe: HardwareProbe,
    *,
    security_required: bool,
    cloud_keys: dict[str, str],
) -> list[Choice]:
    choices = [
        Choice(
            label="hashing",
            value="hashing",
            description="No model download, no network, works anywhere",
        ),
    ]
    if probe.fastembed_available and _can_download_models(probe):
        choices.append(
            Choice(
                label="fastembed",
                value="fastembed",
                description="Local bge-small embedder, best default when offline matters",
            )
        )
    if probe.sentence_transformers_available and _can_download_models(probe):
        choices.append(
            Choice(
                label="sentence transformers",
                value="st:sentence-transformers/all-MiniLM-L6-v2",
                description="Local transformer embedder, good if you already have the extra",
            )
        )
    voyage_key = cloud_keys.get("VOYAGE_API_KEY", "")
    openai_key = cloud_keys.get("OPENAI_API_KEY", "") or cloud_keys.get(
        "OPENROUTER_API_KEY", ""
    )
    if (
        not security_required
        and probe.internet
        and voyage_key
        and _module_available("voyageai")
    ):
        choices.append(
            Choice(
                label="voyage cloud",
                value="voyage:voyage-3",
                description="Cloud embedder, only when data may leave the machine",
            )
        )
    if (
        not security_required
        and probe.internet
        and openai_key
        and _module_available("openai")
    ):
        choices.append(
            Choice(
                label="openai compatible cloud",
                value="openai:text-embedding-3-small",
                description="OpenAI compatible cloud path through OpenRouter or OpenAI",
            )
        )
    return choices


def reranker_choices(
    probe: HardwareProbe, *, security_required: bool
) -> list[Choice]:
    choices = [Choice(label="none", value="", description="Skip reranking for speed")]
    if probe.sentence_transformers_available and _can_download_models(probe):
        choices.append(
            Choice(
                label="ms marco reranker",
                value="RECALL_RERANK=1",
                description="Default local cross encoder, the measured rerank win",
            )
        )
        if not security_required:
            choices.append(
                Choice(
                    label="bge reranker",
                    value="RECALL_RERANK=1;RECALL_RERANK_MODEL=BAAI/bge-reranker-base",
                    description="Heavier local reranker, available if you want to try it",
                )
            )
    return choices


def sparse_choices(probe: HardwareProbe) -> list[Choice]:
    choices = [
        Choice(
            label="postgres fts",
            value="RECALL_SPARSE=fts",
            description="Current sparse retrieval path, no extra GPU requirement",
        )
    ]
    if probe.cuda_available and probe.sentence_transformers_available and _can_download_models(probe):
        choices.append(
            Choice(
                label="splade",
                value="RECALL_SPARSE=splade",
                description="CUDA sparse encoder for higher coverage when the machine can run it",
            )
        )
    return choices


def entailment_choices(probe: HardwareProbe) -> list[Choice]:
    choices = []
    if probe.sentence_transformers_available and _can_download_models(probe):
        choices.append(
            Choice(
                label="qnli judge",
                value="RECALL_ENTAILMENT=1;RECALL_ENTAILMENT_MODEL=cross-encoder/qnli-distilroberta-base;RECALL_ENTAILMENT_REVISION=7dd04ee0a6040c06fb381ad7edcb8585f4d937fd",
                description="Default optional entailment judge, pinned and local",
            )
        )
        choices.append(
            Choice(
                label="nli judge",
                value="RECALL_ENTAILMENT=1;RECALL_ENTAILMENT_MODEL=cross-encoder/nli-deberta-v3-large",
                description="Stronger local judge, larger and unpinned by default",
            )
        )
    return choices


def _prompt(
    input_fn: Callable[[str], str],
    print_fn: Callable[..., None],
    text: str,
) -> str:
    print_fn(text, end="")
    return input_fn("").strip()


def _ask_yes_no(
    input_fn: Callable[[str], str],
    print_fn: Callable[..., None],
    text: str,
    *,
    default: bool | None = None,
) -> bool:
    suffix = " [y/n] "
    if default is True:
        suffix = " [Y/n] "
    elif default is False:
        suffix = " [y/N] "
    while True:
        raw = _prompt(input_fn, print_fn, f"{text}{suffix}").lower()
        if not raw and default is not None:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print_fn("Please answer yes or no.")


def _choose(
    input_fn: Callable[[str], str],
    print_fn: Callable[..., None],
    title: str,
    choices: Sequence[Choice],
) -> Choice:
    if not choices:
        raise ValueError(f"no choices available for {title}")
    print_fn(title)
    for i, choice in enumerate(choices, 1):
        print_fn(f"  {i}. {choice.label}: {choice.description}")
    while True:
        raw = _prompt(input_fn, print_fn, "Select a number: ")
        try:
            idx = int(raw)
        except ValueError:
            idx = 0
        if 1 <= idx <= len(choices):
            return choices[idx - 1]
        print_fn(f"Choose one of 1..{len(choices)}.")


def _update_env_block(path: Path, values: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    start = next((i for i, line in enumerate(lines) if line.strip() == SETUP_BEGIN), None)
    end = next((i for i, line in enumerate(lines) if line.strip() == SETUP_END), None)
    block = [SETUP_BEGIN]
    for key, value in values.items():
        block.append(f"{key}={_quote_env(value)}")
    block.append(SETUP_END)
    if start is not None and end is not None and end >= start:
        lines = [*lines[:start], *block, *lines[end + 1 :]]
    else:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(block)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _update_markdown_block(path: Path, begin: str, end: str, content: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    start = next((i for i, line in enumerate(lines) if line.strip() == begin), None)
    stop = next((i for i, line in enumerate(lines) if line.strip() == end), None)
    block = [begin, *content.splitlines(), end]
    if start is not None and stop is not None and stop >= start:
        lines = [*lines[:start], *block, *lines[stop + 1 :]]
    else:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(block)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _claude_md_block() -> str:
    return (
        "## Using recall\n"
        "\n"
        "This project is indexed by recall. Call `recall_search` before proposing an idea, "
        "forming a hypothesis, or repeating past work. If a closed decision or falsified "
        "hypothesis surfaces, do not re-litigate it.\n"
        "\n"
        "- When `abstained` is true, no hit survived the trust gate — say you do not know instead "
        "of answering from degraded hits.\n"
        "- Use `recall_evidence` instead of `recall_search` when about to answer from memory "
        "rather than just consult it; cite only `chunk_id` values from its `items`.\n"
        "- Write new durable facts to `memory/`, one file per fact, indexed by "
        "`memory/MEMORY.md` (see that file for the format), so `recall_index` can find them.\n"
    )


def scaffold_claude_md(path: Path = DEFAULT_CLAUDE_MD_PATH) -> None:
    _update_markdown_block(path, CLAUDE_MD_BEGIN, CLAUDE_MD_END, _claude_md_block())


def _quote_env(value: str) -> str:
    if value == "":
        return '""'
    if any(ch.isspace() for ch in value) or any(ch in value for ch in '#"'):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def _resolve_output_path(raw: str | None, *, default: Path) -> Path:
    path = Path(raw).expanduser() if raw else default
    if path.exists() and path.is_dir():
        path = path / default.name
    return path.resolve()


def _looks_like_windows_path(raw: str) -> bool:
    return len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha()


def _require_local_path(raw: str, *, label: str) -> Path:
    if _looks_like_windows_path(raw) and os.name != "nt":
        raise ValueError(
            f"{label} looks like a Windows host path ({raw!r}). Inside Docker, use the "
            "container path instead, for example /app/recall/eval/queries.json."
        )
    return Path(raw).expanduser().resolve()


def _require_local_output_path(raw: str | None, *, label: str, default: Path) -> Path:
    if raw and _looks_like_windows_path(raw) and os.name != "nt":
        raise ValueError(
            f"{label} looks like a Windows host path ({raw!r}). Inside Docker, use the "
            "container path instead, for example /app/recall/calibration.json."
        )
    return _resolve_output_path(raw, default=default)


def calibrate_from_files(
    *,
    dsn: str,
    embedder_name: str,
    queries_path: Path,
    corpus_dir: Path | None = None,
    out: Path | None = None,
    env: dict[str, str] | None = None,
) -> CalibrationResult:
    queries_path = _require_local_path(str(queries_path), label="Path to labeled queries JSON")
    corpus_dir = (
        _require_local_path(str(corpus_dir), label="Path to your corpus") if corpus_dir else None
    )
    out = _require_local_output_path(
        str(out) if out else None,
        label="Calibration output path",
        default=DEFAULT_CALIBRATION_PATH,
    )
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    labeled = [q for q in queries if isinstance(q, dict) and not q.get("trust")]
    if not labeled or not all("query" in q and "answerable" in q for q in labeled):
        raise ValueError(
            "queries file entries need 'query' and 'answerable' keys "
            "(see recall/eval/queries.json for the format)"
        )
    if not any(q["answerable"] for q in labeled) or not any(not q["answerable"] for q in labeled):
        raise ValueError(
            "queries file needs at least one answerable AND one unanswerable entry"
        )
    embedder = resolve_embedder(embedder_name, env=env)
    from recall.eval.calibrate import calibrate as run_calibration

    measured = run_calibration(
        dsn,
        embedder,
        corpus_dir=corpus_dir,
        queries_path=queries_path,
    )
    cal = from_samples(
        embedder.name,
        measured.answerable_max_cos,
        measured.unanswerable_max_cos,
    )
    path = save(cal, out)
    return CalibrationResult(path=path, calibration=cal, report=measured)


def run_setup_wizard(
    *,
    dsn: str,
    env_path: Path = DEFAULT_ENV_PATH,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
) -> dict[str, str]:
    probe = probe_hardware()
    print_fn(
        "Hardware check: "
        f"cpu={probe.cpu_count or 'unknown'}, "
        f"gpu={probe.gpu or 'none'}, "
        f"free_disk={probe.free_bytes // (1024 * 1024 * 1024)} GB, "
        f"internet={'yes' if probe.internet else 'no'}"
    )
    security_required = _ask_yes_no(
        input_fn, print_fn, "Is data security necessary for this installation?", default=True
    )
    cloud_keys: dict[str, str] = {}
    if not security_required:
        print_fn("Step 1b, optional API keys. Leave blank to skip.")
        for key in ("VOYAGE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
            value = _prompt(input_fn, print_fn, f"{key}: ")
            if value:
                cloud_keys[key] = value
    embedders = embedder_choices(
        probe, security_required=security_required, cloud_keys=cloud_keys
    )
    embedder = _choose(
        input_fn,
        print_fn,
        "Choose the embedder you want to use:",
        embedders,
    )
    rerankers = reranker_choices(probe, security_required=security_required)
    reranker = _choose(
        input_fn,
        print_fn,
        "Choose whether to enable reranking:",
        rerankers,
    )
    sparse_backend = _choose(
        input_fn,
        print_fn,
        "Choose the sparse retrieval backend:",
        sparse_choices(probe),
    )
    entailment = None
    entailment_options = entailment_choices(probe)
    if entailment_options:
        if _ask_yes_no(
            input_fn,
            print_fn,
            "Enable the optional entailment judge before the final answer?",
            default=False,
        ):
            entailment = _choose(
                input_fn,
                print_fn,
                "Choose the entailment judge model:",
                entailment_options,
            )
    else:
        print_fn(
            "Entailment judge is unavailable on this machine. It needs sentence-transformers "
            "and enough internet and disk to download a model."
        )

    values: dict[str, str] = {
        "RECALL_DSN": dsn,
        "RECALL_SECURITY_REQUIRED": "1" if security_required else "0",
        "RECALL_EMBEDDER": embedder.value,
        "RECALL_SPARSE": "fts",
        "RECALL_ENTAILMENT": "0",
    }
    if reranker.value == "RECALL_RERANK=1":
        values["RECALL_RERANK"] = "1"
    elif reranker.value.startswith("RECALL_RERANK=1;"):
        values["RECALL_RERANK"] = "1"
        for chunk in reranker.value.split(";")[1:]:
            key, _, value = chunk.partition("=")
            if key and value:
                values[key] = value
    else:
        values["RECALL_RERANK"] = "0"

    if sparse_backend.value:
        key, _, value = sparse_backend.value.partition("=")
        if key and value:
            values[key] = value

    if entailment is not None and entailment.value:
        parts = entailment.value.split(";")
        for chunk in parts:
            key, _, value = chunk.partition("=")
            if key and value:
                values[key] = value

    values.update(cloud_keys)

    if _ask_yes_no(
        input_fn,
        print_fn,
        "Do you want to calibrate the threshold now against a labeled query file?",
        default=False,
    ):
        queries_raw = _prompt(
            input_fn, print_fn, "Path to labeled queries JSON [leave blank to skip]: "
        )
        corpus_raw = _prompt(
            input_fn, print_fn, "Path to your corpus [leave blank to skip]: "
        )
        if not queries_raw or not corpus_raw:
            print_fn(
                "Calibration skipped. Run recall calibrate later with a labeled query file "
                "when you have one."
            )
            _update_env_block(env_path, values)
            print_fn(f"Wrote {env_path}")
            return values
        queries = _require_local_path(queries_raw, label="Path to labeled queries JSON")
        corpus = _require_local_path(corpus_raw, label="Path to your corpus")
        out_raw = _prompt(input_fn, print_fn, "Calibration output path [calibration.json]: ")
        try:
            result = calibrate_from_files(
                dsn=dsn,
                embedder_name=embedder.value,
                queries_path=queries,
                corpus_dir=corpus,
                out=_require_local_output_path(
                    out_raw or None,
                    label="Calibration output path",
                    default=DEFAULT_CALIBRATION_PATH,
                ),
                env=cloud_keys,
            )
        except ValueError as exc:
            print_fn(str(exc))
            raise SystemExit(2) from exc
        values["RECALL_CALIBRATION"] = str(result.path)
        print_fn(
            f"Calibration saved to {result.path}. Threshold={result.calibration.threshold:.3f}"
        )
    else:
        print_fn(
            "Calibration skipped. Run recall calibrate later with a labeled query file "
            "when you have one."
        )

    _update_env_block(env_path, values)
    print_fn(f"Wrote {env_path}")
    return values
