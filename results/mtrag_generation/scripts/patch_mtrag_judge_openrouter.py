"""Point the MTRAG judge at an OpenAI-compatible endpoint instead of Azure.

The official README sanctions exactly this: "Our implementation for evaluating with GPT assumes an
Azure endpoint. If you are using an alternate endpoint, you will need to modify the client."

WHY THE MODEL MUST NOT CHANGE. `judge_wrapper.py` hard-codes `gpt-4o-mini-2024-07-18` on the openai
path, and that is the judge behind the published table. The vLLM route uses whatever
`--judge_model` you pass (the README example is `ibm-granite/granite-3.3-8b-instruct`), and judging
with a different model measures a different thing while producing a number that LOOKS comparable.
So this changes the transport and nothing else: same model, same
`temperature 0.0 / top_p 1 / max_tokens 400 / seed 100`.

HOW. Rather than rewriting call sites (whitespace-fragile, and the first attempt failed on exactly
that), it inserts a shim next to each constructor and redirects the constructor NAME to it. The
kwargs at the call site are untouched, so the diff cannot silently drop one. Each shim falls
through to the original Azure class when `OPENAI_COMPAT_BASE_URL` is unset, so this is additive.

Requires `OPENAI_AZURE_HOST` to be set to any non-empty string: an upstream guard raises without
it. Its value is unused on the compat path.

Idempotent, writes a `.orig` beside each file, and FAILS LOUDLY if an anchor is missing rather than
reporting success while changing nothing.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

MARKER = "OPENAI_COMPAT_SHIM"

CLIENT_SHIM = '''
# --- OPENAI_COMPAT_SHIM: alternate endpoint per the README's tip. Transport only, same model.
def _compat_client(**azure_kwargs):
    """An OpenAI-compatible client when OPENAI_COMPAT_BASE_URL is set, else the original Azure one."""
    import os as _os

    base = _os.environ.get("OPENAI_COMPAT_BASE_URL")
    if not base:
        return AzureOpenAI(**azure_kwargs)
    from openai import OpenAI

    return OpenAI(base_url=base, api_key=azure_kwargs.get("api_key"))
'''

CHAT_SHIM = '''
# --- OPENAI_COMPAT_SHIM: same judge model, alternate transport. RAGAS computes only `faithfulness`
# (RL_F) here and the embedding client is commented out upstream, so a chat-only provider suffices.
def _compat_chat(**azure_kwargs):
    import os as _os

    base = _os.environ.get("OPENAI_COMPAT_BASE_URL")
    if not base:
        return AzureChatOpenAI(**azure_kwargs)
    prefix = _os.environ.get("OPENAI_COMPAT_MODEL_PREFIX", "")
    name = azure_kwargs["deployment_name"]
    return ChatOpenAI(
        model=name if name.startswith(prefix) else prefix + name,
        base_url=base,
        api_key=azure_kwargs.get("openai_api_key"),
        timeout=azure_kwargs.get("timeout", 120),
    )
'''

MODEL_ID_OVERRIDE = '''
        # --- OPENAI_COMPAT_SHIM: OpenRouter namespaces models as `openai/<name>`; Azure deployments
        # do not. The underlying checkpoint is identical, only the routing string differs.
        if os.environ.get("OPENAI_COMPAT_BASE_URL"):
            _prefix = os.environ.get("OPENAI_COMPAT_MODEL_PREFIX", "")
            if not self.model_id.startswith(_prefix):
                self.model_id = _prefix + self.model_id
'''


def _edit(path: Path, fn) -> str:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return f"SKIP {path.name}: already patched"
    new = fn(text)
    if new == text:
        raise SystemExit(
            f"FAILED {path.name}: nothing changed. The upstream file differs from what this patch "
            f"expects; re-read it rather than forcing the edit."
        )
    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(new, encoding="utf-8")
    return f"PATCHED {path.name}"


def patch_client(text: str) -> str:
    # Shim goes after the module-level `_verbose` flag, before the class that uses it.
    text, n = re.subn(r"(?m)^(_verbose\s*=\s*False\s*)$", r"\1\n" + CLIENT_SHIM, text, count=1)
    assert n == 1, "could not place the client shim"
    # Redirect the constructor; the call site's kwargs stay exactly as upstream wrote them.
    text, n = re.subn(r"self\.client\s*=\s*AzureOpenAI\(", "self.client = _compat_client(", text, count=1)
    assert n == 1, "could not redirect AzureOpenAI"
    text, n = re.subn(
        r"(?m)^(\s*self\.model_id = model_id\s*)$", r"\1\n" + MODEL_ID_OVERRIDE, text, count=1
    )
    assert n == 1, "could not place the model_id override"
    return text


def patch_wrapper(text: str) -> str:
    text, n = re.subn(
        r"(?m)^(warnings\.filterwarnings\('ignore'\)\s*)$", r"\1\n" + CHAT_SHIM, text, count=1
    )
    assert n == 1, "could not place the chat shim"
    text, n = re.subn(r"llm\s*=\s*AzureChatOpenAI\(", "llm = _compat_chat(", text, count=1)
    assert n == 1, "could not redirect AzureChatOpenAI"
    return text


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    evaldir = root / "scripts" / "evaluation"
    print(_edit(evaldir / "azure_openai_client.py", patch_client))
    print(_edit(evaldir / "judge_wrapper.py", patch_wrapper))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
