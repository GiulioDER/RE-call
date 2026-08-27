"""Keep a hooked arm's context as clean as `--bare` kept it, and prove it rather than assume it.

`--bare` skips hooks, so an arm that needs one must run under a `CLAUDE_CONFIG_DIR` instead, and
that admits `CLAUDE.md` where `--bare` did not. Measured 2026-08-27 with one trivial prompt, no
tools, one turn:

    --bare, anywhere                     ~3,000 input tokens, no memory documents
    config dir, cwd under the profile    47,676 to 66,167, user and project CLAUDE.md present

**The file is found by walking UP from `cwd`, not through any environment variable.** That was not
the first hypothesis and the first two were wrong, which is why they are recorded here:

| attempt | result |
|---|---|
| set `HOME` and `USERPROFILE` to an empty directory | still leaked, 48,758 |
| set eight home-ish variables at once (`HOMEPATH`, `APPDATA`, `LOCALAPPDATA`, `XDG_*`) | still leaked, 48,606 |
| plant a `CLAUDE.md` inside the config dir to shadow it | still leaked, 47,837, and the planted file loaded TOO, so config-dir memory is ADDITIVE |
| **run with `cwd` outside the user profile** | **clean**: the agent reports no memory documents |

Every `cwd` in the first three attempts sat under `C:\\Users\\<user>\\`, so the walk reached
`C:\\Users\\<user>\\.claude\\CLAUDE.md` no matter what the environment said. That is the whole
mechanism, and it makes the fix a PATH rather than a variable.

⚠️ **Two things this does NOT close, both stated so nobody reads more into a clean check.**

1. A non-bare session carries a much larger system prompt: ~35,000 input tokens against `--bare`'s
   ~2,900 on the same prompt, and `--disable-slash-commands` moves that by only ~1,700 while also
   stopping the hook from firing. That difference is identical in both arms, so it cannot bias the
   comparison, but it is not the condition earlier `--bare` results were measured under.
2. A clean check is a check of ONE session. It is cheap, so run it per run, not once ever.
"""

from __future__ import annotations

from pathlib import Path

#: The prompt the check asks. It names the documents it is looking for, which risks a leading
#: answer, so the token count is reported beside it: a leaked CLAUDE.md is tens of thousands of
#: tokens and the two signals have to agree.
ISOLATION_PROMPT = (
    "Answer in one line, in this exact shape: USER=<yes|no> PROJECT=<yes|no>. "
    "USER is yes if your context contains a document with the heading 'User-level notes' "
    "or a rule about 'No dash as punctuation'. PROJECT is yes if your context contains a "
    "document titled 'recall: working rules' or a section 'One session, one workspace'."
)


class IsolationCheckUnavailable(RuntimeError):
    """The check could not be PERFORMED. Not the same as the check failing.

    A run still refuses to proceed, because unverified isolation is not verified isolation.
    But the operator has to know whether they are looking at a leak or at an API failure,
    because the fixes have nothing in common.
    """


def user_profile_root() -> Path:
    """The directory whose ancestors carry the user's `CLAUDE.md`."""

    return Path.home().resolve()


def is_outside_user_profile(path: str | Path) -> bool:
    """True when no ancestor of `path` is the user profile, so the walk cannot reach its memory."""

    candidate = Path(path).resolve()
    profile = user_profile_root()
    return not (candidate == profile or profile in candidate.parents)


def assert_sandbox_isolated(work_root: str | Path) -> None:
    """Refuse a hooked run whose sandboxes sit under the user profile.

    Raising here rather than warning is deliberate: the leak does not fail, it produces a complete
    run whose every session silently carried tens of thousands of tokens of this machine's memory,
    and the artifacts of that run are indistinguishable from a clean one.
    """

    root = Path(work_root).resolve()
    if is_outside_user_profile(root):
        return
    raise SystemExit(
        f"--hook-file needs sandboxes outside the user profile, and {root} is inside "
        f"{user_profile_root()}.\n"
        "A hooked arm runs under CLAUDE_CONFIG_DIR instead of --bare, and CLAUDE.md is found by "
        "walking up from cwd, so every session there would silently receive this machine's user "
        "memory. Pass --work-root pointing somewhere outside the profile, for example "
        "C:/recall-ab-sandbox."
    )


async def verify_isolation(
    *,
    model: str,
    config_dir: Path,
    cwd: Path,
    env: dict[str, str],
    timeout_s: float = 180.0,
) -> tuple[bool, int, str]:
    """Run one real session under the run's own settings and ask what memory it holds.

    Returns `(clean, input_tokens, response)`. Costs one short session, which is the cheapest
    positive control available for a failure mode that otherwise produces a complete, plausible,
    contaminated run.
    """

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks.agent_ab.claude_exec import ClaudeExecConfig, run_claude_case
    from benchmarks.agent_ab.schema import RECALL_OFF

    config = ClaudeExecConfig(
        model=model,
        cwd=cwd,
        timeout_s=timeout_s,
        env=dict(env),
        bare=False,
        config_dir=config_dir,
        strict_mcp_config=False,
        allowed_tools=(),
    )
    record = await run_claude_case(
        {"task_id": "isolation-check", "user_input": ISOLATION_PROMPT}, RECALL_OFF, config
    )
    response = (record.response or "").strip().replace("\n", " ")
    # "the check found a leak" and "the check could not run" are both not-clean, and a caller
    # needs to tell them apart: one is a broken experiment, the other a broken API call.
    # Reporting the second as the first sends the operator hunting a leak that is not there.
    if record.error or response.startswith("API Error") or not response:
        raise IsolationCheckUnavailable(record.error or response or "no response at all")
    clean = "USER=no" in response and "PROJECT=no" in response
    return clean, int(record.input_tokens or 0), response
