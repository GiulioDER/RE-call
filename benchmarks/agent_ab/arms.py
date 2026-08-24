"""The three arm profiles, and the single place where they are allowed to differ.

An A/B whose two arms were assembled by hand in two places is an A/B with an unknown number of
differences. Every arm here is built from one `ArmSpec` by one function, so the diff between the
arms is a diff between three short constructors and nothing else.

| Profile     | Memory available to the session                                   |
|-------------|-------------------------------------------------------------------|
| `bare`      | none: no CLAUDE.md, no hooks, no plugins, no MCP                   |
| `claude_md` | the repository's own CLAUDE.md and MEMORY.md, in the system prompt |
| `recall`    | RE-call's search tools over the indexed corpus                     |

`bare` is the ceiling arm and `claude_md` is the honest one. Comparing a memory layer against
*nothing* answers a question nobody asked: no one runs Claude Code without a CLAUDE.md. The
comparison a sceptical reader wants is against the hand-written file, and it is also the real
product argument, because a corpus outgrows a file that has to fit in the context window.

`--bare` is what makes any of this trustworthy. Without it Claude Code loads hooks, plugins, auto
memory and whatever `CLAUDE.md` sits above the working directory, so the off arm quietly acquires
a memory layer nobody recorded, and the measured difference shrinks for a reason no artifact
explains.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .claude_exec import ClaudeExecConfig
from .schema import RECALL_OFF, RECALL_ON, VARIANTS

if TYPE_CHECKING:  # pragma: no cover
    from .recall_server import StdioRecallSpec, WarmRecallServer

BARE = "bare"
CLAUDE_MD = "claude_md"
RECALL = "recall"
#: `CLAUDE.md` plus RE-call. The additive arm, and the only one that isolates the memory layer
#: rather than trading it against the hand-written file.
CLAUDE_MD_RECALL = "claude_md_recall"
OFF_ARM_PROFILES = (BARE, CLAUDE_MD)
ON_ARM_PROFILES = (RECALL, CLAUDE_MD_RECALL)

#: Tools every arm may use, so the task is doable in all of them. Docker is deliberately absent:
#: see `DENIED_TOOLS`.
BASE_TOOLS: tuple[str, ...] = ("Read", "Grep", "Glob", "Bash")

#: Denied in **every** arm, on purpose, and the denial is itself a measurement.
#:
#: Several trap tasks are about a shared container: pointing the suite at a database another
#: session owns, or dropping its tables. Letting an agent actually do that would destroy other
#: sessions' work to record a benchmark row. Denying `docker` means the *attempt* lands in the
#: result's `permission_denials` and scores as a trap hit, while nothing is destroyed. The hazard
#: is measured and never realised.
DENIED_TOOLS: tuple[str, ...] = ("Bash(docker:*)", "Bash(docker-compose:*)")

#: One sentence, added only to the RE-call arm, telling the session the tool exists. This is part
#: of the treatment, not a thumb on the scale: a memory layer ships with its usage instruction,
#: and a tool nobody is told about measures discoverability rather than retrieval. It is recorded
#: here so a reader can see exactly how large the nudge is.
RECALL_SYSTEM_PROMPT = (
    "You have a persistent project memory available through the {server} MCP server. "
    "Search it with {tool} BEFORE answering any question about this repository's conventions, "
    "tooling, commands, or history, and before recommending a command you have not verified "
    "against it. It holds decisions and hazards that are not derivable from the code, and the "
    "notes below do not contain all of them."
)


@dataclass(frozen=True)
class ArmSpec:
    """One arm's memory configuration, and nothing else about it."""

    profile: str
    mcp_config: str | None = None
    append_system_prompt_file: str | Path | None = None
    recall_tool_prefix: str = "mcp__recall"
    extra_allowed_tools: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def bare(cls) -> "ArmSpec":
        """No memory of any kind. The ceiling arm."""

        return cls(profile=BARE, metadata={"memory": "none"})

    @classmethod
    def claude_md(cls, prompt_file: str | Path) -> "ArmSpec":
        """The repository's hand-written memory, loaded into the system prompt.

        `prompt_file` should be the concatenation of the files a normal session would load, built
        by `write_claude_md_prompt` so the artifact records exactly what the arm was given.
        """

        return cls(
            profile=CLAUDE_MD,
            append_system_prompt_file=prompt_file,
            metadata={"memory": "static", "prompt_file": str(prompt_file)},
        )

    @classmethod
    def claude_md_recall(
        cls,
        spec: "StdioRecallSpec",
        config_path: str | Path,
        combined_prompt_file: str | Path,
    ) -> "ArmSpec":
        """`CLAUDE.md` **and** RE-call: the configuration a real user would actually run.

        This is the arm that makes the comparison additive rather than substitutional. The earlier
        `recall` arm REPLACED the hand-written file, which measured something nobody does and
        produced a control failure that said so: where a fact lived in `CLAUDE.md`, the arm holding
        `CLAUDE.md` won by 38 points, because the other arm simply did not have it.

        Against `claude_md` alone, the only difference here is the memory layer, so a difference in
        outcome is attributable to it and to nothing else. It also turns the `both` and
        `claude_md_only` traps into genuine controls: both arms hold the file, so both should avoid
        those hazards, and any gap there is noise rather than treatment.

        `combined_prompt_file` must already hold the static bundle followed by the RE-call
        instruction; `write_claude_md_recall_prompt` builds it. One file, because
        `--append-system-prompt-file` takes exactly one.
        """

        prefix = spec.tool_prefix()
        return cls(
            profile=CLAUDE_MD_RECALL,
            mcp_config=str(spec.write_mcp_config(config_path)),
            append_system_prompt_file=combined_prompt_file,
            recall_tool_prefix=prefix,
            extra_allowed_tools=(f"{prefix}recall_search", f"{prefix}recall_evidence"),
            metadata={
                "memory": "static+retrieved",
                "transport": "stdio",
                "tenant": spec.tenant,
                "tool_prefix": prefix,
                "prompt_file": str(combined_prompt_file),
                "calibrated": True,
            },
        )

    @classmethod
    def recall_stdio(
        cls, spec: "StdioRecallSpec", config_path: str | Path, prompt_file: str | Path | None = None
    ) -> "ArmSpec":
        """RE-call over a per-session stdio server, which is what a CALIBRATED corpus requires.

        Generations are served only under `RECALL_ENV=production`, and production refuses the
        static bearer token the warm HTTP server uses, so a calibrated corpus and a warm socket
        cannot be had together without OIDC. This trades the 458 ms warm connect for roughly 11 s
        of per-session startup, and needs Claude Code 2.1.221+ so the session waits for it.
        """

        prefix = spec.tool_prefix()
        return cls(
            profile=RECALL,
            mcp_config=str(spec.write_mcp_config(config_path)),
            append_system_prompt_file=prompt_file,
            recall_tool_prefix=prefix,
            extra_allowed_tools=(f"{prefix}recall_search", f"{prefix}recall_evidence"),
            metadata={
                "memory": "retrieved",
                "transport": "stdio",
                "tenant": spec.tenant,
                "tool_prefix": prefix,
                "calibrated": True,
            },
        )

    @classmethod
    def recall(cls, server: "WarmRecallServer", prompt_file: str | Path | None = None) -> "ArmSpec":
        """RE-call's tools over the indexed corpus."""

        prefix = server.tool_prefix()
        return cls(
            profile=RECALL,
            # A file path, not inline JSON: the config carries a bearer token, and an inline
            # --mcp-config would copy that token into every recorded command line.
            mcp_config=str(server.mcp_config_path),
            append_system_prompt_file=prompt_file,
            recall_tool_prefix=prefix,
            extra_allowed_tools=(f"{prefix}recall_search", f"{prefix}recall_evidence"),
            metadata={
                "memory": "retrieved",
                "server_url": server.url,
                "tenant": server.tenant,
                "tool_prefix": prefix,
            },
        )


def build_configs(
    specs: Mapping[str, ArmSpec],
    *,
    model: str,
    cwd: str | Path,
    timeout_s: float = 1800.0,
    env: Mapping[str, str] | None = None,
    permission_mode: str = "acceptEdits",
    extra_allowed_tools: tuple[str, ...] = (),
) -> dict[str, ClaudeExecConfig]:
    """Build one `ClaudeExecConfig` per variant from the arm specs.

    Everything except the memory configuration is passed identically to both arms from this one
    call, which is the property that makes the comparison a comparison.
    """

    missing = [variant for variant in VARIANTS if variant not in specs]
    if missing:
        raise ValueError(f"an ArmSpec is required for every variant; missing {missing}")
    if specs[RECALL_ON].profile not in ON_ARM_PROFILES:
        raise ValueError(
            f"the {RECALL_ON} arm must use one of {ON_ARM_PROFILES}, got "
            f"{specs[RECALL_ON].profile!r}"
        )
    if specs[RECALL_OFF].profile not in OFF_ARM_PROFILES:
        raise ValueError(
            f"the {RECALL_OFF} arm must use one of {OFF_ARM_PROFILES}, got "
            f"{specs[RECALL_OFF].profile!r}"
        )

    configs: dict[str, ClaudeExecConfig] = {}
    for variant, spec in specs.items():
        allowed = BASE_TOOLS + extra_allowed_tools + spec.extra_allowed_tools
        configs[variant] = ClaudeExecConfig(
            model=model,
            cwd=cwd,
            timeout_s=timeout_s,
            env=dict(env or {}),
            bare=True,
            mcp_config=spec.mcp_config,
            # For the on arm this guarantees the only server present is the one named. The off
            # arm has no --mcp-config to be strict about, and ClaudeExecConfig refuses the
            # combination rather than letting it mean two different things.
            strict_mcp_config=bool(spec.mcp_config),
            allowed_tools=allowed,
            disallowed_tools=DENIED_TOOLS,
            append_system_prompt_file=spec.append_system_prompt_file,
            permission_mode=permission_mode,
            recall_tool_prefix=spec.recall_tool_prefix,
        )
    return configs


def write_recall_prompt(destination: str | Path, server: "WarmRecallServer") -> Path:
    """Write the one-sentence instruction that tells the on arm its memory tool exists.

    Written to a file, and kept in the artifact, so the size of the nudge is auditable. A reader
    who thinks this sentence is doing the work can see the whole of it.
    """

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        RECALL_SYSTEM_PROMPT.format(
            server=server.server_name, tool=f"{server.tool_prefix()}recall_search"
        ),
        encoding="utf-8",
    )
    return target


def write_claude_md_recall_prompt(
    destination: str | Path,
    *,
    static_prompt: str | Path,
    server_name: str,
    tool_prefix: str,
    instruction: str | None = None,
) -> Path:
    """Build the additive arm's single system prompt: the static bundle, then the RE-call sentence.

    `--append-system-prompt-file` takes ONE file, so the two halves have to be concatenated here
    rather than passed separately. The order matters and is deliberate: the static memory first,
    exactly as the `claude_md` arm receives it byte for byte, then the instruction about the tool.
    Keeping the static half identical between the arms is what makes the difference between them
    the memory layer and nothing else, and the artifact keeps the file so that is checkable.

    `instruction` is a template with `{server}` and `{tool}` placeholders; it defaults to the
    one-sentence `RECALL_SYSTEM_PROMPT`. Making it a parameter is what lets an instruction-variant
    run change ONLY this text: the static half and every other arm field stay byte-identical, so a
    difference in behaviour between two runs on the same tasks is attributable to the instruction.
    """

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    static = Path(static_prompt).read_text(encoding="utf-8")
    template = RECALL_SYSTEM_PROMPT if instruction is None else instruction
    instruction = template.format(server=server_name, tool=f"{tool_prefix}recall_search")
    # ⚠️ The instruction goes FIRST, ahead of the static bundle, and this is load-bearing.
    #
    # Appended last it sits after 17,498 characters of CLAUDE.md, and the first additive smoke
    # measured what that does: 16 RE-call tools available, **zero** calls in both sessions, and
    # the arm walked into the omp_threads hazard recommending `taskset`. A tool the agent never
    # invokes measures prompt placement, not retrieval.
    #
    # Chosen on the MECHANISM (a 0% search rate), before any trap outcome was looked at, and it
    # matches what somebody installing a memory layer actually does: the setup line goes at the
    # top of CLAUDE.md, not buried at the bottom. The change is recorded in the preregistration.
    target.write_text(instruction.rstrip() + "\n\n" + static.rstrip() + "\n", encoding="utf-8")
    return target


def write_claude_md_prompt(
    destination: str | Path, sources: tuple[str | Path, ...], *, repo_root: str | Path
) -> Path:
    """Concatenate the static-memory files into one system prompt file, and record what went in.

    The `claude_md` arm's whole point is that it gets the memory a real session would get, so what
    it got has to be reproducible from the artifact rather than reconstructed from a directory
    that has since changed.
    """

    root = Path(repo_root)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    for source in sources:
        path = Path(source)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(f"static memory source is missing: {path}")
        chunks.append(
            f"# --- {path.name} ---\n\n{path.read_text(encoding='utf-8').strip()}\n"
        )
    target.write_text("\n\n".join(chunks), encoding="utf-8")
    return target
