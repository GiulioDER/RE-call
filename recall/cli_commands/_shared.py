"""Helpers shared by more than one command family.

Everything here is used by at least two family modules; a helper with exactly one consumer
lives in that consumer's module instead.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from recall._env import refuse_removed_entailment
from recall.calibration import Calibration
from recall.capabilities import diagnose_exception
from recall.embeddings import Embedder, resolve_embedder
from recall.store import PgVectorStore
from recall.trust import terminal_safe, trusted_search
from recall.trust_policy import TrustPolicy
from recall.types import TrustedResult

#: The published calibration guide. The wheel ships no `docs/`, so a pip user can open only this.
CALIBRATION_DOC_URL = "https://github.com/GiulioDER/RE-call/blob/master/docs/CALIBRATION.md"


def _make_embedder(name: str) -> Embedder:
    """Resolve any spelling `resolve_embedder` accepts, not just the two built-ins.

    The setup wizard offers `st:<model>` and `voyage:<model>`; a hardcoded two-way branch here
    (and a matching argparse `choices=`) rejected exactly the values it had just written to .env,
    so an operator who picked MiniLM or Voyage could not index at all.
    """
    try:
        return resolve_embedder(name)
    except (MemoryError, RecursionError):
        # Not operator mistakes: the process is actually dying. Converting these to a tidy
        # one-liner would hide that, so they propagate like KeyboardInterrupt/SystemExit
        # (which are BaseException, not Exception, and were never caught below regardless).
        raise
    except Exception as exc:  # noqa: BLE001 - see below  # BROAD-CATCH: error-translation
        # Deliberately broad, and an enumerated tuple was tried first and was wrong. The
        # spellings `choices=` used to block reach real constructors: `st:<model>` raises
        # huggingface_hub.RepositoryNotFoundError (an OSError subclass) for a typo, the cloud
        # embedders probe the API in __init__ and re-raise the vendor SDK's own exception
        # (openai.AuthenticationError inherits only from Exception), and an offline box raises
        # httpx errors on the DEFAULT path. Every one of those is an operator mistake and
        # belongs on one line.
        diagnostic = diagnose_exception(
            "embedder", "construction", exc, target=name,
            remediation="check `recall doctor` and install the extra named by the diagnostic",
        )
        raise SystemExit(diagnostic.render()) from exc


def _lite_embedder_spec(
    args: argparse.Namespace, recorded: str | None, db: Path, *, force: bool, how_to_switch: str
) -> tuple[str, bool]:
    """The embedder spec for the lite store `db`, and whether it switches away from the one the
    store recorded (CCA DAT-006, DAT-007).

    A typed `--embedder` wins, then RECALL_EMBEDDER, then `recorded`; "" when there is none. A
    switch re-embeds what the run indexes, leaves other sources on the old vectors, and the
    recalibration rebinds the store, while the MCP server `recall setup --lite` configured still
    names the old embedder. So a RECALL_EMBEDDER left in a shell, which is how a switch happens by
    accident, is refused unless `force`; a typed one is deliberate and only warned about.
    """
    # Only a typed --embedder, before or after the subcommand: the global option's default is
    # RECALL_EMBEDDER or fastembed, which must not count as deliberate (`recall.cli.StoreGiven`).
    typed = getattr(args, "embedder", "") if getattr(args, "embedder_given", False) else ""
    env = os.environ.get("RECALL_EMBEDDER", "").strip()
    spec = typed or env or recorded or ""
    if not recorded or spec == recorded:
        return spec, False
    if not typed and not force:
        raise SystemExit(
            f"RECALL_EMBEDDER={spec!r} differs from {recorded!r}, the embedder {db} was built "
            f"with; {how_to_switch}"
        )
    print(
        f"warning: {db} was built with {recorded!r}; embedding with {spec!r} instead. What this run "
        "indexes is re-embedded, other sources keep the old vectors, and the MCP server "
        f"`recall setup --lite` configured still names {recorded!r} until setup is run again.",
        file=sys.stderr,
    )
    return spec, True


def _lite_read_embedder(args: argparse.Namespace, db: Path, command: str, *, erasure: bool = False) -> str:
    """The embedder a command that only reads or erases the lite store `db` builds (CCA ENV-002).

    A typed `--embedder` wins, then RECALL_EMBEDDER, then the spec the store recorded, and only
    then the global default, which knows nothing of the store. A RECALL_EMBEDDER that differs from
    the recorded spec is refused, as index and refresh refuse it: a model other than the one that
    built the vectors ranks nonsense without raising.

    An `erasure` embeds nothing and needs the embedder for the store's width only, so it takes the
    recorded spec over RECALL_EMBEDDER and refuses nothing: a stray variable must not stop a forget.
    """
    from recall.lite.inspect import lite_file_meta

    recorded = (lite_file_meta(db) or {}).get("embedder_spec")
    typed = getattr(args, "embedder", "") if getattr(args, "embedder_given", False) else ""
    if erasure:
        return typed or recorded or str(args.embedder)
    env = os.environ.get("RECALL_EMBEDDER", "").strip()
    if recorded and not typed and env and env != recorded:
        raise SystemExit(
            f"recall {command}: RECALL_EMBEDDER={env!r} differs from {recorded!r}, the embedder {db} "
            "was built with; unset RECALL_EMBEDDER, or pass --embedder to use another one anyway"
        )
    return typed or env or recorded or str(args.embedder)


def _refuse_unbound_embedder(
    embedder: Embedder, bound: str | None, db: Path, args: argparse.Namespace, *, switched: bool
) -> None:
    """Refuse an embedder other than the model `db`'s calibration is bound to, unless it was typed
    or the switch confirmed (CCA ENV-001).

    The recorded spec alone cannot say what built the vectors: the MCP server refreshes with the
    embedder `recall setup --lite` wrote into its environment and rebinds the calibration
    (`embedder_model`) without touching the spec, so a spec that matches can name another model
    than the store holds, and two commands would then re-embed the store back and forth unseen.
    """
    if bound is None or switched or getattr(args, "embedder_given", False) or embedder.name == bound:
        return
    raise SystemExit(
        f"{db} is bound to the model {bound!r}, not {embedder.name!r}; the MCP server may have "
        "refreshed it with the embedder `recall setup --lite` configured. Run setup again to "
        "choose one, or pass --embedder to switch deliberately"
    )


def _refuse_removed_entailment() -> None:
    """Refuse, as a one line CLI error, a configuration that still asks for the removed judge.

    The decision is `recall._env.refuse_removed_entailment`, shared with MCP server startup so the
    two cannot disagree about which values refuse; this only turns its ValueError into the
    `SystemExit` every other CLI configuration error uses, instead of a traceback.
    """
    try:
        refuse_removed_entailment()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _print_result(result: TrustedResult) -> None:
    flags = []
    if result.abstained:
        flags.append("ABSTAIN")
    if result.gap_warning:
        flags.append("GAP")
    if result.staleness.stale:
        flags.append("STALE")
    if result.trust_state != "trusted":
        # The CLI reaches this BY DEFAULT in development mode: `_cli_trust` synthesises an
        # uncertified threshold, which is exactly the degraded shape that leaves verdicts `ok`.
        # Without this flag a degraded run and a trusted one differed by one boolean buried in
        # the evidence JSON, and the human-readable listing did not differ at all.
        flags.append(f"DEGRADED:{result.failure_code or 'unknown'}")
    print(f"[{' '.join(flags) if flags else 'ok'}] query={result.query!r}")
    # Additive identity line. The same three fields the MCP result and both framework adapters
    # already carry; the CLI was the one surface where an operator could not tell WHICH embedding
    # profile, retrieval profile and index generation produced what they are reading. All three
    # are library- or operator-chosen rather than corpus-chosen, and all three are filtered
    # anyway: a value that reaches a terminal is filtered on the way out, not on the way in.
    d = result.diagnostics
    print(
        f"  index: embedding={terminal_safe(d.embedding_profile)} "
        f"retrieval={terminal_safe(d.retrieval_profile)} "
        f"generation={terminal_safe(d.index_generation)}"
    )
    print(f"  query support: {terminal_safe(result.query_support.status)} "
          "(retrieval eligibility does not establish answerability)")
    coverage = result.calibration_coverage
    print(f"  calibration coverage: {terminal_safe(coverage.status)} "
          f"calibration={terminal_safe(coverage.calibration_id or 'none')}")
    if coverage.report is not None:
        from recall.near_miss import describe

        print("  " + terminal_safe(describe(coverage.report)))
    if result.reason:
        print(f"  reason: {result.reason}")
    for h in result.hits:
        # All three are corpus-controlled and all three are printed to a terminal, which
        # INTERPRETS ANSI escapes rather than showing them — a file name carrying `\x1b[2K\r`
        # erases the line it was printed on. Same class as the `advice` injection, different
        # interpreter. `terminal_safe` filters, so ordinary names render exactly as authored.
        preview = terminal_safe(h.chunk.text).replace("\n", " ")[:52]
        name = terminal_safe(h.provenance.file or h.chunk.source)
        redirect = (
            f" -> use {terminal_safe(h.validity.superseded_by)}" if h.validity.superseded_by else ""
        )
        print(
            f"  {h.verdict:<14} conf={h.confidence:.2f} cos={h.cosine:.3f}  "
            f"{name}{redirect}  {preview!r}"
        )
        # `chunk_id` is the identifier a citation resolves to, so an operator debugging an
        # evidence bundle needs it here.
        #
        # It is filtered and QUOTED, and neither is because the id is known to carry corpus bytes:
        # both minting sites hash `<path>:<ordinal>` into a digest (`recall/index.py`,
        # `recall/generations.py`), and a digest of a hostile name is inert. An earlier version of
        # this comment asserted the opposite — that the id is literally `<file>#<ord>` and so "as
        # corpus-controlled as `name`" — which is false twice over.
        #
        # The treatment stays anyway, for the reason that survives being wrong about the format:
        # `Chunk.id` is whatever the caller constructed, this module does not own the minting
        # scheme, and it cannot assert a property of one it does not own. The `!r` matters
        # independently of ANSI — `terminal_safe` deliberately adds no quotes, so an unquoted
        # value sitting ahead of two library-authored `key=value` fields can forge them, and an
        # id reading `x ordinal=0 valid_from=2099-01-01` would render its own ordinal first.
        valid_from = h.validity.valid_from.isoformat() if h.validity.valid_from else "-"
        print(
            f"                 chunk_id={terminal_safe(h.chunk.id)!r} "
            f"ordinal={h.provenance.ord} valid_from={valid_from}"
        )


def _cli_policy() -> "TrustPolicy":
    """The CLI's trust policy: strict unless `RECALL_TRUST_MODE=development` is set.

    Strict by default even here. A local tool that silently degraded would teach the habit this
    session removes, and the CLI is the surface most people meet first. Setting the variable is a
    deliberate act that shows up in shell history; forgetting to set it produces a refusal that
    names the remedy, which is the better failure.
    """
    from recall.trust_policy import TrustPolicy

    return TrustPolicy.from_env()


def _cli_trust(
    embedder: Embedder,
    calibration: Calibration | None,
    *,
    policy: "TrustPolicy | None" = None,
) -> tuple["TrustPolicy", Calibration | None]:
    """Resolve the CLI's policy, and in development mode announce the threshold it falls back to.

    In development mode with no calibration, every verdict degrades to `unverified` — correct for
    a library caller, but it would stop the CLI demonstrating the trust layer at all (no
    `superseded`, no `ABSTAIN`), which is most of what the CLI is for.

    So it supplies a threshold and SAYS SO on stdout. The number is the same one the library used
    to fall back to invisibly; printing it is the entire difference. A CLI that quietly taught
    "0.50 is the threshold" is what requirement 14 removes, and a printed, explicitly uncertified
    threshold teaches the opposite lesson.

    `policy` lets a caller state its posture instead of reading the environment. It does not
    weaken anything: the announcement below is keyed off `policy.strict`, so a caller that passes
    a development policy gets the uncertified-threshold notice printed exactly as an operator who
    set `RECALL_TRUST_MODE` would. Passing None keeps the environment as the only source.
    """
    policy = policy or _cli_policy()
    if calibration is None and not policy.strict:
        from recall.calibration import Calibration as _Calibration
        from recall.guards import DEFAULT_GAP_THRESHOLD

        calibration = _Calibration(embedder=embedder.name, threshold=DEFAULT_GAP_THRESHOLD)
        print(
            f"[development] using an UNCERTIFIED demonstration threshold of "
            f"{DEFAULT_GAP_THRESHOLD}. This is not a calibration: it is bound to no tenant, "
            f"generation or corpus, and production refuses rather than assuming it."
        )
    return policy, calibration


def _run_queries(
    store: PgVectorStore,
    embedder: Embedder,
    queries: list[str],
    calibration: Calibration | None,
    policy: "TrustPolicy | None" = None,
) -> None:
    """Run each query and print the result.

    `policy` overrides the environment for callers that know their own trust posture. Exactly one
    caller does: `_quickstart` provisions an uncalibrated corpus and is a demonstration by
    definition, so inheriting the strict default means `trusted_search` refuses every one of its
    three queries. See the note at the call site for why that is an override rather than a
    changed default.
    """
    policy, calibration = _cli_trust(embedder, calibration, policy=policy)
    for q in queries:
        _print_result(
            trusted_search(
                store,
                embedder,
                q,
                calibration=calibration,
                policy=policy,
            )
        )
        print()


def _positive_int(value: str) -> int:
    """A count that must be at least 1, refused at parse time.

    `--limit -1` silently sliced the LAST file off the corpus instead of the first, and
    `--limit 0` reported a clean `0 file(s) read` with exit 0, which reads as "this corpus
    states nothing".
    """
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number
