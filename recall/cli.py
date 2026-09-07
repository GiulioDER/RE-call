from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from recall._env import load_dotenv
from recall.calibration import Calibration
from recall.embeddings import resolve_embedder
from recall.entailment import EntailmentJudge, resolve_entailment_judge
from recall.trust_policy import TrustPolicy
from recall.embeddings import Embedder
from recall.observability import configure_logging
from recall.schema import (
    ConcurrentMigrator,
    InterruptedConcurrentIndex,
    MigrationChecksumMismatch,
    SchemaError,
    SchemaIncompatible,
)
from recall.store import (
    DEFAULT_TENANT,
    PgVectorStore,
    _env_opt_out,
    require_secure_dsn,
    warn_if_insecure_dsn,
)
from recall.trust import terminal_safe, trusted_search
from recall.types import TrustedResult
from recall_mcp.translation import provider_from_env, translate_for_display
if TYPE_CHECKING:
    from recall.reasoning import ReasoningResponse
    from recall.reasoning_proposals import InferenceProposal

# `recall setup` writes its answers to .env, so the file has to be read BEFORE the DSN
# defaults below are computed from os.environ. Without this the wizard appears to succeed
# and the very next command silently ignores every setting it just captured.
#
# The failure is RECORDED here rather than acted on: SystemExit is not safe at import time
# (it would kill `import recall.cli` for library consumers), so refusing a command over a
# broken .env has to happen inside `main()`, where it is. Printing a warning here and moving
# on was tried and an audit caught what it misses: warn-and-continue still lets the exact
# hazard through, a request that carries the wrong DSN, it just prints a line first.
_DOTENV_ERROR: Exception | None = None
try:
    load_dotenv()
except Exception as _dotenv_exc:  # noqa: BLE001 - see below  # BROAD-CATCH: fail-open
    # Deliberately broad: this runs at IMPORT time, so anything escaping here kills
    # `recall --help`, every command, and `import recall.cli` for library consumers and test
    # collection. Enumerating types was tried twice and was wrong twice — (OSError,
    # UnicodeDecodeError) missed the ValueError that a NUL byte produces, and a NUL is valid
    # UTF-8 so the read itself succeeds.
    _DOTENV_ERROR = _dotenv_exc
    try:
        print(
            f"warning: .env could not be applied — {type(_dotenv_exc).__name__}: {_dotenv_exc}",
            file=sys.stderr,
        )
    except Exception:  # noqa: BLE001 - this handler must not be able to fail either  # BROAD-CATCH: fail-open
        # A write to a closed or broken stderr (a daemonised or service-wrapped host) must not
        # take an import down. The refusal in `main()` below does not depend on this line
        # having printed; it depends only on `_DOTENV_ERROR` being set.
        pass

DEFAULT_DSN = os.environ.get(
    "RECALL_SERVING_DSN",
    os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall"),
)
DEFAULT_MIGRATION_DSN = os.environ.get("RECALL_MIGRATION_DSN")


def _require_secure(dsn: str) -> None:
    """Indirection so ONE call site decides which DSNs are guarded; see `main`.

    A bug audit proposed converting the PermissionError this raises into a SystemExit, on the
    grounds that every other operator-facing refusal in this file is a SystemExit and this one
    arrives as a traceback. That was REJECTED: `test_cli_db_commands_fail_closed_on_insecure_
    default_dsn` asserts the PermissionError propagates, and it is a security test pinning
    fail-closed behaviour. Rewriting a security assertion to accommodate a cosmetic improvement
    is the wrong trade. The exception type is deliberate; do not "tidy" it.

    Resolving `require_secure_dsn` through the module global at call time is also deliberate:
    that test monkeypatches it, and a `from`-bound local would make the patch inert.
    """
    require_secure_dsn(dsn)


def _make_embedder(name: str) -> Embedder:
    """Resolve any spelling `resolve_embedder` accepts, not just the two built-ins.

    The setup wizard offers `st:<model>` and `voyage:<model>`; a hardcoded two-way branch here
    (and a matching argparse `choices=`) rejected exactly the values it had just written to .env,
    so an operator who picked MiniLM or Voyage could not index at all. A registered local profile
    is selected through `RECALL_EMBED_PROFILE`, which is resolved before the database opens.
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
            "embedder",
            "construction",
            exc,
            target=name,
            remediation="check `recall doctor` and install the extra named by the diagnostic",
        )
        raise SystemExit(diagnostic.render()) from exc


def _entailment_judge(force: bool = False) -> EntailmentJudge | None:
    """Resolve the optional judge, turning a bad env value into a refusal, not a traceback.

    Two defects this exists to prevent, both found by audit:

    * `resolve_entailment_judge` raises ValueError for any RECALL_ENTAILMENT outside its
      true/false sets. Calling it unconditionally on search/demo/code made a TYPO in the .env
      that `recall setup` itself writes traceback out of every search. Before that call was
      added the variable was never read on those paths, so this was a new failure mode.
    * That raise also happened BEFORE the `--entail` fallback could run, so an invalid env
      value disabled an explicit flag. `force` resolves through the same resolver with the
      opt-in overridden, so `--entail` works whatever the env says.
    """
    env = {**os.environ, "RECALL_ENTAILMENT": "1"} if force else None
    try:
        return resolve_entailment_judge(env)
    except (MemoryError, RecursionError):
        raise  # the process is dying, not misconfigured; see _make_embedder above
    except Exception as exc:  # noqa: BLE001 - same reasoning as _make_embedder above  # BROAD-CATCH: error-translation
        # ValueError alone was not enough, and leaving the sibling narrow while broadening
        # `_make_embedder` was inconsistent: `resolve_entailment_judge` CONSTRUCTS the judge,
        # and QnliEntailmentJudge.__init__ eagerly builds a CrossEncoder — so a typo'd
        # RECALL_ENTAILMENT_MODEL raises huggingface's RepositoryNotFoundError (an OSError),
        # and a missing `recall[entail]` extra raises ImportError. Both are operator errors.
        raise SystemExit(f"entailment judge: {type(exc).__name__}: {exc}") from exc


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


def _print_localized_result(result: TrustedResult, locale: str) -> None:
    """Print optional display translations without changing the canonical CLI result."""

    from recall_mcp.translation import normalize_locale

    try:
        normalized = normalize_locale(locale)
        if normalized is None:
            return
        provider = provider_from_env()
        values, translated, warning = translate_for_display(
            [hit.chunk.text for hit in result.hits], normalized, provider
        )
    except ValueError as exc:
        raise SystemExit(f"translation: {exc}") from exc
    print(f"[localized:{normalized} provider={provider.name} translated={translated}]")
    if warning:
        print(f"  warning: {warning}")
    for hit, value in zip(result.hits, values, strict=True):
        print(f"  chunk_id={terminal_safe(hit.chunk.id)!r}  {terminal_safe(value)!r}")


def _print_evidence(
    result: TrustedResult, max_items: int, *, document_mode: bool = False
) -> None:
    """Print the generator-neutral evidence bundle and the exact prompt it renders to.

    JSON, not prose, and that is a safety property rather than a formatting preference. Every
    string here is corpus-controlled, and `json.dumps` escapes control characters — so the ANSI
    payload `terminal_safe` strips from the human-readable listing above arrives as a literal
    `\\u001b` here instead of driving the terminal. The operator sees the byte that is actually in
    their corpus, which is what a debugging surface owes them.

    This is the CLI's whole exposure of `recall.evidence`: the bundle a generator would be given,
    plus `system` and `user` exactly as `render_evidence_prompt` produces them, so an operator can
    inspect the boundary without writing a program against the library.
    """
    from recall.evidence import EvidencePolicy, build_evidence_bundle, render_evidence_prompt

    # `max(1, ...)`: `-k` has no lower bound, and `EvidencePolicy` refuses `max_items < 1`, so
    # `recall search q -k 0 --evidence` raised an uncaught ValueError out of a dataclass
    # constructor. A CLI flag combination must not produce a traceback.
    bundle = build_evidence_bundle(
        result,
        EvidencePolicy(
            max_items=max(1, max_items),
            bundle_mode="document" if document_mode else "retrieval",
        ),
    )
    system, user = render_evidence_prompt(bundle)
    payload = {
        "bundle": asdict(bundle),
        "prompt": {"system": system, "user": user},
    }
    print(json.dumps(payload, indent=2, default=str))


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
    embedder: Embedder, calibration: Calibration | None
) -> tuple["TrustPolicy", Calibration | None]:
    """Resolve the CLI's policy, and in development mode announce the threshold it falls back to.

    In development mode with no calibration, every verdict degrades to `unverified` — correct for
    a library caller, but it would stop the CLI demonstrating the trust layer at all (no
    `superseded`, no `ABSTAIN`), which is most of what the CLI is for.

    So it supplies a threshold and SAYS SO on stdout. The number is the same one the library used
    to fall back to invisibly; printing it is the entire difference. A CLI that quietly taught
    "0.50 is the threshold" is what requirement 14 removes, and a printed, explicitly uncertified
    threshold teaches the opposite lesson.
    """
    policy = _cli_policy()
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


def _refuse_untrusted_reasoning_inspection(trust_state: str, policy: "TrustPolicy") -> None:
    if trust_state != "trusted" and policy.strict:
        raise SystemExit(
            "reasoning inspection refused in strict mode: generation identity or calibration is "
            "missing. Set RECALL_TRUST_MODE=development to inspect degraded artifacts."
        )


def _reasoning_trace_export(response: "ReasoningResponse") -> dict[str, object]:
    trace = response.to_dict()["reasoning_trace"]
    if trace is None:
        reason = (
            response.refusal_reason or response.trusted_evidence.failure_code or response.outcome
        )
        raise SystemExit(f"reasoning trace unavailable: {reason}")
    assert isinstance(trace, dict)
    initial = trace.get("initial_retrieval")
    if isinstance(initial, dict):
        initial.pop("reason", None)
    return trace


def _run_queries(
    store: PgVectorStore,
    embedder: Embedder,
    queries: list[str],
    calibration: Calibration | None,
    entailment: EntailmentJudge | None = None,
) -> None:
    policy, calibration = _cli_trust(embedder, calibration)
    for q in queries:
        _print_result(
            trusted_search(
                store,
                embedder,
                q,
                calibration=calibration,
                policy=policy,
                entailment=entailment,
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


def _non_negative_int(value: str) -> int:
    """A count where zero is meaningful but a negative one is not.

    `--overlap` is the case: 0 means "no shared context between adjacent pieces", which is a real
    choice, while a negative value is written verbatim into an immutable lineage record describing
    a pipeline nothing can have run.
    """
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if number < 0:
        raise argparse.ArgumentTypeError(f"cannot be negative, got {number}")
    return number


def _parse_status_vocabulary(raw: str | None) -> tuple[str, ...] | None:
    """Parse `--status-vocabulary`. `None` means the shipped memo set.

    The split is the flag's SHAPE; every judgement about the result belongs to
    `coerce_status_vocabulary`, which already refuses an empty list, a bare string, blank and
    non-str entries and casefold collisions, and which strips. Deliberately NOT the labelling
    runner's `tuple(v.strip() for v in raw.split(",") if v.strip())`: that comprehension swallows
    exactly the blanks the coercion exists to refuse, so `Final,,Rejected` would pass quietly and
    `,` would collapse to an empty vocabulary — which refuses every status claim at a BATCH rung,
    the original defect re-entered through the flag added to remove it.
    """
    if raw is None:
        return None
    from recall.truth_extraction.types import coerce_status_vocabulary

    try:
        return coerce_status_vocabulary(raw.split(","))
    except ValueError as exc:
        print(f"recall extract: --status-vocabulary: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _rejected_claims(ledger_path: Path) -> frozenset[str]:
    """Every claim key a human has rejected, read without creating the ledger.

    Returns an empty set when the sidecar does not exist: nothing has been rejected yet. A
    ledger that exists but cannot be read is a different matter and propagates, because it
    might hold a rejection for the very claim being asked about.
    """
    import sqlite3

    if not ledger_path.exists():
        return frozenset()
    try:
        with sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True) as conn:
            return frozenset(
                row[0] for row in conn.execute("SELECT claim_key FROM rejected_claims")
            )
    except sqlite3.Error as exc:
        from recall.rewrite import RewriteRefused

        raise RewriteRefused(
            f"rejection ledger at {ledger_path} could not be read: {exc}"
        ) from exc


def _already_declared(root: Path, proposal: "InferenceProposal") -> bool:
    """True when the memo already states this proposal's key, so it needs no second review."""
    from recall.document import parse_document
    from recall.rewrite import RewriteRefused, _resolve, destination, route_relation

    try:
        routed = route_relation(
            proposal.proposed_relation,
            proposal.subject_id,
            proposal.object_id,
        )
        if destination(routed.key) != "frontmatter":
            # A derived-block key is multi valued for `contradicts` and `same_entity`, so
            # "already there" is not a property of the key alone. Left to the write path.
            return False
        corpus_root = root if root.is_dir() else root.parent
        path = corpus_root / routed.edit_file
        if not path.is_file():
            # Falls back to the write path's OWN resolution, for the corpus names that are not
            # usable relative paths. A memo whose filename is not valid UTF-8 is named by the
            # stand-in `encodable_name` returns, so the join above opens nothing and every run
            # re-offered an edge that memo already declares as unreviewed work. A queue that
            # never converges is the defect DECLARED exists to prevent. The join stays first
            # because this runs once per proposal and `_resolve` walks the whole corpus.
            path = corpus_root / _resolve(corpus_root, routed.edit_file)
        meta = parse_document(path.read_text(encoding="utf-8-sig")).meta
    except (RewriteRefused, UnicodeDecodeError, OSError):
        return False
    return routed.key in meta


def _run_rewrite(args: argparse.Namespace) -> None:
    """`recall rewrite plan|apply|reject|verify`. Filesystem only; never opens the database.

    Proposals are re-derived from the corpus on every verb rather than stored, because a
    proposal is a reading of the corpus as it stands now. See `rewrite.corpus_proposals`.
    """
    from datetime import datetime, timezone

    from recall.document import parse_document
    from recall.frontmatter import supersedes_key
    from recall.promotion import (
        accept_reviewed_proposal,
        promote_accepted_proposal,
        review_proposal,
    )
    from recall.rewrite import (
        RejectionLedger,
        apply_rewrite,
        claim_key,
        corpus_proposals,
        default_ledger_path,
        plan_rewrite,
    )

    root = Path(args.path)
    if not root.exists():
        print(f"recall rewrite: no such path: {root}", file=sys.stderr)
        raise SystemExit(2)

    if args.rewrite_cmd == "verify":
        # The check `recall lint` makes, scoped to what this command writes: a declared edge
        # whose target does not resolve is the defect `supersedes_key` exists for, and it is
        # exactly what a bad rewrite would leave behind.
        corpus_root = root if root.is_dir() else root.parent
        try:
            corpus_paths = sorted(corpus_root.glob(args.glob))
        except (ValueError, NotImplementedError, OSError) as exc:
            print(f"recall rewrite: --glob {args.glob!r}: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        # Checking ONE memo still resolves against the corpus it lives in; scoping the corpus
        # to the single file reported every edge that does resolve as unresolved.
        paths = corpus_paths if root.is_dir() else [root]

        def _rel(path: Path) -> str:
            try:
                return path.relative_to(corpus_root).as_posix()
            except ValueError:
                return path.name

        # A LIST per key, not a single name. Overwriting collapsed `legal/old.md` and
        # `eng/old.md` onto one entry, so an edge naming `old.md` was reported as resolved
        # when it resolves to two files and therefore to none: `lint` calls that
        # `ambiguous-supersedes-target` and `_resolve` refuses to write it.
        by_key: dict[str, list[str]] = {}
        for path in corpus_paths:
            if path.is_file():
                by_key.setdefault(supersedes_key(path.name), []).append(_rel(path))

        unresolved = 0
        for path in paths:
            if not path.is_file():
                continue
            try:
                meta = parse_document(path.read_text(encoding="utf-8-sig")).meta
            except (UnicodeDecodeError, OSError) as exc:
                print(f"  UNREADABLE {_rel(path)}: {exc}")
                continue
            target = meta.get("supersedes")
            if not isinstance(target, str) or not target:
                continue
            matches = by_key.get(supersedes_key(target), [])
            if not matches:
                print(
                    f"  UNRESOLVED {_rel(path)}: supersedes {target!r}, "
                    f"which is not in the corpus"
                )
                unresolved += 1
            elif len(matches) > 1:
                print(
                    f"  AMBIGUOUS {_rel(path)}: supersedes {target!r}, which matches "
                    f"{len(matches)} files: {', '.join(matches)}"
                )
                unresolved += 1
        print(f"\n{unresolved} unresolved edge(s)")
        if unresolved:
            raise SystemExit(1)
        return

    # `RewriteRefused` is caught by the dispatcher in `main`, which turns it into the same
    # `recall rewrite: <reason>` and exit 2. Catching it again here only risked the two
    # disagreeing, and the local handler referenced a name this function never imported.
    proposals = corpus_proposals(root, args.glob)
    ledger_path = default_ledger_path(root if root.is_dir() else root.parent)

    if args.rewrite_cmd == "plan":
        # Read WITHOUT creating the ledger. Opening it for write made `plan` create
        # `<root>/.recall/rejections.sqlite3` while printing "nothing written", and made the
        # command fail outright on a corpus the user cannot write to. A plan is a listing, not
        # a decision, so a missing ledger simply means nothing has been rejected yet.
        rejected = _rejected_claims(ledger_path)
        for proposal in proposals:
            claim = claim_key(
                proposal.proposed_relation, proposal.subject_id, proposal.object_id
            )
            if claim in rejected:
                mark = "REJECTED"
            elif _already_declared(root, proposal):
                # Otherwise an accepted proposal reappears every run, indistinguishable from
                # unreviewed work. It needs no storage: the corpus file already states it.
                mark = "DECLARED"
            else:
                mark = "review  "
            print(f"  {mark} {proposal.id}  {proposal.proposed_relation}")
            print(f"      {proposal.subject_id} -> {proposal.object_id}")
            print(f"      {proposal.explanation}")
            # Printed so a plan from THIS command and a plan from `recall_rewrite_plan` over
            # MCP name the same thing. Their proposal ids never match; their claim keys do.
            print(f"      claim {claim}")
        print(f"\n{len(proposals)} proposal(s)")
        print("dry run — nothing written. Declare one with `recall rewrite apply`.")
        return

    wanted = getattr(args, "claim", None)
    if wanted:
        chosen = next(
            (
                p
                for p in proposals
                if claim_key(p.proposed_relation, p.subject_id, p.object_id) == wanted
            ),
            None,
        )
    else:
        chosen = next((p for p in proposals if p.id == args.proposal), None)
    if chosen is None:
        given = f"claim {wanted!r}" if wanted else f"proposal {args.proposal!r}"
        print(
            f"recall rewrite: no {given} in {root}. "
            f"Run `recall rewrite plan {root}` to list them.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    now = datetime.now(timezone.utc)
    claim = claim_key(chosen.proposed_relation, chosen.subject_id, chosen.object_id)

    if args.rewrite_cmd == "reject":
        with RejectionLedger(ledger_path) as ledger:
            ledger.reject(
                claim, reviewer_id=args.reviewer, reason=args.note, rejected_at=now
            )
        print(f"recorded {args.reviewer}'s rejection of {chosen.id} as claim {claim}")
        return

    reviewed = review_proposal(
        chosen, reviewer_id=args.reviewer, reviewed_at=now, audit_note=args.note
    )
    fact = promote_accepted_proposal(accept_reviewed_proposal(reviewed), promoted_at=now)
    plan = plan_rewrite(root, fact)
    print(f"  {plan.edit_file}: + {plan.key}: {plan.value}  (in the {plan.block} block)")
    if not args.apply:
        # Dry run by DEFAULT: this edits the user's own documents, and a tool that rewrites
        # your memory the first time you try it has earned distrust.
        #
        # The ledger is consulted HERE, not only under --apply. Skipping it meant the preview
        # of an already rejected claim printed "Re-run with --apply to write this edge", and
        # --apply then refused it: a preview that promises a write the real run declines.
        if claim in _rejected_claims(ledger_path):
            print(f"not written: claim {claim} was already rejected by a reviewer")
            raise SystemExit(1)
        print("dry run — nothing written. Re-run with --apply to write this edge.")
        return
    with RejectionLedger(ledger_path) as ledger:
        result = apply_rewrite(root, fact, ledger=ledger, apply=True)
    if result.written:
        print("written")
        return
    # Non-zero when nothing was written. Exiting 0 on a refusal left a script unable to tell a
    # completed declaration from a declined one. 2 stays reserved for caller error.
    print(f"not written: {result.refusal}")
    raise SystemExit(1)


def _run_extract(args: argparse.Namespace) -> None:
    """`recall extract run|show`. Reads the corpus, writes nothing, never opens the database.

    Extraction is OFF unless `RECALL_TRUTH_EXTRACTION` is set, mirroring `entailment.py`, and an
    unknown engine name is refused rather than downgraded to the deterministic one: silently
    running a different engine than the one named would make the audit record wrong about how a
    claim was produced.
    """
    from recall.frontmatter import encodable_name
    from recall.truth_extraction import resolve_extraction_engine
    from recall.truth_extraction.extract import extract_corpus_claims_for_report

    try:
        engine = resolve_extraction_engine()
    except (ValueError, ImportError) as exc:
        print(f"recall extract: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if engine is None:
        print(
            "recall extract: extraction is off. Set RECALL_TRUTH_EXTRACTION=1 to enable it. "
            "See docs/TRUTH_EXTRACTION_DESIGN.md for what it does and what it refuses.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    # Parsed BEFORE the path check and before the glob, because it depends on nothing about the
    # corpus. The sibling `--recheck` check below carries the same reasoning from the other
    # direction: a flag error knowable at parse time must not arrive after a corpus of model
    # calls has been paid for, under a complete-looking report.
    status_vocabulary = _parse_status_vocabulary(args.status_vocabulary)

    root = Path(args.path if args.extract_cmd == "run" else args.file)
    if not root.exists():
        # Refused, not reported as an empty corpus. `0 claim(s) for review` on a typo reads as
        # "this corpus states nothing", which is the opposite of "I never looked". `lint` and
        # `check` both exit 2 here; this matches them.
        print(f"recall extract: no such path: {root}", file=sys.stderr)
        raise SystemExit(2)
    if args.extract_cmd == "show" and not root.is_file():
        print(f"recall extract show: not a file: {root}", file=sys.stderr)
        raise SystemExit(2)

    def _glob(directory: Path) -> list[Path]:
        # `Path.glob` raises ValueError on an empty pattern and NotImplementedError on an
        # absolute one. Neither is an OSError, so both escaped as a traceback.
        try:
            return sorted(directory.glob(args.glob))
        except (ValueError, NotImplementedError, OSError) as exc:
            print(f"recall extract: --glob {args.glob!r}: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc

    if args.extract_cmd == "run":
        # The CORPUS is always the full glob. Slicing it with `--limit` shrinks the set a
        # supersession target resolves against, so a memo naming a real neighbour is refused
        # with "which is not a file in the corpus" about a file sitting right beside it. That
        # fabricated refusal reads exactly like a real one, and `--limit` is a sampling flag
        # whose entire purpose is to look at part of a corpus without changing the answers.
        corpus_paths = _glob(root) if root.is_dir() else _glob(root.parent)
        paths = corpus_paths if root.is_dir() else [root]
        if args.limit is not None:
            paths = paths[: args.limit]
    else:
        # `show` REPORTS one file but resolves targets against the corpus that file lives in,
        # for the same reason.
        paths = [root]
        corpus_paths = _glob(root.parent)

    corpus_root = root if root.is_dir() else root.parent

    def _key(path: Path) -> str:
        # Keyed by path relative to the corpus root, not by bare basename. The default glob is
        # recursive, so `legal/policy.md` and `eng/policy.md` collapsed onto one dict key: one
        # file was silently dropped, and the ladder's "matches N files in the corpus" refusal
        # could never fire, because the index had deduplicated the ambiguity away.
        #
        # `encodable_name` for the same reason it is in `rewrite._key`: a filename that is not
        # valid UTF-8 is a lone surrogate here, and it flows into the prompt, the cache key and
        # the report. Each of those was hardened separately; the boundary is one place.
        try:
            return encodable_name(path.relative_to(corpus_root).as_posix())
        except ValueError:
            return encodable_name(path.name)

    documents: dict[str, str] = {}
    unreadable: list[tuple[str, str]] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            documents[_key(path)] = path.read_text(encoding="utf-8-sig")
        except (UnicodeDecodeError, OSError) as exc:
            # Per file, and note `UnicodeDecodeError` is a ValueError, NOT an OSError. One memo
            # that is not UTF-8 aborted the whole run and discarded every file already decoded.
            # `lint.py` and `fix.py` both catch the pair and keep going.
            unreadable.append((_key(path), str(exc)))
    corpus_names = tuple(sorted(_key(p) for p in corpus_paths if p.is_file()))

    # Validated BEFORE any extraction runs. Deferring it meant a user paid for a whole corpus of
    # model calls and then got exit 2 on a flag combination knowable at parse time, with a
    # complete looking report already printed above the error.
    if getattr(args, "recheck", False) and getattr(args, "cache", None) is None:
        print(
            "recall extract: --recheck needs --cache PATH; there is nothing to re-check "
            "against an empty cache",
            file=sys.stderr,
        )
        raise SystemExit(2)

    cache = None
    if getattr(args, "cache", None) is not None:
        from recall.truth_extraction._sqlite_cache import (
            ExtractionCacheRefused,
            SqliteExtractionCache,
        )

        try:
            cache = SqliteExtractionCache(args.cache)
        except ExtractionCacheRefused as exc:
            # Refused at OPEN, before a single memo is read, because the alternative is
            # discovering halfway through a corpus that the path was somebody else's database.
            print(f"recall extract: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc

    try:
        extractions = extract_corpus_claims_for_report(
            documents,
            engine=engine,
            corpus_names=corpus_names,
            cache=cache,
            status_vocabulary=status_vocabulary,
        )
        for name, reason in unreadable:
            print(f"  UNREADABLE {name}: {reason}")
        for item in extractions:
            for claim in item.claims:
                print(f"  {item.file}: {claim.kind}")
                print(f"      quote {claim.quote!r}")
            for refusal in item.rejections:
                print(f"  SKIP {item.file}: {refusal.rung}: {refusal.reason}")
            if item.batch_rejection is not None:
                print(
                    f"  REFUSED {item.file}: {item.batch_rejection.rung}: "
                    f"{item.batch_rejection.reason}"
                )
        total = sum(len(item.claims) for item in extractions)
        print(f"\n{len(documents)} file(s) read, {total} claim(s) for review")
        if status_vocabulary is not None:
            # Named only when it is not the default: two runs whose outputs differ have nothing
            # on screen saying why otherwise. The parenthetical is why: this report is the ONLY
            # place a custom vocabulary is honoured. `recall rewrite` extracts under the shipped
            # set regardless, so a status claim printed above can vanish from `rewrite plan`
            # with nothing else on screen to explain it, right after this command's own closing
            # line says "review with `recall rewrite plan`".
            print(
                f"status vocabulary: {', '.join(status_vocabulary)} "
                "(measurement only; recall rewrite still writes the shipped set)"
            )

        if cache is not None and getattr(args, "recheck", False):
            from recall.truth_extraction._cache import recheck_cached_extractions

            # The SAME `corpus_names` the extraction ran with. `extraction_cache_key` hashes them,
            # so passing the document keys instead produced a different key for every file, every
            # lookup missed, and the report read "0 checked, rate not measured" without saying why.
            report = recheck_cached_extractions(
                documents,
                engine=engine,
                corpus_names=corpus_names,
                cache=cache,
                # The SAME vocabulary the extraction ran with, for the same reason as
                # `corpus_names`: it is in the cache key AND the prompt. Omitted, every lookup
                # misses and the report reads `checked=0` — a determinism measurement that
                # silently became a non-measurement.
                status_vocabulary=status_vocabulary,
            )
            rate = "not measured" if report.mismatch_rate is None else f"{report.mismatch_rate:.3f}"
            print(
                f"recheck: {report.checked} checked, {report.mismatched} mismatched, "
                f"{report.errored} errored, rate {rate}"
            )

        # Dry run is the ONLY run. This command has no --apply, because declaring a claim needs a
        # named human at `recall rewrite apply`, not a flag here.
        print("nothing written — review with `recall rewrite plan`")
    finally:
        if cache is not None:
            # try/finally, so the counters are REPORTED and the sqlite connection closed on
            # every exit path. Doing it only after the last print meant any exception in
            # extraction, printing or recheck leaked the connection and, worse, skipped the
            # one line that makes a silently degraded cache visible.
            if cache.write_failures or cache.corrupt or cache.stale:
                # Read directly, not through getattr with a default: a renamed counter must
                # break a test, not quietly report a degraded run as clean.
                print(
                    f"cache: {cache.write_failures} write failure(s), "
                    f"{cache.corrupt} unusable, {cache.stale} from an older cache version"
                )
            cache.close()


_COMMAND_REGISTRATIONS: tuple[tuple[frozenset[str], str, str], ...] = (
    (frozenset({"doctor"}), "recall.cli_commands.doctor_cmd", "register"),
    (
        frozenset({"setup", "wizard", "uninstall"}),
        "recall.cli_commands.setup_wizard",
        "register",
    ),
    (frozenset({"quickstart"}), "recall.cli_commands.setup_wizard", "register_quickstart"),
    (frozenset({"schema"}), "recall.cli_commands.schema_cmd", "register"),
    (frozenset({"manifest"}), "recall.cli_commands.manifest_cmd", "register"),
    (frozenset({"generation"}), "recall.cli_commands.generation_cmd", "register"),
    (frozenset({"graph"}), "recall.cli_commands.graph_cmd", "register"),
    (
        frozenset({"index", "forget", "search", "scopes"}),
        "recall.cli_commands.index_search",
        "register",
    ),
    (frozenset({"reasoning"}), "recall.cli_commands.reasoning_cmd", "register"),
    (frozenset({"extract", "rewrite"}), "recall.cli_commands.extract_rewrite", "register"),
    (frozenset({"demo", "code"}), "recall.cli_commands.index_search", "register_demo_code"),
    (frozenset({"lint", "check"}), "recall.cli_commands.lint_check", "register"),
    (frozenset({"calibration"}), "recall.cli_commands.calibration_cmd", "register"),
    (frozenset({"provenance"}), "recall.cli_commands.provenance_cmd", "register"),
    (frozenset({"backup"}), "recall.cli_commands.backup_cmd", "register"),
    (frozenset({"secret"}), "recall.cli_commands.secret_cmd", "register"),
)


def build_parser(command: str | None = None) -> argparse.ArgumentParser:
    """Build the command tree without opening a database or resolving providers.

    Passing a command limits imports to the module that registers that command. The default
    builds the complete tree for API introspection and top level help.
    """

    parser = argparse.ArgumentParser(
        prog="recall",
        description="Retrieval-augmented memory for long-running agents.",
        epilog=(
            "Starting out? `recall quickstart` demonstrates the system; `recall setup` is THE "
            "install; `recall wizard` is the saved-config workflow; `recall doctor` diagnoses "
            "an existing install."
        ),
    )
    parser.add_argument(
        "--serving-dsn",
        "--dsn",
        dest="dsn",
        default=DEFAULT_DSN,
        help="unprivileged application DSN (env: RECALL_SERVING_DSN; --dsn is deprecated)",
    )
    parser.add_argument(
        "--migration-dsn",
        default=DEFAULT_MIGRATION_DSN,
        help="DDL-owner DSN used only by `schema apply` (env: RECALL_MIGRATION_DSN)",
    )
    parser.add_argument(
        "--embedder",
        default=os.environ.get("RECALL_EMBEDDER", "fastembed"),
        help=(
            "hashing, fastembed[:model], st:<model>, voyage[:model], openai[:model]. "
            "Set RECALL_EMBED_PROFILE for a registered profile such as "
            "bge-small-context-section-v1 or bge-large-context-section-v1."
        ),
    )
    parser.add_argument(
        "--table",
        default="chunks",
        help="table to read/write (default: chunks). Use a throwaway name to keep an experiment out of your real memory index.",
    )
    parser.add_argument(
        "--tenant",
        default=DEFAULT_TENANT,
        help=f"tenant namespace to operate on (default: {DEFAULT_TENANT}).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # These modules own the complete argument declarations and handlers. Keeping registration in
    # one place prevents the executable parser and the API introspection parser from drifting.
    # The command index is only an import optimization. An unknown command falls back to the full
    # registry so adding a command cannot silently make the executable undiscoverable.
    registrations = (
        _COMMAND_REGISTRATIONS
        if command is None or not any(command in commands for commands, _, _ in _COMMAND_REGISTRATIONS)
        else tuple(item for item in _COMMAND_REGISTRATIONS if command in item[0])
    )
    for _, module_name, function_name in registrations:
        module = importlib.import_module(module_name)
        getattr(module, function_name)(sub)
    return parser


_GLOBAL_OPTIONS_WITH_VALUES = frozenset(
    {"--serving-dsn", "--dsn", "--migration-dsn", "--embedder", "--table", "--tenant"}
)


def _command_from_argv(argv: list[str]) -> str | None:
    """Find the top level command without importing command modules."""

    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in _GLOBAL_OPTIONS_WITH_VALUES:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        return token
    return None


_SCHEMA_REMEDY: dict[type[SchemaError], str] = {
    SchemaIncompatible: (
        "This table was created for a different embedder or RE-call version. Pass the matching "
        "--embedder or use a different --table."
    ),
    MigrationChecksumMismatch: (
        "A migration file no longer matches the bytes recorded as applied. Restore the committed "
        "file or use a reviewed upgrade; do not edit applied migration history."
    ),
    ConcurrentMigrator: "Another migrator holds the lock. Wait for it to finish, then retry.",
    InterruptedConcurrentIndex: (
        "A concurrently built index was left invalid. Drop the named index and re-run "
        "`recall schema apply`."
    ),
}


def schema_error_message(exc: SchemaError) -> str:
    remedy = _SCHEMA_REMEDY.get(type(exc))
    return str(exc) if remedy is None else f"{exc}\n\n{remedy}"


def _main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):  # clean UTF-8 output on Windows consoles
        # `errors=` as well as `encoding=`, because reconfiguring the encoding RESETS errors to
        # strict. The inherited handler is surrogateescape, and dropping it made every `print`
        # of a filename raise for a name that is not valid UTF-8: `recall extract run` over a
        # corpus holding one such file exited 1 with EMPTY stdout, throwing away a completed
        # extraction at the REPORT step. That is the same "one bad memo kills the run" failure
        # the extractor guards against everywhere else, arriving at the last possible moment.
        # Showing a mangled name beats showing nothing.
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        # For the ENCODING, not the error handler, and the first version of this comment had it
        # wrong: CPython already defaults stderr to `backslashreplace`, and keeps it there even
        # under `PYTHONIOENCODING=utf-8:strict`, which sets stdout to strict alone. So deleting
        # this line would not turn a refusal into a traceback. What it does is give stderr the
        # same UTF-8 encoding stdout gets on a Windows console, and hold the handler if a
        # caller has replaced stderr with a strict wrapper of its own.
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    # Without this the library's loggers have no handler, so every _log.info is discarded — which
    # is how `index` came to prune rows while printing nothing about it.
    configure_logging()
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser(_command_from_argv(raw_argv))
    args = parser.parse_args(raw_argv)
    # Commands that will actually open a connection FAIL CLOSED on the insecure default DSN;
    # everything else only warns.
    #
    # Every command that will open a connection FAILS CLOSED on the insecure default DSN;
    # the rest only warn.
    #
    # An earlier version of this set listed six commands by hand and missed four that connect
    # (generation, calibration, schema, and lint --semantic), so the guard read as coverage and
    # was not. The set is derived from the parsers now: a subcommand declares `_opens_db=True`
    # beside its own definition, so a new one cannot be added without answering the question.
    opens_db = bool(getattr(args, "_opens_db", False))
    if args.cmd == "provenance" and getattr(args, "sqlite_path", None):
        # The local provenance adapter is deliberately independent of PostgreSQL and the
        # embedder. It still revalidates source bytes below, so this is not a trust bypass.
        opens_db = False
    if args.cmd == "lint":  # only the --semantic path reaches a database
        opens_db = bool(getattr(args, "semantic", False))
    if args.cmd == "schema" and getattr(args, "schema_cmd", None) == "grants":
        opens_db = False  # prints SQL for an operator to run; opens nothing

    if (
        opens_db
        and args.cmd not in {"setup", "wizard"}  # see the setup-specific carve-out for _require_secure below —
        # `recall setup` is the command you run to REPAIR a broken .env, so blocking it on a
        # broken .env is the same dead end that carve-out exists to avoid, one guard down. A
        # round-6 audit caught this: it fired unconditionally and refused `setup` even when the
        # operator had already passed an explicit --dsn that resolved the ambiguity.
        #
        # `setup` is not left silent: the note comes from the import-time stderr print above
        # (near `_DOTENV_ERROR = _dotenv_exc`), which runs for every command before args.cmd is
        # even known — NOT from run_setup_wizard, which has no .env-specific messaging of its
        # own. A round-7 audit caught an earlier version of this comment misattributing it,
        # which is worth naming: believing the notice were conditional on reaching the wizard
        # could lead a later change to gate or remove the import-time print, leaving `setup`
        # with zero indication anything was wrong.
        and _DOTENV_ERROR is not None
        and not _env_opt_out("RECALL_IGNORE_BROKEN_DOTENV")
    ):
        # `.env` exists but could not be applied, so any variable it would have set — most
        # dangerously RECALL_SERVING_DSN — is silently absent from this process, and args.dsn
        # below is the LOCAL fallback rather than whatever was configured. Warning about that
        # at import time and proceeding anyway was tried; it still lets a request reach the
        # wrong database, which is the exact hazard this whole guard exists to prevent, so a
        # DB-opening command refuses instead. Reading it, fixing it, or deleting it are all
        # legitimate; running against a database neither the operator nor the file chose is not.
        raise SystemExit(
            f".env exists but could not be applied "
            f"({type(_DOTENV_ERROR).__name__}: {_DOTENV_ERROR}), and this command connects to a "
            f"database. Fix the file, or set RECALL_IGNORE_BROKEN_DOTENV=1 to proceed anyway — "
            f"variables the file would have set (including RECALL_SERVING_DSN) are absent, so "
            f"the DSN in effect may not be the one you intended."
        )

    if opens_db:
        if args.cmd == "setup":
            # The wizard is the command you run to REPAIR a bad configuration, so a bare
            # refusal is a dead end: it takes `dsn=args.dsn` verbatim and never prompts for
            # one. Still guarded, because it does connect when the operator accepts the
            # calibrate prompt, and also when the operator accepts the CLAUDE.md/memory
            # scaffold prompt (which defaults to yes and auto-indexes memory/) — but the
            # refusal has to name the way out.
            try:
                _require_secure(args.dsn)
            except PermissionError as exc:
                raise SystemExit(
                    f"{exc}\n\n"
                    "This is `recall setup`, which cannot prompt its way out of this: it uses "
                    "the DSN it was given and never asks for another. Passing that same value "
                    "again with `--dsn` or `--serving-dsn` does not help, because the refusal "
                    "is about the credentials inside the DSN, not about how it reached the "
                    "command. Re-run with a DSN carrying a real password, or set "
                    "RECALL_ALLOW_INSECURE_DSN=1 to accept the risk deliberately."
                ) from exc
        elif args.cmd != "wizard":
            _require_secure(args.dsn)
    else:
        warn_if_insecure_dsn(args.dsn)  # loud stderr note if default creds target a remote host

    # The DDL-owner credential was never checked or even warned about on any path, which is the
    # wrong way round: it is the most privileged DSN this CLI accepts.
    migration_dsn = getattr(args, "migration_dsn", None)
    if migration_dsn and opens_db:  # grants stays exempt because it does not open a database
        _require_secure(migration_dsn)

    handler = getattr(args, "func", None)
    if handler is None:
        raise SystemExit(f"no handler registered for command: {args.cmd}")
    handler(args)


def main(argv: list[str] | None = None) -> None:
    try:
        _main(argv)
    except SchemaError as exc:
        raise SystemExit(schema_error_message(exc)) from exc


if __name__ == "__main__":
    main()
