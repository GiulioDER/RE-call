from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from recall._env import load_dotenv
from recall.calibration import Calibration, load_for
from recall.embeddings import resolve_embedder
from recall.entailment import EntailmentJudge, resolve_entailment_judge
from recall.setup import CalibrationResult
from recall.trust_policy import TrustPolicy
from recall.embeddings import Embedder, HashingEmbedder
from recall.index import Indexer, PruneGuardTripped, chunk_code, chunk_text
from recall.lint import DEFAULT_GLOB
from recall.observability import configure_logging
from recall.store import (
    DEFAULT_TENANT,
    PgVectorStore,
    _env_opt_out,
    require_secure_dsn,
    warn_if_insecure_dsn,
)
from recall.trust import terminal_safe, trusted_search
from recall.types import TrustedResult

if TYPE_CHECKING:
    from recall.reasoning import ReasoningResponse

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
except Exception as _dotenv_exc:  # noqa: BLE001 - see below
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
    except Exception:  # noqa: BLE001 - this handler must not be able to fail either
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
    so an operator who picked MiniLM or Voyage could not index at all.
    """
    try:
        return resolve_embedder(name)
    except (MemoryError, RecursionError):
        # Not operator mistakes: the process is actually dying. Converting these to a tidy
        # one-liner would hide that, so they propagate like KeyboardInterrupt/SystemExit
        # (which are BaseException, not Exception, and were never caught below regardless).
        raise
    except Exception as exc:  # noqa: BLE001 - see below
        # Deliberately broad, and an enumerated tuple was tried first and was wrong. The
        # spellings `choices=` used to block reach real constructors: `st:<model>` raises
        # huggingface_hub.RepositoryNotFoundError (an OSError subclass) for a typo, the cloud
        # embedders probe the API in __init__ and re-raise the vendor SDK's own exception
        # (openai.AuthenticationError inherits only from Exception), and an offline box raises
        # httpx errors on the DEFAULT path. Every one of those is an operator mistake and
        # belongs on one line.
        raise SystemExit(f"embedder {name!r}: {type(exc).__name__}: {exc}") from exc


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
    except Exception as exc:  # noqa: BLE001 - same reasoning as _make_embedder above
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


def _print_evidence(result: TrustedResult, max_items: int) -> None:
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
    bundle = build_evidence_bundle(result, EvidencePolicy(max_items=max(1, max_items)))
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


def _already_declared(root: Path, proposal: object) -> bool:
    """True when the memo already states this proposal's key, so it needs no second review."""
    from recall.document import parse_document
    from recall.rewrite import RewriteRefused, _resolve, destination, route_relation

    try:
        routed = route_relation(
            proposal.proposed_relation,  # type: ignore[attr-defined]
            proposal.subject_id,  # type: ignore[attr-defined]
            proposal.object_id,  # type: ignore[attr-defined]
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
            if not target:
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
    from recall.truth_extraction.extract import extract_corpus_claims

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
        extractions = extract_corpus_claims(
            documents, engine=engine, corpus_names=corpus_names, cache=cache
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

        if cache is not None and getattr(args, "recheck", False):
            from recall.truth_extraction._cache import recheck_cached_extractions

            # The SAME `corpus_names` the extraction ran with. `extraction_cache_key` hashes them,
            # so passing the document keys instead produced a different key for every file, every
            # lookup missed, and the report read "0 checked, rate not measured" without saying why.
            report = recheck_cached_extractions(
                documents, engine=engine, corpus_names=corpus_names, cache=cache
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


def main(argv: list[str] | None = None) -> None:
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
    parser = argparse.ArgumentParser(prog="recall")
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
    # No `choices=`: the accepted set is whatever `resolve_embedder` supports
    # (hashing, fastembed[:model], st:<model>, voyage[:model], openai[:model]), and
    # duplicating it here is how it drifted out of step with the setup wizard.
    parser.add_argument(
        "--embedder",
        default=os.environ.get("RECALL_EMBEDDER", "fastembed"),
        help="hashing, fastembed[:model], st:<model>, voyage[:model], openai[:model]",
    )
    parser.add_argument(
        "--table",
        default="chunks",
        help="table to read/write (default: chunks). Use a throwaway name to keep an "
        "experiment out of your real memory index.",
    )
    parser.add_argument(
        "--tenant",
        default=DEFAULT_TENANT,
        help=f"tenant namespace to operate on (default: {DEFAULT_TENANT}). Every command is "
        f"scoped to one tenant; `forget` in particular deletes nothing outside it, so an "
        f"erasure request against another tenant needs this flag.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser(
        "setup", help="run the first install wizard and write a local .env file"
    ).set_defaults(
        _opens_db=True  # the wizard connects when the operator accepts the calibrate prompt
    )

    p_schema = sub.add_parser("schema", help="inspect or apply versioned database migrations")

    p_schema.set_defaults(_opens_db=True)
    p_schema.add_argument(
        "--dim",
        type=int,
        default=None,
        help="embedding dimension (default: infer from --embedder)",
    )
    schema_sub = p_schema.add_subparsers(dest="schema_cmd", required=True)
    schema_sub.add_parser("status", help="show installed and required schema versions")
    schema_sub.add_parser("plan", help="show pending migrations without changing the database")
    schema_sub.add_parser("apply", help="apply pending migrations with the migration role")
    p_schema_grants = schema_sub.add_parser(
        "grants",
        help="print the GRANT statements a serving role needs (prints SQL, runs none)",
    )
    p_schema_grants.add_argument("--role", required=True, help="the serving role name")
    p_schema_grants.add_argument(
        "--enterprise",
        action="store_true",
        help="also grant the enterprise control-plane tables and their sequence",
    )

    p_manifest = sub.add_parser("manifest", help="create or verify immutable corpus manifests")
    manifest_sub = p_manifest.add_subparsers(dest="manifest_cmd", required=True)
    p_manifest_create = manifest_sub.add_parser(
        "create", help="canonicalise an S3 object inventory"
    )
    p_manifest_create.add_argument("--corpus-version", required=True)
    p_manifest_create.add_argument("--objects", required=True, help="JSON array of object entries")
    p_manifest_create.add_argument("--output", required=True)
    p_manifest_verify = manifest_sub.add_parser("verify", help="verify every immutable S3 object")
    p_manifest_verify.add_argument("manifest")
    p_manifest_verify.add_argument("--version-id")
    p_manifest_verify.add_argument("--sha256")
    p_manifest_verify.add_argument("--size", type=int)

    p_generation = sub.add_parser("generation", help="manage immutable blue green generations")

    p_generation.set_defaults(_opens_db=True)
    generation_sub = p_generation.add_subparsers(dest="generation_cmd", required=True)
    p_build = generation_sub.add_parser("build", help="create and build a generation")
    p_build.add_argument("manifest")
    p_build.add_argument("--manifest-version-id")
    p_build.add_argument("--manifest-sha256")
    p_build.add_argument("--manifest-size", type=int)
    p_build.add_argument("--embedder-provider", default=None)
    p_build.add_argument("--embedder-revision", default=None)
    p_build.add_argument("--embedder-artifact-digest", default=None)
    p_build.add_argument("--unverified-development", action="store_true")
    p_build.add_argument("--chunker", choices=["text", "code"], default="text")
    p_build.add_argument("--max-chars", type=int, default=800)
    p_build.add_argument("--overlap", type=int, default=80)
    p_validate = generation_sub.add_parser("validate", help="validate a built generation")
    p_validate.add_argument("generation_id")
    p_promote = generation_sub.add_parser("promote", help="promote a ready generation")
    p_promote.add_argument("generation_id")
    p_promote.add_argument("--unsafe-development-promotion", action="store_true")
    generation_sub.add_parser("rollback", help="atomically restore the previous generation")
    generation_sub.add_parser("list", help="list immutable generation history")
    p_gc = generation_sub.add_parser("gc", help="collect expired retired generations")
    p_gc.add_argument("--retention-days", type=int, default=7)
    p_gc.add_argument("--retain-previous", type=int, default=2)

    p_index = sub.add_parser("index", help="index a folder of markdown or code")

    p_index.set_defaults(_opens_db=True)
    p_index.add_argument("path")
    p_index.add_argument(
        "--glob",
        default=DEFAULT_GLOB,
        help="file glob to index — e.g. '**/*.py' for code (auto-uses code chunking). Default: markdown.",
    )
    p_index.add_argument(
        "--allow-prune",
        action="store_true",
        help="permit this run to drop most of the indexed corpus. Re-indexing removes files that "
        "are gone from disk; when most of them vanish at once that is refused, because it "
        "usually means the corpus is missing rather than deleted. Pass this once you have "
        "confirmed the files really are gone.",
    )

    p_forget = sub.add_parser(
        "forget",
        help="permanently delete indexed memory for the given source(s) — irreversible",
    )

    p_forget.set_defaults(_opens_db=True)
    p_forget.add_argument(
        "sources",
        nargs="+",
        help="source value(s) to forget, exactly as stored (see the `source` field in "
        "`recall search` output)",
    )
    p_forget.add_argument(
        "--yes",
        action="store_true",
        help="actually delete. Without this flag, forget only PREVIEWS what would be removed "
        "and changes nothing — this command is the right-to-erasure path, it is "
        "irreversible, and it is also invoked from scripts, so a typo or an unattended "
        "run must not silently wipe a corpus. Re-run with --yes once the preview looks right.",
    )

    p_search = sub.add_parser("search", help="search the index")

    p_search.set_defaults(_opens_db=True)
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)
    p_search.add_argument(
        "--entail",
        action="store_true",
        # No install command in this string, deliberately. argparse wraps help through
        # `textwrap.wrap`, which defaults to `break_on_hyphens=True`, so `recall-rag[entail]`
        # renders as `recall-` / `rag[entail]` at COLUMNS 63-69 and 111-123 — including 120,
        # which is a very common terminal width. A command that arrives pre-broken is worse
        # than no command. The exact line lives in the ImportError that fires when the extra is
        # actually missing (`recall/entailment.py`), which argparse never touches.
        help="opt-in entailment stage: demote hits that don't answer the query "
        "(requires the entail extra; downloads the QNLI judge on first use)",
    )
    p_search.add_argument(
        "--evidence",
        action="store_true",
        help="also print the generator-neutral evidence bundle and the exact prompt it renders "
        "to, as JSON. Only verdict-ok hits enter the bundle, in retrieval order; an abstention "
        "produces an empty bundle. Additive: the normal listing is printed either way.",
    )

    p_reasoning = sub.add_parser(
        "reasoning",
        help="explicit opt-in reasoning projection, proposals, query, trace and audit tools",
    )
    p_reasoning.set_defaults(_opens_db=True)
    reasoning_sub = p_reasoning.add_subparsers(dest="reasoning_cmd", required=True)
    p_reasoning_projection = reasoning_sub.add_parser(
        "projection", help="build and inspect the derived reasoning projection"
    )
    p_reasoning_projection.add_argument(
        "--include-text",
        action="store_true",
        help="include evidence text in the projection summary input. Defaults off for privacy.",
    )
    p_reasoning_proposals = reasoning_sub.add_parser(
        "proposals", help="inspect deterministic inference proposals"
    )
    p_reasoning_proposals.add_argument(
        "--include-extracted",
        action="store_true",
        help="also list proposals replayed from prose extraction recorded at ingest. Refuses "
        "if nothing was recorded: extraction never runs on the query path.",
    )
    p_reasoning_query = reasoning_sub.add_parser("query", help="run a bounded reasoning query")
    p_reasoning_query.add_argument("query")
    p_reasoning_query.add_argument("-k", type=int, default=5)
    p_reasoning_query.add_argument("--source")
    p_reasoning_query.add_argument(
        "--mode",
        choices=["evidence_assembly", "proposal_assisted", "review_required", "retrieval_only"],
        default="proposal_assisted",
    )
    p_reasoning_query.add_argument("--max-steps", type=int, default=12)
    p_reasoning_query.add_argument("--max-graph-nodes", type=int, default=32)
    p_reasoning_query.add_argument("--max-evidence-tokens", type=int, default=2048)
    p_reasoning_trace = reasoning_sub.add_parser(
        "trace", help="run a bounded query and export only the reasoning trace"
    )
    p_reasoning_trace.add_argument("query")
    p_reasoning_trace.add_argument("--output", required=True)
    p_reasoning_trace.add_argument("-k", type=int, default=5)
    p_reasoning_trace.add_argument("--source")
    p_reasoning_trace.add_argument("--max-steps", type=int, default=12)
    p_reasoning_trace.add_argument("--max-graph-nodes", type=int, default=32)
    p_reasoning_trace.add_argument("--max-evidence-tokens", type=int, default=2048)
    p_reasoning_audit = reasoning_sub.add_parser(
        "audit", help="run the reasoning integration audit"
    )
    p_reasoning_audit.add_argument("--query", default="reasoning audit sentinel")

    # No `_opens_db`: extraction is an ingest-side filesystem concern and never connects. The
    # set of DB-opening commands is derived from these declarations, so leaving it off IS the
    # answer to the question that guard asks, not an omission.
    p_extract = sub.add_parser(
        "extract",
        help="extract structured truth claims from memo prose (no DB needed; writes nothing)",
    )
    extract_sub = p_extract.add_subparsers(dest="extract_cmd", required=True)
    # `description` as well as `help`: `help` shows in the PARENT's listing, `description` in
    # this subparser's own `--help`, which is where someone checks what it will do to their
    # files. Stating "writes nothing" only in the parent listing leaves that question
    # unanswered exactly where it gets asked.
    _extract_run_blurb = (
        "Extract claims from a corpus. This writes nothing: it has no --apply, because "
        "declaring a claim needs a named human at `recall rewrite apply`. Review with "
        "`recall rewrite plan`."
    )
    p_extract_run = extract_sub.add_parser(
        "run", help=_extract_run_blurb, description=_extract_run_blurb
    )
    p_extract_run.add_argument("path")
    p_extract_run.add_argument("--glob", default=DEFAULT_GLOB)
    p_extract_run.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="read at most this many files. Targets still resolve against the WHOLE corpus, "
        "so sampling does not change the answers.",
    )
    p_extract_run.add_argument(
        "--recheck",
        action="store_true",
        help="re-call the engine on cached keys and report the mismatch rate, to MEASURE "
        "whether determinism holds rather than assume it (needs --cache)",
    )
    # A PATH again, now that there is a store behind it. It was briefly a boolean, because an
    # earlier version accepted a path, ignored it entirely and built a process-local cache, so
    # nothing was written to PATH and a second run hit nothing. Advertising a persistence that
    # does not exist is worse than not offering it; the flag came back when the persistence did.
    p_extract_run.add_argument(
        "--cache",
        default=None,
        metavar="PATH",
        help="persist extraction results at PATH, so re-ingesting an unchanged memo does not "
        "re-pay for it. Also what makes --recheck possible.",
    )
    _extract_show_blurb = (
        "Show the claims and refusals for a single file. Targets are resolved against the "
        "file's own directory, not against the file alone. This writes nothing."
    )
    p_extract_show = extract_sub.add_parser(
        "show", help=_extract_show_blurb, description=_extract_show_blurb
    )
    p_extract_show.add_argument("file")
    p_extract_show.add_argument("--glob", default=DEFAULT_GLOB)

    # No `_opens_db` here either: review and declaration are filesystem work.
    p_rewrite = sub.add_parser(
        "rewrite",
        help="review extracted claims and declare accepted ones in corpus frontmatter",
    )
    rewrite_sub = p_rewrite.add_subparsers(dest="rewrite_cmd", required=True)

    _plan_blurb = "Show every proposal the corpus states, and change nothing."
    p_rw_plan = rewrite_sub.add_parser("plan", help=_plan_blurb, description=_plan_blurb)
    p_rw_plan.add_argument("path")
    p_rw_plan.add_argument("--glob", default=DEFAULT_GLOB)

    _apply_blurb = (
        "Declare ONE reviewed proposal in its memo. DRY RUN by default: it prints the plan and "
        "changes nothing unless --apply is given. --reviewer and --note are required because "
        "nothing reaches corpus metadata without a named human."
    )
    p_rw_apply = rewrite_sub.add_parser("apply", help=_apply_blurb, description=_apply_blurb)
    p_rw_apply.add_argument("path")
    p_rw_apply.add_argument("--glob", default=DEFAULT_GLOB)
    # Either identity, but one of them. `--claim` exists because `recall_rewrite_plan` over MCP
    # derives proposals from the STORE graph while this command derives them from the
    # filesystem extractor, and an id hashes in provider, tenant, generation and pipeline: the
    # two id spaces are disjoint, so every id that surface emits is one `--proposal` refuses.
    # Claim keys are generation independent, which is also why the rejection ledger uses them.
    _apply_id = p_rw_apply.add_mutually_exclusive_group(required=True)
    _apply_id.add_argument("--proposal", help="the proposal id to declare")
    _apply_id.add_argument(
        "--claim", help="the claim key to declare, as reported by recall_rewrite_plan over MCP"
    )
    p_rw_apply.add_argument(
        "--reviewer", required=True, help="identity of the human accepting this proposal"
    )
    p_rw_apply.add_argument(
        "--note", required=True, help="why it was accepted; kept in the audit record"
    )
    p_rw_apply.add_argument(
        "--apply", action="store_true", help="actually write the edit to the memo file"
    )

    # `reject` takes a PATH, which the original design sketch did not. The ledger is keyed by
    # CLAIM (relation plus the two normalised document names), deliberately not by proposal id,
    # because ids hash in the generation and would forget every rejection at the next re-index.
    # Resolving an id to a claim key therefore needs the corpus.
    _reject_blurb = (
        "Record a human's refusal so the proposal does not resurface. Needs the corpus, "
        "because the ledger is keyed by claim rather than by proposal id."
    )
    p_rw_reject = rewrite_sub.add_parser("reject", help=_reject_blurb, description=_reject_blurb)
    p_rw_reject.add_argument("path")
    p_rw_reject.add_argument("--glob", default=DEFAULT_GLOB)
    _reject_id = p_rw_reject.add_mutually_exclusive_group(required=True)
    _reject_id.add_argument("--proposal")
    _reject_id.add_argument("--claim", help="the claim key, as reported by recall_rewrite_plan")
    p_rw_reject.add_argument("--reviewer", required=True)
    p_rw_reject.add_argument("--note", required=True)

    _verify_blurb = "Check that every declared supersedes edge still resolves to one file."
    p_rw_verify = rewrite_sub.add_parser(
        "verify", help=_verify_blurb, description=_verify_blurb
    )
    p_rw_verify.add_argument("path")
    p_rw_verify.add_argument("--glob", default=DEFAULT_GLOB)

    sub.add_parser("demo", help="index corpus/ and run sample memory queries").set_defaults(
        _opens_db=True
    )
    sub.add_parser(
        "code", help="index recall's own source and run sample code queries"
    ).set_defaults(_opens_db=True)

    p_lint = sub.add_parser(
        "lint",
        help="check a corpus's supersession graph for broken/missing edges (no DB needed)",
    )

    p_lint.set_defaults(_opens_db=True)
    p_lint.add_argument("path")
    p_lint.add_argument("--glob", default=DEFAULT_GLOB)
    p_lint.add_argument(
        "--semantic",
        action="store_true",
        help="also run the retrieval-based MISSING-edge check: flag memos highly similar to a "
        "prior closed decision they don't reference (needs the DB + embedder; opt-in)",
    )
    p_lint.add_argument(
        "--fix",
        action="store_true",
        help="propose the frontmatter `supersedes:` edge for each closure marker whose target "
        "is provable. DRY RUN by default — prints the plan and changes nothing.",
    )
    p_lint.add_argument(
        "--apply",
        action="store_true",
        help="with --fix, actually write the proposed edges to the memo files",
    )
    p_lint.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="cosine threshold for --semantic (default: the calibrated abstention threshold "
        "for this embedder; must be calibrated per embedder — see FINDINGS section 2)",
    )

    p_check = sub.add_parser(
        "check",
        help="write-time gate: for the memo(s) you are committing, ask for the supersession "
        "edge while you still know the answer (no DB needed)",
    )
    p_check.add_argument("paths", nargs="+", help="the memo file(s) being written")
    p_check.add_argument(
        "--corpus",
        default=None,
        help="corpus dir used to filter candidates to real documents (default: each file's own "
        "directory)",
    )
    p_check.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when a memo needs an edge — use this in a pre-commit hook",
    )

    # Top-level `calibrate` is the INSTALL-TIME step: it fits the abstention threshold to the
    # operator's own corpus, which is the only thing that makes the shipped default meaningful.
    # The generation-bound measurement is an enterprise operation and lives under `calibration`,
    # beside the artifacts it produces.
    p_cal = sub.add_parser(
        "calibrate",
        help="calibrate the abstention threshold for this embedder against labeled queries",
    )
    p_cal.set_defaults(_opens_db=True)
    p_cal.add_argument("queries", help="JSON list of {query, answerable, relevant_ids} entries")
    p_cal.add_argument(
        "--corpus", default=None, help="corpus dir (default: the built-in eval corpus)"
    )
    p_cal.add_argument("--out", default=None, help="output path (default: calibration.json)")

    p_calibration = sub.add_parser("calibration", help="inspect or transfer calibration artifacts")

    p_calibration.set_defaults(_opens_db=True)
    calibration_sub = p_calibration.add_subparsers(dest="calibration_cmd", required=True)
    p_cal_measure = calibration_sub.add_parser(
        "calibrate", help="measure a calibration bound to one immutable generation"
    )
    p_cal_measure.add_argument("--generation", required=True)
    p_cal_measure.add_argument("--queries", dest="query_file", required=True)
    p_cal_measure.add_argument("--publish", action="store_true")
    calibration_sub.add_parser("list")
    p_cal_show = calibration_sub.add_parser("show")
    p_cal_show.add_argument("calibration_id")
    p_cal_export = calibration_sub.add_parser("export")
    p_cal_export.add_argument("calibration_id")
    p_cal_export.add_argument("--output", required=True)
    p_cal_import = calibration_sub.add_parser("import")
    p_cal_import.add_argument("path")

    args = parser.parse_args(argv)
    # Commands that will actually open a connection FAIL CLOSED on the insecure default DSN;
    # everything else only warns.
    #
    # This is grafted FORWARD from the merge, not restored from before it: the pre-merge CLI had
    # no such check at all, only the warning. An earlier version of this comment claimed the
    # merge had dropped the wiring, which was simply false, and a bug audit caught it.
    #
    # ⚠️ This set is INCOMPLETE and known to be: `generation`, `calibration`, `schema` and
    # `lint --semantic` all open connections and are not in it, and `--migration-dsn` (the
    # DDL-owner credential) is never checked on any path. Tracked as follow-up; do not read the
    # presence of this guard as coverage.
    # Every command that will open a connection FAILS CLOSED on the insecure default DSN;
    # the rest only warn.
    #
    # An earlier version of this set listed six commands by hand and missed four that connect
    # (generation, calibration, schema, and lint --semantic), so the guard read as coverage and
    # was not. The set is derived from the parsers now: a subcommand declares `_opens_db=True`
    # beside its own definition, so a new one cannot be added without answering the question.
    opens_db = bool(getattr(args, "_opens_db", False))
    if args.cmd == "lint":  # only the --semantic path reaches a database
        opens_db = bool(getattr(args, "semantic", False))
    if args.cmd == "schema" and getattr(args, "schema_cmd", None) == "grants":
        opens_db = False  # prints SQL for an operator to run; opens nothing

    if (
        opens_db
        and args.cmd != "setup"  # see the setup-specific carve-out for _require_secure below —
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
                    "This is `recall setup`, which cannot prompt its way out of this: pass a "
                    "DSN explicitly with `recall --dsn <dsn> setup`, or set "
                    "RECALL_ALLOW_INSECURE_DSN=1 to accept the risk deliberately."
                ) from exc
        else:
            _require_secure(args.dsn)
    else:
        warn_if_insecure_dsn(args.dsn)  # loud stderr note if default creds target a remote host

    # The DDL-owner credential was never checked or even warned about on any path, which is the
    # wrong way round: it is the most privileged DSN this CLI accepts.
    migration_dsn = getattr(args, "migration_dsn", None)
    if migration_dsn and opens_db and args.cmd == "schema":  # `opens_db` so grants stays exempt
        _require_secure(migration_dsn)

    if args.cmd == "setup":
        from recall.setup import run_setup_wizard

        # Pass the caller's table through: the wizard checks the chosen embedder's width against
        # it, and checking a different table than the one in use is worse than not checking.
        run_setup_wizard(dsn=args.dsn, table=args.table)
        return

    if args.cmd == "schema":
        from recall.schema import apply_migrations, schema_plan, schema_status

        if args.schema_cmd == "grants":
            # Prints SQL for an operator to run as the object owner; touches no database, so
            # it needs neither a DSN nor an embedder.
            from recall.schema import serving_grants

            for statement in serving_grants(
                args.role, table=args.table, enterprise=args.enterprise
            ):
                print(statement)
            return
        dim = args.dim if args.dim is not None else _make_embedder(args.embedder).dim
        inspect_dsn = args.migration_dsn or args.dsn
        if args.schema_cmd == "status":
            status = schema_status(inspect_dsn, table=args.table, dim=dim)
            print(f"table: {status.table}")
            print(f"current: {status.current_version or 'none'}")
            print(f"required: {status.required_version}")
            print(f"compatible: {'yes' if status.compatible else 'no'}")
            for migration in status.migrations:
                print(f"{migration.version} {migration.state:<7} {migration.filename}")
            if not status.compatible:
                raise SystemExit(1)
            return
        if args.schema_cmd == "plan":
            pending = schema_plan(inspect_dsn, table=args.table, dim=dim)
            if not pending:
                print("schema is current; no changes planned")
            else:
                for migration in pending:
                    print(f"would apply {migration.version} {migration.filename}")
            return
        if not args.migration_dsn:
            raise SystemExit(
                "schema apply requires --migration-dsn or RECALL_MIGRATION_DSN; "
                "the serving DSN is never used for DDL"
            )
        applied = apply_migrations(args.migration_dsn, table=args.table, dim=dim)
        if not applied:
            print("schema is current; nothing applied")
        else:
            for migration in applied:
                print(f"applied {migration.version} {migration.filename}")
        return

    if args.cmd == "manifest":
        from recall.lineage import IndexManifestV1, ManifestObjectV1
        from recall.manifest import S3ObjectReader, load_inventory, load_manifest

        if args.manifest_cmd == "create":
            manifest = IndexManifestV1(
                args.tenant,
                args.corpus_version,
                load_inventory(args.objects),
            )
            Path(args.output).write_text(manifest.to_json(), encoding="utf-8")
            print(f"wrote {args.output} sha256={manifest.digest} objects={len(manifest.objects)}")
            return
        reader = S3ObjectReader.from_environment()
        if args.manifest.startswith("s3://"):
            if args.version_id is None or args.sha256 is None or args.size is None:
                raise SystemExit("an S3 manifest requires --version-id, --sha256 and --size")
            reference = ManifestObjectV1(
                args.manifest,
                args.version_id,
                "application/json",
                args.size,
                args.sha256,
            )
            manifest = IndexManifestV1.from_json(reader.fetch(reference).data)
        else:
            manifest = load_manifest(args.manifest)
        if manifest.tenant_id != args.tenant:
            raise SystemExit(
                f"manifest tenant {manifest.tenant_id!r} does not match --tenant {args.tenant!r}"
            )
        reader.verify(manifest)
        print(f"verified sha256={manifest.digest} objects={len(manifest.objects)}")
        return

    if args.cmd == "generation":
        from recall.generations import GenerationManager

        manager = GenerationManager(args.dsn, args.tenant)
        if args.generation_cmd == "list":
            for generation in manager.list_generations():
                print(
                    f"{generation.generation_id} {generation.state.value:<18} "
                    f"pipeline={generation.pipeline_fingerprint} "
                    f"corpus={generation.corpus_fingerprint}"
                )
            return
        if args.generation_cmd == "rollback":
            print(f"active generation: {manager.rollback()}")
            return
        if args.generation_cmd == "gc":
            collected = manager.gc(
                retention_days=args.retention_days,
                retain_previous=args.retain_previous,
            )
            print(f"collected {len(collected)} generation(s): {', '.join(collected) or '(none)'}")
            return
        if args.generation_cmd == "validate":
            generation_validation = manager.validate(args.generation_id)
            print(
                f"ready {generation_validation.generation_id}: "
                f"{generation_validation.sources} sources, "
                f"{generation_validation.chunks} chunks"
            )
            return
        if args.generation_cmd == "promote":
            manager.promote(
                args.generation_id,
                unsafe_development=args.unsafe_development_promotion,
            )
            print(f"active generation: {args.generation_id}")
            return

        from recall.lineage import (
            ChunkerIdentity,
            EmbedderIdentity,
            IndexManifestV1,
            ManifestObjectV1,
            PipelineIdentity,
        )
        from recall.manifest import S3ObjectReader, load_manifest

        environment = manager.environment
        reader = S3ObjectReader.from_environment()
        if args.manifest.startswith("s3://"):
            if (
                args.manifest_version_id is None
                or args.manifest_sha256 is None
                or args.manifest_size is None
            ):
                raise SystemExit(
                    "an S3 manifest requires --manifest-version-id, --manifest-sha256 and "
                    "--manifest-size"
                )
            reference = ManifestObjectV1(
                args.manifest,
                args.manifest_version_id,
                "application/json",
                args.manifest_size,
                args.manifest_sha256,
            )
            manifest = IndexManifestV1.from_json(reader.fetch(reference).data)
        else:
            if environment == "production":
                raise SystemExit("production generation builds require a versioned S3 manifest")
            manifest = load_manifest(args.manifest)
        embedder = _make_embedder(args.embedder)
        revision = args.embedder_revision
        provider = args.embedder_provider
        if isinstance(embedder, HashingEmbedder):
            provider = provider or "recall"
            revision = revision or "hashing-md5-bow-v1"
        else:
            provider = provider or "fastembed"
        identity = EmbedderIdentity(
            provider=provider,
            model=embedder.name,
            dimension=embedder.dim,
            revision=revision,
            artifact_digest=args.embedder_artifact_digest,
            unverified_reason=(
                "explicit development build"
                if args.unverified_development
                and not revision
                and not args.embedder_artifact_digest
                else None
            ),
        )
        if args.chunker == "code":
            generation_chunker = functools.partial(chunk_code, max_chars=args.max_chars)
            chunker_identity = ChunkerIdentity(
                "recall.chunk_code", 1, {"max_chars": args.max_chars}
            )
        else:
            generation_chunker = functools.partial(
                chunk_text, max_chars=args.max_chars, overlap=args.overlap
            )
            chunker_identity = ChunkerIdentity(
                "recall.chunk_text",
                1,
                {"max_chars": args.max_chars, "overlap": args.overlap},
            )
        pipeline = PipelineIdentity(identity, chunker_identity)
        generation = manager.create(
            manifest,
            pipeline,
            allow_unverified=args.unverified_development,
        )
        generation_stats = manager.build(
            generation.generation_id,
            reader,
            embedder,
            generation_chunker,
        )
        print(
            f"built {generation_stats.generation_id}: {generation_stats.objects} objects, "
            f"{generation_stats.chunks} chunks, {generation_stats.reused_objects} objects "
            f"reused; run `recall generation validate {generation_stats.generation_id}`"
        )
        return

    if args.cmd == "lint":  # pure filesystem check — no embedder, no DB
        from recall.lint import lint_corpus

        try:
            issues = lint_corpus(args.path, glob=args.glob)
        except FileNotFoundError as exc:
            print(f"recall lint: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        for i in issues:
            print(f"{i.level:<8} {i.code:<26} {i.file}: {i.message}")
        errors = sum(1 for i in issues if i.level == "error")
        warnings = len(issues) - errors

        # Bound HERE, not inside `if args.fix:`, because `--semantic` is reachable without
        # `--fix` and consumes it below. Initialising it in the fix block made plain
        # `recall lint <path> --semantic` die with UnboundLocalError before doing any work.
        _validated_emb: Embedder | None = None

        if args.fix:
            from recall.fix import apply_proposal, propose_fixes

            proposals, unfixable = propose_fixes(args.path, glob=args.glob)
            print()
            for p in proposals:
                print(f"  {p.edit_file}: + supersedes: {p.target}")
                print(f"      because {p.evidence_file} says {p.evidence!r}")
            for u in unfixable:
                print(f"  SKIP {u.file}: {u.reason}")
            print(f"\n{len(proposals)} edge(s) proposable, {len(unfixable)} need a human")
            if args.apply and args.semantic:
                # Resolve the embedder BEFORE writing. `--semantic` needs one, and dropping
                # argparse's `choices=` moved an unknown spelling's failure from "exit 2 before
                # anything happened" to "after apply_proposal has already rewritten the memos".
                # This is the only destructive path that resolved it late.
                _validated_emb = _make_embedder(args.embedder)
            if not args.apply:
                # Dry run by DEFAULT: this edits the user's own documents, and a tool that
                # rewrites your memory the first time you try it has earned distrust.
                print("dry run — nothing written. Re-run with --apply to write these edges.")
            else:
                root = Path(args.path)
                written = 0
                for p in proposals:
                    try:
                        apply_proposal(root, p)
                    except ValueError as exc:
                        # One memo the writer refuses must not discard the rest of the run. The
                        # loop previously had no guard, so a single undecodable file aborted with
                        # a traceback AFTER the earlier proposals had already been written — the
                        # worst of both, a partial apply the user has to reconstruct by hand.
                        #
                        # `ValueError`, not `UnreadableMemo`, because the writer has a second
                        # refusal now: `insert_frontmatter_line` rejects a value carrying a line
                        # break. `UnreadableMemo` subclasses `ValueError`, so this still catches
                        # everything it did, and the widening is what keeps that second refusal
                        # from reintroducing the exact partial apply this handler exists to stop.
                        # `propose_fixes` filters those values first, so reaching here means a
                        # caller built the `Proposal` itself.
                        #
                        # The memo is named explicitly rather than left to the exception's own
                        # text. `UnreadableMemo` opens with the path, but the widening admits
                        # exceptions raised further down that do not: a filename that is not
                        # valid UTF-8 arrives surrogate-escaped from `Path.glob`, passes the
                        # line-break check, and makes the writer's `.encode("utf-8")` raise
                        # `UnicodeEncodeError` — also a `ValueError`, and its message names a
                        # code point, not a file. A nameless SKIP in a list of many, followed by
                        # "skipped 1", leaves the operator of a destructive command unable to
                        # tell WHICH memo was passed over.
                        print(f"  SKIP {p.edit_file}: {exc}")
                        continue
                    written += 1
                print(f"wrote {written} edge(s), skipped {len(proposals) - written}.")

        chains = []
        if args.semantic:  # opt-in retrieval-based missing-edge check (needs DB + embedder)
            from recall.semantic_lint import semantic_lint

            # Reuse the instance built for pre-write validation. Constructing twice is not
            # free: the cloud embedders probe the API inside __init__, so a second build is a
            # second billable request, and the local ones reload the model.
            emb = _validated_emb or _make_embedder(args.embedder)
            # --threshold's help promises "the calibrated abstention threshold for this
            # embedder" as the default; hardcoding 0.70 made that untrue on every corpus
            # whose calibration says otherwise.
            _cal = load_for(emb.name)
            if args.threshold is not None:
                thr, _src = args.threshold, "--threshold"
            elif _cal is not None:
                thr, _src = _cal.threshold, f"calibrated for {emb.name}"
            else:
                # `load_for` returns None WITHOUT raising when the artifact is keyed to a
                # different embedder, so this fallback is reachable even when a calibration
                # file exists — notably because the setup wizard keys it by the embedder
                # SPELLING while `recall calibrate` keys it by `embedder.name`. Saying which
                # threshold was used is the difference between a silent wrong answer and a
                # visible one; the help text promises the calibrated value.
                thr, _src = 0.70, f"UNCALIBRATED default, no calibration matched {emb.name}"
            print(f"semantic threshold: {thr:.2f} ({_src})")
            chains = semantic_lint(args.dsn, emb, args.path, threshold=thr, glob=args.glob)
            for c in chains:
                print(
                    f"warning  unlinked-chain             {c.new_memo}: highly similar "
                    f"(cos={c.cosine:.2f}) to closed decision {c.prior!r} it does not "
                    f"reference — add `supersedes: {c.prior}`?"
                )
            warnings += len(chains)

        print(f"{errors} errors, {warnings} warnings")
        if errors:
            raise SystemExit(1)
        return

    if args.cmd == "extract":  # pure filesystem path — no embedder, no DB
        _run_extract(args)
        return

    if args.cmd == "rewrite":  # pure filesystem path — no embedder, no DB
        from recall.rewrite import RewriteRefused

        # `required=True` is satisfied by `--reviewer ""`. The gate at the parser is the first
        # half; this is the second. A gate a caller passes by typing nothing is a field, not a
        # person, and this is the one command in the library that edits a user's own memos.
        for field in ("reviewer", "note"):
            value = getattr(args, field, None)
            if value is not None and not value.strip():
                print(f"recall rewrite: --{field} must not be empty", file=sys.stderr)
                raise SystemExit(2)
        try:
            _run_rewrite(args)
        except RewriteRefused as exc:
            print(f"recall rewrite: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        return

    if args.cmd == "check":  # pure filesystem check — no embedder, no DB
        from recall.check import check_file, corpus_names, format_prompt

        needs = 0
        for raw in args.paths:
            f = Path(raw)
            if not f.exists():
                print(f"recall check: no such file: {raw}", file=sys.stderr)
                raise SystemExit(2)
            names = corpus_names(args.corpus or f.parent)
            result = check_file(f, names)
            if result.needs_attention:
                needs += 1
                print(format_prompt(result))
        if needs:
            print(f"\n{needs} memo(s) state a closure in prose only.")
            if args.strict:
                raise SystemExit(1)
        return

    if args.cmd == "calibration":
        from recall.calibration_v2 import (
            CalibrationError,
            CalibrationRepository,
            load_query_set,
        )

        repository = CalibrationRepository(args.dsn, args.tenant)
        try:
            if args.calibration_cmd == "list":
                print(json.dumps(repository.list_records(), indent=2, default=str))
            elif args.calibration_cmd == "show":
                print(
                    json.dumps(repository.show_record(args.calibration_id), indent=2, default=str)
                )
            elif args.calibration_cmd == "export":
                print(repository.export_bundle(args.calibration_id, args.output))
            elif args.calibration_cmd == "import":
                print(repository.import_bundle(args.path))
            elif args.calibration_cmd == "calibrate":
                labels, _digest = load_query_set(args.query_file)
                artifact = repository.calibrate(
                    args.generation, labels, _make_embedder(args.embedder)
                )
                if args.publish:
                    artifact = repository.publish(artifact.calibration_id)
                print(f"calibration: {artifact.calibration_id}")
                print(f"status: {artifact.status.value}")
                print(f"generation: {artifact.generation_id}")
                print(f"pipeline: {artifact.pipeline_fingerprint}")
                print(f"corpus: {artifact.corpus_fingerprint}")
                print(f"queries: {artifact.query_set_digest}")
            else:  # unreachable while argparse constrains the choices, but never guess a default
                raise SystemExit(f"unknown calibration subcommand: {args.calibration_cmd}")
        except CalibrationError as exc:
            raise SystemExit(str(exc)) from exc
        return

    # Legacy process-global calibration is deliberately never auto-loaded here. See the longer
    # note below, kept beside the search/calibrate path where the design question originated.
    calibration = None

    embedder = _make_embedder(args.embedder)
    if args.cmd == "reasoning":
        from recall.generation_store import GenerationStore
        from recall_mcp.service import (
            reasoning_audit,
            reasoning_projection,
            reasoning_proposals,
            reasoning_query,
        )

        if os.environ.get("RECALL_ENV", "development").lower() == "production":
            reasoning_store_context: PgVectorStore = GenerationStore(
                args.dsn, embedder.dim, tenant=args.tenant
            )
        else:
            reasoning_store_context = PgVectorStore(
                args.dsn, dim=embedder.dim, table=args.table, tenant=args.tenant
            )
        with reasoning_store_context as store:
            store.check_schema()
            _reasoning_policy, _reasoning_calibration = _cli_trust(embedder, calibration)
            if args.reasoning_cmd == "projection":
                projection = reasoning_projection(store, include_text=args.include_text)
                _refuse_untrusted_reasoning_inspection(projection.trust_state, _reasoning_policy)
                print(projection.model_dump_json(indent=2))
                return
            if args.reasoning_cmd == "proposals":
                try:
                    proposal_result = reasoning_proposals(
                        store, include_extracted=args.include_extracted
                    )
                except ValueError as exc:
                    # `--include-extracted` refuses when nothing was recorded at ingest. Left
                    # raw it was the flag's only reachable outcome AND a traceback, where every
                    # neighbouring refusal in this CLI prints one line and exits 2.
                    print(f"recall reasoning: {exc}", file=sys.stderr)
                    raise SystemExit(2) from exc
                trust_state = (
                    "trusted" if proposal_result.generation_id != "legacy" else "degraded"
                )
                _refuse_untrusted_reasoning_inspection(trust_state, _reasoning_policy)
                print(proposal_result.model_dump_json(indent=2))
                return

            if args.reasoning_cmd in {"query", "trace"}:
                response = reasoning_query(
                    store,
                    embedder,
                    args.query,
                    source=args.source,
                    k=args.k,
                    mode=getattr(args, "mode", "proposal_assisted"),
                    max_steps=args.max_steps,
                    max_graph_nodes=args.max_graph_nodes,
                    max_evidence_tokens=args.max_evidence_tokens,
                    policy=_reasoning_policy,
                    calibration=_reasoning_calibration,
                )
                if args.reasoning_cmd == "trace":
                    payload = _reasoning_trace_export(response)
                    Path(args.output).write_text(
                        json.dumps(payload, indent=2, default=str),
                        encoding="utf-8",
                    )
                    print(f"trace: {args.output}")
                    return
                print(json.dumps(response.to_dict(), indent=2, default=str))
                return
            if args.reasoning_cmd == "audit":
                print(
                    reasoning_audit(
                        store,
                        embedder,
                        query=args.query,
                        policy=_reasoning_policy,
                        calibration=_reasoning_calibration,
                    ).model_dump_json(indent=2)
                )
                return
            raise SystemExit(f"unknown reasoning subcommand: {args.reasoning_cmd}")

    if args.cmd == "calibrate":
        from recall.calibration import ENV_VAR, _resolve_path
        from recall.setup import calibrate_from_files

        try:
            calibration_result: CalibrationResult = calibrate_from_files(
                dsn=args.dsn,
                embedder_name=embedder.name,
                queries_path=Path(args.queries),
                corpus_dir=Path(args.corpus) if args.corpus else None,
                out=Path(args.out) if args.out else None,
            )
        except ValueError as exc:
            raise SystemExit(2) from exc
        measured = calibration_result.report
        cal = calibration_result.calibration
        path = calibration_result.path
        print(f"embedder:  {embedder.name}")
        print(f"threshold: {cal.threshold} (scale {cal.scale})")
        sep = "n/a" if cal.separability is None else f"{cal.separability:.3f}"
        ci = cal.separability_ci
        # The interval, not just the point, because the bar is applied to its lower bound — a
        # reader who sees only "0.95" cannot reconstruct why a certification failed.
        sep_ci = "" if ci is None else f" [{ci[0]:.3f}, {ci[1]:.3f}]"
        print(
            f"separability (AUC): {sep}{sep_ci} over {cal.n_answerable} answerable / "
            f"{cal.n_unanswerable} unanswerable"
        )
        print(
            f"FCR at default 0.50: {measured.fcr_at_050:.2f} -> at calibrated: "
            f"{measured.fcr_at_suggested:.2f}"
        )
        print(f"saved: {path}")
        if args.out and Path(args.out).resolve() != _resolve_path(None).resolve():
            print(
                f"note: searches load {_resolve_path(None)} by default — set "
                f"{ENV_VAR}={path} for this file to be used"
            )

        # Exit non-zero on a threshold the data does not support. The file is still written: the
        # artifact records `certified: false` and the reason, and refusing to write would destroy
        # the evidence of WHY. What changes is that a calibration step can now fail — measured on
        # LongMemEval, an uncertified threshold refused 44% of the questions retrieval had just
        # answered correctly, and neither the API nor the file said anything was wrong.
        if cal.certified is False:
            print(f"\nNOT CERTIFIED: {cal.certification_reason}", file=sys.stderr)
            print(
                "Saved anyway — there is no better threshold for this data — but abstention on "
                "this corpus is not trustworthy. Do NOT read an abstention as evidence that the "
                "answer is absent.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if cal.certified is None:
            print(f"\nnot judged: {cal.certification_reason}", file=sys.stderr)
        return

    # ⚠️ Deliberately NOT `load_for(embedder.name)`, and a bug audit talked me into that once.
    #
    # `trusted_search` only consults the generation-bound resolver when `calibration is None`
    # (recall/trust.py). Passing a legacy artifact sets calibration_status="legacy_unbound",
    # which the strict policy maps to CALIBRATION_UNCERTIFIED, so pre-loading it REFUSES
    # searches on a deployment that has a properly certified, generation-bound calibration.
    # recall/trust.py states the rule directly: legacy JSON "is deliberately never auto-loaded:
    # it has no tenant, generation, pipeline, corpus, or labelled query-set binding".
    #
    # 🔑 The open consequence, which is a DESIGN question and not an oversight: the artifact
    # `recall calibrate` writes is therefore not read back by this path. Resolve that by
    # deciding where install-time calibration binds, not by reinstating the line below.
    calibration = None

    if args.cmd == "index":
        if os.environ.get("RECALL_ENV", "development").lower() == "production":
            raise SystemExit(
                "local filesystem indexing is development-only; build from an immutable S3 "
                "manifest in production"
            )
        chunker = chunk_code if args.glob.endswith(".py") else chunk_text
        with PgVectorStore(
            args.dsn, dim=embedder.dim, table=args.table, tenant=args.tenant
        ) as store:
            store.check_schema()
            indexer = Indexer(store, embedder, chunker=chunker, allow_prune=args.allow_prune)
            try:
                stats = indexer.index_path(args.path, glob=args.glob)
            except PruneGuardTripped as exc:
                # The message carries the recovery instructions; a traceback would bury them.
                raise SystemExit(str(exc)) from exc
            # `files` counts what was RE-indexed, not what is in the index, so an unchanged
            # re-run reports 0/0 — which reads as "the index is empty" unless `skipped` is shown
            # beside it. `deleted` matters more: pruning is the destructive half of `index`, and
            # reporting it only through a log record meant a deletion could happen in silence.
            summary = f"indexed {stats.chunks} chunks from {stats.files} files"
            if stats.skipped:
                summary += f", {stats.skipped} unchanged"
            if stats.deleted:
                summary += f", pruned {stats.deleted} source(s) no longer on disk"
            print(summary)
    elif args.cmd == "forget":
        from recall.generation_store import GenerationStore
        from recall.generations import NoActiveGeneration

        generation_mode = os.environ.get("RECALL_ENV", "development").lower() == "production"
        # Keep a GenerationStore-typed handle alongside the widened one: the corpus probe below
        # exists only on the subclass, and narrowing here is what lets the type checker see it.
        gen_store: GenerationStore | None = (
            GenerationStore(args.dsn, embedder.dim, tenant=args.tenant) if generation_mode else None
        )
        forget_store: PgVectorStore = (
            gen_store
            if gen_store is not None
            else PgVectorStore(args.dsn, dim=embedder.dim, table=args.table, tenant=args.tenant)
        )
        with forget_store as store:
            store.check_schema()
            requested = list(dict.fromkeys(args.sources))
            # Reject a blank argument before anything commits. `recall forget "$A" "$B" --yes`
            # with one variable unset otherwise erased the first source, raised out of the
            # per-source loop, and printed nothing at all.
            if any(not source.strip() for source in requested):
                raise SystemExit(
                    "forget: empty source argument (an unset shell variable?); nothing deleted"
                )
            if gen_store is not None:
                # Widen the existence check, do not drop it, and ask the right question.
                # `source_content_hashes()` is scoped to ONE generation, so FILTERING on it
                # called a source that had left the active generation "not found" and left it
                # with its rows and no tombstone. But no check at all is not the answer either:
                # forgetting a never-indexed URI writes a permanent tombstone (nothing deletes
                # one, and `build()` skips every manifest entry it matches), so a typo would
                # irreversibly bar that URI. The question is "does the corpus contain this",
                # which the MANIFEST answers and chunk rows do not: an object that chunks to
                # nothing is built as `empty_objects` and writes no row, yet is unquestionably
                # part of the corpus and must be erasable.
                # Scoped to `requested`, and the SAME call the MCP surface makes. Asking the
                # wholesale question here instead put the two surfaces on separate copies of
                # the live-state list, and only this one was pinned by a test.
                known = (
                    gen_store.sources_in_any_generation()
                    | gen_store.manifest_uris_matching(list(requested))
                    | gen_store.sources_in_legacy_table()
                )
                targets = [s for s in requested if s in known]
                unseen = [s for s in requested if s not in known]
                unseen_note = (
                    "not present in any generation, manifest, or the adopted v0.8 table, so "
                    f"NOT erased and NOT tombstoned (check for typos): {', '.join(unseen)}"
                )
            else:
                # The v0.8 table has no generations, so the probe covers everything the
                # tenant owns and an absent source really is a typo. Computed here rather than
                # above because the generation branch never reads it, and on a GenerationStore
                # it costs an active-generation lookup plus a DISTINCT scan.
                try:
                    visible_now = set(store.source_content_hashes())
                except NoActiveGeneration:
                    visible_now = set()
                targets = [s for s in requested if s in visible_now]
                unseen = [s for s in requested if s not in visible_now]
                unseen_note = f"not found (check for typos): {', '.join(unseen)}"
            if not args.yes:
                print(
                    f"DRY RUN: would forget {len(targets)} source(s): "
                    f"{', '.join(targets) if targets else '(none)'}"
                )
                if unseen:
                    print(unseen_note)
                print("nothing deleted — re-run with --yes to actually delete.")
            else:
                # One source per call: `delete_sources` commits a separate transaction each,
                # so a failure part way through leaves the earlier ones erased. Reporting from
                # a finally means a partial erasure is never silent.
                removed = 0
                erased: list[str] = []
                try:
                    for source in targets:
                        removed += store.delete_sources([source])
                        erased.append(source)
                finally:
                    if len(erased) == len(targets):
                        print(f"forgot {removed} chunk(s) from {len(erased)} source(s)")
                    else:
                        # Name them. On an irreversible path, "the rest" is not an answer the
                        # operator can act on, and the survivors are otherwise recoverable only
                        # by re-deriving the argument order by hand.
                        missed = [s for s in targets if s not in set(erased)]
                        print(
                            f"forgot {removed} chunk(s) from {len(erased)} of {len(targets)} "
                            f"source(s); NOT reached: {', '.join(missed)}"
                        )
                    if unseen:
                        print(unseen_note)
    elif args.cmd == "search":
        # `resolve_entailment_judge` reads RECALL_ENTAILMENT (the opt-in the setup wizard
        # writes) plus RECALL_ENTAILMENT_MODEL / _REVISION. Constructing QnliEntailmentJudge()
        # directly ignored all three, so a pinned model was silently replaced by the default
        # download. The explicit --entail flag still forces it on when the env says nothing.
        # `--entail` resolves with the opt-in FORCED and never consults the env's own value,
        # so a malformed RECALL_ENTAILMENT cannot defeat an explicit flag. Checking the plain
        # resolver first would refuse before the flag was ever considered. Forcing goes THROUGH
        # the resolver rather than constructing the judge bare, because the bare form ignores
        # RECALL_ENTAILMENT_MODEL/_REVISION — the defect this block exists to fix. `recall
        # setup` writes RECALL_ENTAILMENT="0", so the forcing path is the common one.
        entail_judge = _entailment_judge(force=True) if args.entail else _entailment_judge()
        if os.environ.get("RECALL_ENV", "development").lower() == "production":
            from recall.generation_store import GenerationStore

            store_context: PgVectorStore = GenerationStore(
                args.dsn, embedder.dim, tenant=args.tenant
            )
        else:
            store_context = PgVectorStore(
                args.dsn, dim=embedder.dim, table=args.table, tenant=args.tenant
            )
        with store_context as store:
            store.check_schema()
            _search_policy, _search_calibration = _cli_trust(embedder, calibration)
            _search_result = trusted_search(
                store,
                embedder,
                args.query,
                # `-k` has no lower bound and `trusted_search`
                # refuses k < 1 as its FIRST statement, so `-k 0` tracebacked out of the
                # library before any of this command's own guards were reached. Clamped at
                # the source; the clamp in `_print_evidence` stays as defence in depth.
                k=max(1, args.k),
                calibration=_search_calibration,
                entailment=entail_judge,
                policy=_search_policy,
            )
            _print_result(_search_result)
            if args.evidence:
                _print_evidence(_search_result, max_items=args.k)
    elif args.cmd == "demo":
        if os.environ.get("RECALL_ENV", "development").lower() == "production":
            raise SystemExit("the filesystem demo is unavailable in production")
        # Resolved BEFORE the store opens and the corpus is indexed: a bad
        # RECALL_ENTAILMENT value raises, and failing after the expensive work is the
        # shape `search` already avoids.
        _demo_judge = _entailment_judge()
        with PgVectorStore(
            args.dsn, dim=embedder.dim, table=args.table, tenant=args.tenant
        ) as store:
            store.check_schema()
            stats = Indexer(store, embedder).index_path("corpus")
            print(f"indexed {stats.chunks} chunks from {stats.files} files\n")
            _run_queries(
                store,
                embedder,
                [
                    "what did we decide about caching?",
                    "do we inject retrieved context into the prompt?",
                    "how many requests per second can a client make?",
                    "how do we handle penguins on mars?",
                ],
                calibration,
                _demo_judge,
            )
    elif args.cmd == "code":
        if os.environ.get("RECALL_ENV", "development").lower() == "production":
            raise SystemExit("local source indexing is unavailable in production")
        # index recall's own package source (content-agnostic engine, code-aware chunking)
        src = Path(__file__).resolve().parent
        # Resolved BEFORE the store opens and the corpus is indexed: a bad
        # RECALL_ENTAILMENT value raises, and failing after the expensive work is the
        # shape `search` already avoids.
        _demo_judge = _entailment_judge()
        with PgVectorStore(
            args.dsn, dim=embedder.dim, table="recall_code", tenant=args.tenant
        ) as store:
            store.check_schema()
            stats = Indexer(store, embedder, chunker=chunk_code).index_path(src, glob="**/*.py")
            print(f"indexed {stats.chunks} code chunks from {stats.files} files\n")
            _run_queries(
                store,
                embedder,
                [
                    "where is reciprocal rank fusion implemented?",
                    "how are embeddings stored in postgres?",
                    "how does cross-encoder reranking reorder hits?",
                ],
                calibration,
                _demo_judge,
            )


if __name__ == "__main__":
    main()
