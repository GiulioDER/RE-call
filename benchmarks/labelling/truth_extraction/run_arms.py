"""Run the two extraction arms over the frozen adjudication pack and score precision.

The pre-registration
(`results/truth_extraction/PREREGISTRATION-prose-extraction.md`) measures PRECISION here and
nowhere else. The 47 gold edge questions can only measure recall, and `build_gold.py` freezes only
the superseded PEP's body as their input, so of the 8 edges restated in prose just 3 are restated
by the document the extractor actually reads. A recall denominator of 3 supports no claim.

So the unit of measurement is one adjudicated `(sentence, target)` candidate. Both arms see the
same 38 candidates and the same source bodies, and each either PROPOSES the edge or REFUSES it.
Precision is proposals that a human labelled `Y`, over proposals made. Rows the adjudicator left
blank are excluded from the denominator rather than guessed at.

## Why R1 needs a reference-pattern extension, and why that is not cheating

`recall.fix._REF` matches three forms: `[[wikilink]]`, `name.md`, and a bare stem carrying a 20xx
year. PEPs use none of them; they cite each other as ``:pep:`387` `` or ``PEP 387``. Dropped onto
PEPs unchanged, R1 proposes zero for a reason that has nothing to do with the relation being
absent, which `recall/eval/promotion/aggregate.py` calls a vacuous arm: a model beating it 0.55 to
0.00 would have beaten a tool that was never pointed at this corpus.

Only the reference pattern is extended. Every refusal rule is imported from `recall.fix` and run
unmodified, because those rules are the arm: `_is_index`, `_is_reported_speech`, `_is_hedged`,
`_is_partial_scope`, and the single-file resolution. If this module ever reimplements one of them,
the comparison stops being about the proposer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.labelling.truth_extraction.artifact_contract import (
    validate_arm_result,
    validate_fixtures_result,
)
from benchmarks.labelling.truth_extraction.peps_header import paragraphs
from recall.eval.metrics import wilson_ci
from recall.truth_extraction.types import SupersessionClaim
from recall.fix import (
    _ACTIVE,
    _PASSIVE,
    _is_hedged,
    _is_index,
    _is_partial_scope,
    _is_reported_speech,
)

PACK = Path(__file__).resolve().parent
CSV_PATH = PACK / "adjudication.csv"
KEY_PATH = PACK / "adjudication_key.json"
FIXTURES = PACK / "fixtures"

#: PEP cross-reference forms, the one thing R1 is allowed to gain. Frozen in the
#: pre-registration so the extension cannot be widened after seeing a score.
_PEP_REF = r"(?::pep:`(\d{1,4})`|\bPEP\s+(\d{1,4})\b|\bpep-(\d{4})\b)"

_ACTIVE_RE = re.compile(rf"\b(?P<marker>{_ACTIVE})[^\n.;]{{0,40}}?{_PEP_REF}", re.IGNORECASE)
_PASSIVE_RE = re.compile(rf"(?P<marker>{_PASSIVE})[^\n.;]{{0,40}}?{_PEP_REF}", re.IGNORECASE)


def _pep_stem(match: re.Match[str]) -> str:
    number = next(g for g in match.groups()[1:] if g)
    return f"pep-{int(number):04d}"


@dataclass(frozen=True)
class Candidate:
    """One adjudicated row, joined to the body the arms actually read."""

    item: str
    sentence: str
    target: str
    source_pep: str
    verdict: str  # "Y", "N", or "" for undecidable


@dataclass
class ArmResult:
    arm: str
    proposed: list[str] = field(default_factory=list)
    refused: dict[str, str] = field(default_factory=dict)  # item -> reason
    #: Candidates the arm never got to READ, which is an apparatus failure and not a refusal:
    #: a sentence the rules arm could not locate, or one whose document the model arm lost to a
    #: batch rung. Counting these as refusals flatters precision and hides a broken run.
    not_read: dict[str, str] = field(default_factory=dict)  # item -> why it was never read
    referred: list[str] = field(default_factory=list)  # requires_review, scored separately
    calls: int = 0  # model calls made, 0 for a deterministic arm
    cache_hits: int = 0  # of those calls, how many the cache answered without the model
    #: prompt revision, model, engine, taken from the extractions themselves rather than from the
    #: engine object, so the artifact records what actually answered.
    identity: dict[str, str] = field(default_factory=dict)
    batch_failures: dict[str, str] = field(default_factory=dict)  # file -> reason


def load_pack(csv_path: Path = CSV_PATH, key_path: Path = KEY_PATH) -> list[Candidate]:
    # `utf-8-sig` on BOTH, matching the pack's own tests. A spreadsheet's "CSV UTF-8" writes a
    # BOM, it lands on the first header field, and every `row["item"]` becomes a KeyError. The
    # pack's tests moved to utf-8-sig and this scorer, one directory over, did not: the reader
    # that consumes the labels for the published number was the one still exposed.
    #
    # The paths are parameters so a test can hand this a BOM'd copy. Defaulting to the committed
    # pack keeps every caller unchanged, and the alternative — corrupting the real files to test
    # the reader — is not available on 38 rows of irreplaceable human labelling.
    key = json.loads(key_path.read_text(encoding="utf-8-sig"))
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        Candidate(
            item=row["item"],
            # The key holds the RAW sentence; the CSV holds the CSV-injection-defended form.
            # The arms must read what the PEP says, so the key is the source of truth here.
            sentence=key[row["item"]]["evidence_sentence"],
            target=key[row["item"]]["candidate_target"],
            source_pep=key[row["item"]]["source_pep"],
            verdict=row["your_verdict_Y_or_N"].strip().upper(),
        )
        for row in rows
    ]


def rules_arm(pack: list[Candidate], peps_dir: Path, corpus: set[str]) -> ArmResult:
    """R1. `recall.fix`'s refusal rules, unmodified, over PEP reference forms.

    The marker is located inside the REAL body rather than inside the isolated sentence, because
    `_is_reported_speech` scans backwards to the nearest clause boundary and would see a sentence
    fragment as its own subject. Feeding it the fragment would make the rule weaker than it is,
    which would flatter the model arm.
    """
    result = ArmResult(arm="R1-rules")
    for cand in pack:
        if _is_index(f"{cand.source_pep}.rst"):
            result.refused[cand.item] = "index document"
            continue

        raw = (peps_dir / f"{cand.source_pep}.rst").read_text(encoding="utf-8", errors="replace")
        # The PARAGRAPH, unwrapped, not the raw body. RST wraps at ~79 columns, so an adjudicated
        # sentence spans two lines in the file and `raw.find(sentence)` misses it — it missed 30
        # of 38 and recorded them as refusals, which would have read as a selective arm rather
        # than an arm that never saw the candidates. The paragraph is also the right context for
        # `_is_reported_speech`, which scans back to a clause boundary.
        body = next((para for para in paragraphs(raw) if cand.sentence in para), None)
        if body is None:
            # NOT a refusal. The arm did not decline this candidate, it never saw it, and
            # counting the two together is what let an R1 run that located nothing publish as a
            # careful arm. Recorded in its own bucket, which the write-site validator gates.
            result.not_read[cand.item] = "sentence not located in body"
            continue
        at = body.find(cand.sentence)

        matched = None
        for pattern in (_ACTIVE_RE, _PASSIVE_RE):
            for m in pattern.finditer(body, at, at + len(cand.sentence)):
                if _pep_stem(m) == cand.target:
                    matched = m
                    break
            if matched:
                break
        if matched is None:
            result.refused[cand.item] = "no marker names this target within the window"
            continue

        # ⚠️ The spans are `recall.fix._accept`'s, computed the same way, because importing the
        # RULES unmodified and then handing them different text is not running the same arm.
        # The first version did exactly that, in both directions:
        #
        #   `_is_hedged` got the 40 characters AFTER the whole match instead of the text between
        #   the marker and the reference, which INVERTS it on the form the experiment is built
        #   around: "supersedes/augments :pep:`387`" was proposed where production refuses, and
        #   "supersedes :pep:`387` or something" was refused where production accepts.
        #
        #   `_is_partial_scope` got `body[end("marker"):start()]`, which is a start > stop slice
        #   and empty by construction, since both patterns BEGIN with the marker group. Measured
        #   over the pack: empty on 9 of 9 matched rows. The `or` fallback then handed it a span
        #   that includes the reference text, which production deliberately excludes.
        #
        # Neither changed R1's number on this pack, verified row by row, and both sat on the
        # residual class the pre-registration names as decisive.
        ref_start = min(
            matched.start(group)
            for group in range(2, (matched.lastindex or 1) + 1)
            if matched.group(group)
        )
        between = body[matched.end("marker") : ref_start]

        if _is_reported_speech(body, matched.start()):
            result.refused[cand.item] = "reported speech"
        elif _is_hedged(body, matched.start(), between):
            result.refused[cand.item] = "hedged"
        elif _is_partial_scope(between):
            result.refused[cand.item] = "partial scope"
        elif cand.target not in corpus:
            result.refused[cand.item] = "target not in corpus"
        elif cand.target == cand.source_pep:
            result.refused[cand.item] = "self reference"
        else:
            result.proposed.append(cand.item)
    return result


def model_arm(
    pack: list[Candidate],
    peps_dir: Path,
    corpus: set[str],
    engine: Any,
    cache: Any = None,
    status_vocabulary: tuple[str, ...] | None = None,
) -> ArmResult:
    """M1. The shipped extraction engine, one call per SOURCE document, not per candidate.

    The engine reads a whole document and returns every claim it finds, so 38 candidates spanning
    ~30 distinct PEPs cost ~30 calls rather than 38. Each candidate is then decided by asking
    whether that document's claims name its target.

    Scored DIRECTION-AGNOSTICALLY, matching the pack. `candidate_target` is simply the PEP the
    sentence names, and the adjudicator judged whether a real supersession edge exists with it in
    either direction, because the corpus states both voices ("this PEP is replaced by X" and "X
    ... to be replaced by this PEP").

    Direction-agnostic scoring was ALSO forced, at the time the first run was scored, by a gap in
    the prompt: `SupersessionClaim.superseded` is documented as "the document this file replaces"
    and `_prompt.py` never told the MODEL that, so the direction of M1's output was unspecified by
    construction. `truth-extraction-prompt-v2` closes that gap and states the direction. The
    scoring here stays direction-agnostic anyway, because the PACK is: the adjudicator judged
    whether a real edge exists with the named PEP, not which end of it the source sits on, and
    re-scoring against a direction the labels never recorded would be reading a judgement that
    was not made. Direction is measured separately, in `tests/test_truth_extraction_direction.py`.
    """
    from recall.truth_extraction.extract import extract_file_claims

    result = ArmResult(arm="M1-model")
    names = sorted(corpus)
    by_source: dict[str, set[str]] = {}
    calls = 0

    for source_pep in sorted({c.source_pep for c in pack}):
        path = peps_dir / f"{source_pep}.rst"
        extraction = extract_file_claims(
            file=f"{source_pep}.rst",
            text=path.read_text(encoding="utf-8", errors="replace"),
            corpus_names=names,
            engine=engine,
            cache=cache,
            status_vocabulary=status_vocabulary,
        )
        calls += 1
        result.cache_hits += bool(extraction.cached)
        _record_identity(result, extraction)
        # `isinstance`, not `claim.kind ==`: the union has four members and only one carries
        # `superseded`, so the string check narrows nothing for the type checker and would read
        # an attribute the other three do not have if a kind were ever mislabelled.
        by_source[source_pep] = {
            claim.superseded
            for claim in extraction.claims
            if isinstance(claim, SupersessionClaim)
        }
        if extraction.batch_rejection is not None:
            # The whole file was refused. Recorded rather than silently counted as "no claims":
            # a run where every call fails looks EXACTLY like a selective arm otherwise, and the
            # first M1 run reported 30 total failures as 0 proposals before this existed.
            result.batch_failures[source_pep] = extraction.batch_rejection.reason

    for cand in pack:
        if cand.source_pep in result.batch_failures:
            # The document was refused whole at a batch rung, so the model never read this
            # candidate. Recorded apart from the semantic refusals: lumped in with them, half a
            # broken run's candidates read as an arm that considered them and declined.
            result.not_read[cand.item] = f"document refused whole: {result.batch_failures[cand.source_pep]}"
        elif cand.target in by_source.get(cand.source_pep, set()):
            result.proposed.append(cand.item)
        else:
            result.refused[cand.item] = "no supersession claim naming this target"

    result.calls = calls
    return result


def score(result: ArmResult, pack: list[Candidate]) -> dict[str, Any]:
    """Precision over decided proposals. Blank-labelled rows leave the denominator."""
    label = {c.item: c.verdict for c in pack}
    proposed = result.proposed
    decided = [i for i in proposed if label[i] in {"Y", "N"}]
    undecidable = [i for i in proposed if label[i] == ""]
    true_positive = [i for i in decided if label[i] == "Y"]

    positives = [c.item for c in pack if c.verdict == "Y"]
    # Wilson, not bootstrap. The decision rule is keyed on the LOWER BOUND rather than the point
    # estimate, and at these n a percentile bootstrap of a degenerate sample reports certainty
    # from a handful of observations. See `recall/eval/metrics.wilson_ci`.
    lower, upper = wilson_ci([label[i] == "Y" for i in decided])
    return {
        "arm": result.arm,
        "proposed": len(proposed),
        "proposed_scored": len(decided),
        "proposed_undecidable_excluded": len(undecidable),
        "true_positive": len(true_positive),
        "false_positive": len(decided) - len(true_positive),
        # None, not 0.0, on an empty denominator: an arm that proposed nothing DECLINED to answer,
        # and reporting that as precision 0.0 reads as "it was wrong" — the distinction the
        # pre-registration's baseline depends on. None rather than NaN because this is written to
        # JSON, where NaN is an extension a strict parser rejects.
        "precision": (len(true_positive) / len(decided)) if decided else None,
        "precision_wilson_lower": lower if decided else None,
        "precision_wilson_upper": upper if decided else None,
        # NULL, not 0.0. Nothing in either arm ever populates `referred`: no extraction path
        # emits a `requires_review` status, so this was 0.0 by CONSTRUCTION. The
        # pre-registration predicts 0.15 [0.05, 0.40] for it, so publishing the structural zero
        # would have scored a prediction as falsified using a measurement that was never taken.
        # `null` says "not measured", which is what happened.
        "referral_rate": None,
        "referral_rate_note": (
            "NOT MEASURED: no extraction path emits a review-required status, so there is "
            "nothing to count. P6 is unscored rather than falsified."
        ),
        "recall_vs_adjudicated_positives": (
            len(true_positive) / len(positives) if positives else None
        ),
        "adjudicated_positives": len(positives),
        "refusal_reasons": _tally(result.refused),
        # Apparatus, not judgement, and gated at the write site.
        "candidates_not_read": len(result.not_read),
        "not_read_reasons": _tally(result.not_read),
    }


def preregistered_verdict(report: dict[str, Any], sibling_proposals: int | None = None) -> str:
    """The pre-registration's decision rule, applied by the runner rather than by a reader.

    Keyed on the Wilson LOWER BOUND, not the point estimate. The rule is fixed in
    `results/truth_extraction/PREREGISTRATION-prose-extraction.md` and reproduced here as code so
    that it is applied to the number instead of being reread after seeing it.

    ⚠️ One reading is mine and is stated rather than hidden. The rule gives each tier a point band
    AND a Wilson gate, and says nothing about a result that meets the band and misses the gate:
    0.90 precision on n=10 has a Wilson lower around 0.60, which is inside "batch reviewable" by
    point estimate and outside it by gate. It lands in the tier BELOW, because the gate exists to
    stop a small sample buying a tier, and the conservative reading is the only one consistent
    with that.

    ⚠️ **The underpowered clause is CROSS-ARM**: the pre-registration reads "fewer than 10
    proposals in EITHER arm". A per-arm reading is not the rule, and the difference is live —
    R1 proposed 9, so under the real clause M1 cannot be given a tier at all however it scores.
    `sibling_proposals` is that other arm's count. Passing `None` means it is unknown, and the
    honest answer is then `PENDING SIBLING ARM` rather than a tier chosen from half the inputs.

    ⚠️ **`SUSPICIOUS` sits ABOVE the tiers and swallows `HIGH CONFIDENCE`.** Its conditions
    (precision at or above 0.95, n at or above 20) are a superset of high confidence's, and its
    action is "run the four upper falsifier checks in order first", which is a human step. So
    this function can never return `HIGH CONFIDENCE`, and that is the rule working: a result too
    good to believe must be disbelieved before it is tiered, not tiered and then questioned.

    The P7 clause — any transplanted fixture proposed fails the public bridge regardless of every
    other number — is NOT applied here. It is measured by the `fixtures` arm against a different
    corpus, and folding a verdict this function cannot see into its output would be asserting a
    result it never read.
    """
    scored, precision = report["proposed_scored"], report["precision"]
    if report["proposed"] == 0:
        # A non-measurement, not a null. An arm that proposes nothing has not been compared.
        #
        # Stated deviation: the table puts this row BELOW the underpowered one, which makes it
        # unreachable, since proposing 0 is also "fewer than 10". Both refuse to choose a tier,
        # so the operational consequence is identical and only the diagnosis differs; "proposed
        # nothing" is the more precise one, so it is reported.
        return "VACUOUS ARM"
    if scored < 10:
        return "UNDERPOWERED"
    lower, upper = report["precision_wilson_lower"], report["precision_wilson_upper"]
    # `+ 1e-9`, because the rule says "half width ABOVE 0.30" and binary floating point does not
    # represent 0.30. A half-width that is exactly 0.30 in decimal computes as
    # 0.30000000000000004 for some bound pairs and 0.29999999999999993 for others, so without a
    # tolerance the verdict at the boundary depends on which arithmetic produced the interval.
    if (upper - lower) / 2 > 0.30 + 1e-9:
        return "UNDERPOWERED"
    if sibling_proposals is None:
        return "PENDING SIBLING ARM"
    if sibling_proposals < 10:
        return "UNDERPOWERED"
    if precision >= 0.95 and scored >= 20:
        return "SUSPICIOUS"
    if precision < 0.50:
        return "LINT POINTER"
    if precision < 0.80:
        return "REVIEWING AID"
    if precision < 0.95:
        return "BATCH REVIEWABLE" if lower >= 0.70 else "REVIEWING AID"
    # At or above 0.95 with n < 20, which `SUSPICIOUS` above does not cover. The high-confidence
    # row needs n at or above 20, so this falls to the tier its Wilson gate supports.
    return "BATCH REVIEWABLE" if lower >= 0.70 else "REVIEWING AID"


def _record_identity(result: ArmResult, extraction: Any) -> None:
    """Pin the extractor's identity, and refuse a run that changed it mid-way.

    A run that switched prompt revision or model between file 3 and file 4 would otherwise be
    written out under whichever identity happened to be read last, and the artifact would name an
    extractor that produced only part of it.
    """
    identity = {
        "prompt_revision": extraction.prompt_revision,
        "model_id": extraction.model_id,
        "model_revision": extraction.revision,
        "engine_id": extraction.engine_id,
        # From the EXTRACTION, which is where `FileExtraction` documents it as part of the audit
        # identity: "changing it IS rewording the prompt". Taking it from the CLI flag instead
        # meant a cache hit under a different vocabulary was recorded as the flag's value, which
        # is the defect `extract.py`'s `replace(cached, cached=True)` exists to prevent, moved
        # one layer up. It also belongs in the mid-run identity check for the same reason.
        "status_vocabulary": ",".join(extraction.status_vocabulary),
    }
    if not result.identity:
        result.identity = identity
    elif result.identity != identity:
        raise SystemExit(
            f"the extractor's identity changed mid-run: {result.identity} then {identity}. "
            f"The artifact would name an extractor that produced only part of it"
        )


def fixtures_arm(
    engine: Any, cache: Any = None, fixtures: Path = FIXTURES
) -> tuple[ArmResult, dict[str, Any]]:
    """P7, the load-bearing public prediction: M1 must refuse all four transplanted failures.

    These are the four survivors of `recall/fix.py`'s mechanical rules on the private 792-memo
    corpus, every one wrong on review: reported speech, two partial scope, one hedged. They are
    the public bridge to a private result, because anyone can run them.

    ⚠️ A refusal only counts if it is SEMANTIC. Dropping a claim because its target is not in the
    corpus would be the target-resolution rung doing the work, and P7 would be measuring the
    corpus list rather than the model's reading. So the corpus is built from the fixtures' own
    wikilinks: every target they name IS resolvable, and a refusal has to come from the language.
    The rung that refused each claim is recorded so a reader can check that.
    """
    from recall.truth_extraction.extract import extract_file_claims

    paths = sorted(fixtures.glob("*.md"))
    if not paths:
        raise SystemExit(f"no fixtures under {fixtures}")
    names = sorted(
        {p.stem for p in paths}
        | {m for p in paths for m in re.findall(r"\[\[([^\]]+)\]\]", p.read_text(encoding="utf-8"))}
    )

    result = ArmResult(arm="M1-fixtures")
    detail: dict[str, Any] = {}
    for path in paths:
        extraction = extract_file_claims(
            file=path.name,
            text=path.read_text(encoding="utf-8"),
            corpus_names=names,
            engine=engine,
            cache=cache,
        )
        result.calls += 1
        result.cache_hits += bool(extraction.cached)
        _record_identity(result, extraction)
        proposed = [c for c in extraction.claims if isinstance(c, SupersessionClaim)]
        if proposed:
            result.proposed.append(path.stem)
        else:
            result.refused[path.stem] = "no supersession claim"
        if extraction.batch_rejection is not None:
            result.batch_failures[path.stem] = extraction.batch_rejection.reason
        detail[path.stem] = {
            "proposed": [c.superseded for c in proposed],
            "rejection_rungs": sorted({r.rung for r in extraction.rejections}),
        }
    return result, {
        "arm": "M1-fixtures",
        "fixtures": len(paths),
        "refused": len(result.refused),
        "proposed": len(result.proposed),
        "proposed_items": result.proposed,
        "corpus_names": names,
        "per_fixture": detail,
        # P7 predicts exactly 4 of 4. Stated as the prediction it is, so a reader does not have
        # to hold the pre-registration open beside the artifact.
        "p7_prediction": "4 of 4 refused",
        "p7_holds": len(result.refused) == len(paths),
    }


def _tally(refused: dict[str, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for reason in refused.values():
        out[reason] = out.get(reason, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _git(repo: Path, *args: str) -> str | None:
    """A git fact, `""` when git answered with nothing, or `None` when it could not answer.

    The two were collapsed into `""`, so `recall_tree_dirty` recorded FALSE both for a clean tree
    and for a machine with no git on PATH, and the fatal message blamed the checkout for a
    missing binary. A provenance field that cannot tell "no" from "I could not look" is not
    provenance.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def pack_digest() -> str:
    """The adjudication the score is measured against, pinned into the artifact.

    Both files, because the CSV holds the verdicts and the key holds the sentences they answer,
    and an artifact that pinned only the verdicts could not tell a relabelled pack from a
    re-evidenced one. Same argument as the digest in
    `tests/test_truth_extraction_adjudication.py`, at a coarser grain: this one only has to say
    WHICH pack, not what changed in it.
    """
    return hashlib.sha256(CSV_PATH.read_bytes() + KEY_PATH.read_bytes()).hexdigest()


def build_provenance(
    peps_dir: Path, invocation: str, result: ArmResult, *, corpus_files: int | None = None
) -> dict[str, Any]:
    """Everything a reader needs to say WHICH corpus and WHICH code produced this number.

    Three ways this recorded a confident lie, all reproduced:

    - it took HEAD of whatever git repository happened to contain `--peps-dir`. Pointed at a
      throwaway repo holding 52 copied PEP files, it recorded that repo's commit as the corpus
      version and validated cleanly against a census of 733;
    - it had no dirty flag for the CORPUS, only for `recall`, so nine uncommitted edits moved
      precision from 0.375 to 0.4286 under a byte-identical provenance block;
    - it treated an unknown SHA as fatal and a wrong one as fine.
    """
    repo = Path(__file__).resolve().parents[3]
    peps_repo = peps_dir if (peps_dir / ".git").exists() else peps_dir.parent
    peps_sha = _git(peps_repo, "rev-parse", "HEAD")
    if not peps_sha:
        # Refused rather than defaulted. An arm result whose corpus version is unknown cannot be
        # compared with the census, and the census is what its recall ceiling comes from.
        raise SystemExit(
            f"cannot read the PEPs commit under {peps_dir}: an arm result that does not name its "
            f"corpus version is not attributable. Point --peps-dir at a git checkout, and check "
            f"that `git` is on PATH — a missing binary produced this same message"
        )
    peps_dirty = _git(peps_repo, "status", "--porcelain")
    if peps_dirty is None:
        raise SystemExit(f"cannot read the working-tree state of {peps_repo}")
    if peps_dirty:
        # As fatal as an unknown SHA, and for the identical reason: the recorded commit would
        # name a corpus that is not the one the arm read.
        raise SystemExit(
            f"{peps_repo} has uncommitted changes, so {peps_sha[:12]} does not name the corpus "
            f"this run would measure. Commit or stash them, or point at a clean checkout"
        )
    recall_dirty = _git(repo, "status", "--porcelain")
    provenance: dict[str, Any] = {
        "peps_sha": peps_sha,
        "peps_tree_dirty": False,  # refused above, so this is a fact rather than a hope
        "recall_commit": _git(repo, "rev-parse", "--short", "HEAD") or "unknown",
        "recall_tree_dirty": None if recall_dirty is None else bool(recall_dirty),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "invocation": invocation,
        "pack_digest": pack_digest(),
    }
    if corpus_files is not None:
        _check_against_census(corpus_files)
    provenance.update(result.identity)
    return provenance


def _check_against_census(corpus_files: int) -> None:
    """Invariant I9: the file count must match the frozen census, or the corpus is not that one.

    Nothing checked it, so an artifact could record 52 files against a census of 733 and still
    validate. The census is where this experiment's recall ceiling comes from; measuring a
    different corpus against it is not a weaker result, it is a different question.
    """
    census_path = Path(__file__).resolve().parents[3] / "results/truth_extraction/census.json"
    if not census_path.exists():  # pragma: no cover - the census is committed
        raise SystemExit(f"{census_path} is missing; the corpus cannot be checked against it")
    expected = json.loads(census_path.read_text(encoding="utf-8"))["n_files"]
    if corpus_files != expected:
        raise SystemExit(
            f"the corpus holds {corpus_files} files and the frozen census counted {expected}. "
            f"This is not the corpus the recall ceiling was computed on"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peps-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=["rules", "model", "fixtures"], default="rules")
    parser.add_argument("--status-vocabulary", help="comma-separated, e.g. the PEP "
                        "statuses. Omit for the shipped memo-shaped set.")
    parser.add_argument("--cache", type=Path, help="sqlite extraction cache, so a re-run "
                        "does not pay twice and repeat runs are byte-identical")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--sibling", type=Path,
        help="the OTHER arm's artifact. The pre-registered underpowered clause reads 'fewer "
             "than 10 proposals in EITHER arm', so a tier cannot be chosen from one arm alone; "
             "without this the verdict is PENDING SIBLING ARM",
    )
    args = parser.parse_args()

    invocation = "python -m benchmarks.labelling.truth_extraction.run_arms " + " ".join(
        f"--{k.replace('_', '-')} {v}"
        for k, v in sorted(vars(args).items())
        if v is not None and k != "out"
    )

    # Parsed BEFORE anything is paid for. An empty parse reached the prompt as an empty
    # vocabulary and then made the artifact unwritable at the end, after every call.
    vocab: tuple[str, ...] | None = None
    if args.status_vocabulary is not None:
        vocab = tuple(v.strip() for v in args.status_vocabulary.split(",") if v.strip())
        if not vocab:
            raise SystemExit(f"--status-vocabulary {args.status_vocabulary!r} parsed to no words")

    sibling_proposals: int | None = None
    if args.sibling:
        sibling_proposals = int(
            json.loads(args.sibling.read_text(encoding="utf-8"))["proposed"]
        )

    def _engine_and_cache() -> tuple[Any, Any]:
        from recall.truth_extraction import resolve_extraction_engine

        engine = resolve_extraction_engine()
        if engine is None:
            raise SystemExit(
                "extraction is off: set RECALL_TRUTH_EXTRACTION=1 and "
                "RECALL_TRUTH_EXTRACTION_ENGINE=openai"
            )
        cache = None
        if args.cache:
            from recall.truth_extraction._sqlite_cache import SqliteExtractionCache

            cache = SqliteExtractionCache(args.cache)
        return engine, cache

    # ⚠️ Provenance is resolved BEFORE the first engine call, not after the last. It refuses a
    # non-git corpus, a dirty corpus and a file count that disagrees with the census, all of
    # which are knowable at startup — and a model run used to discover them after paying for 30
    # calls and then discard the result.
    corpus = {p.stem for p in args.peps_dir.glob("pep-*.rst")}
    if not corpus:
        raise SystemExit(f"no pep-*.rst under {args.peps_dir}")
    provenance = build_provenance(
        args.peps_dir, invocation, ArmResult(arm="pre-flight"),
        corpus_files=None if args.arm == "fixtures" else len(corpus),
    )

    if args.arm == "fixtures":
        # A different corpus and a different question, so it does not go through `score`: there is
        # no adjudicated pack here and no precision to compute. It is a 4-of-4 refusal check.
        engine, cache = _engine_and_cache()
        result, report = fixtures_arm(engine, cache)
        report["model_calls"] = result.calls
        report["cache_hits"] = result.cache_hits
        report["batch_failures"] = result.batch_failures
        report["_provenance"] = {**provenance, **result.identity}
        _emit(report, args.out, validator=validate_fixtures_result)
        return

    pack = load_pack()
    if args.arm == "rules":
        result = rules_arm(pack, args.peps_dir, corpus)
    else:
        engine, cache = _engine_and_cache()
        result = model_arm(pack, args.peps_dir, corpus, engine, cache, vocab)

    report = score(result, pack)
    report["verdict"] = preregistered_verdict(report, sibling_proposals)
    report["sibling_proposals"] = sibling_proposals
    report["proposed_items"] = result.proposed
    report["corpus_files"] = len(corpus)
    report["model_calls"] = result.calls
    report["cache_hits"] = result.cache_hits
    report["batch_failures"] = result.batch_failures
    report["documents_refused_whole"] = len(result.batch_failures)
    report["_provenance"] = {**provenance, **result.identity}
    # Recorded in the provenance, not beside the counts: it is part of what produced the number,
    # and `_REQUIRED_MODEL_PROVENANCE` makes a model arm that omits it unwritable. Taken from the
    # EXTRACTION where one ran, not from the CLI argument: a cache hit carries the vocabulary the
    # warm run used, and `extract.py` returns `replace(cached, cached=True)` precisely so that it
    # does. Reading the flag back would have reintroduced one layer up the defect that fixed.
    report["_provenance"]["status_vocabulary"] = result.identity.get(
        "status_vocabulary",
        ",".join(vocab) if vocab else ("shipped default" if args.arm == "model" else None),
    )
    _emit(report, args.out)


def _emit(
    report: dict[str, Any],
    out: Path | None,
    *,
    validator: Callable[[Mapping[str, Any]], None] = validate_arm_result,
) -> None:
    """Validate, print, then write. In that order, so a refused payload is never on disk.

    The validator is a parameter with a default rather than a flag, because a flag would have a
    value that turns validation OFF, and the one artifact that needed a different check is
    exactly the one that would have been written with it off.
    """
    validator(report)
    print(json.dumps(report, indent=2))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
