"""Statistical analysis of the head-to-head benchmark artifacts.

::

    python -m benchmarks.analyze --compare A.json B.json
    python -m benchmarks.analyze --curve 'benchmarks/results/*.json' --plot curve.png

Reads the JSON artifacts written by ``benchmarks.run``. Never writes one, never touches the
database, never calls a model: everything here is offline arithmetic over files that have already
been paid for, so it can be re-run as often as a reviewer likes.

Why a PAIRED test rather than the per-arm confidence intervals the artifact already carries:
both arms answer the IDENTICAL question set. Comparing two independent Wilson intervals throws
that away and asks the much weaker question "could these two rates have come from the same
population", with the between-question variance (some LOCOMO questions are hard for everything)
left in the noise term. McNemar conditions on the questions and looks only at the DISCORDANT
pairs -- the questions where the two systems disagreed -- which is the only place evidence about
a difference actually lives. On a 150-question set the paired test routinely resolves a
difference that overlapping independent intervals call inconclusive, and it is the analysis a
hostile reader will ask for if the published claim rests on the unpaired one.

The second thing published numbers need is `discrimination`. A memory system that abstains on
every question scores a perfect 1.0 on adversarial abstention; the metric alone cannot tell
refusal from discrimination. Youden's J (abstain-when-you-should minus abstain-when-you-should-
not) is 0.0 for that system, which is the whole point of reporting it beside the rate.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from fractions import Fraction
from glob import glob
from pathlib import Path
from typing import Any

#: Below this many discordant pairs the chi-square approximation to the binomial is unreliable
#: (the usual textbook rule of thumb; b+c is the effective sample size of a McNemar test, not the
#: number of questions), so the exact binomial is used instead. At or above it the two agree to
#: several decimals and the closed form is reported. The threshold is stated in the output's
#: ``test`` field rather than left implicit, because "which test" is the first thing a reviewer
#: checks and the second thing an author is tempted to pick after seeing the p-values.
EXACT_MAX_DISCORDANT = 25

_EXACT = "exact_binomial"
_CHI2 = "chi2_continuity_corrected"


# -- outcome predicates (named, not lambdas: they appear in tracebacks and in the tests) --------

def _is_answerable(outcome: Mapping[str, Any]) -> bool:
    return not bool(outcome["is_adversarial"])


def _is_adversarial(outcome: Mapping[str, Any]) -> bool:
    return bool(outcome["is_adversarial"])


def _was_correct(outcome: Mapping[str, Any]) -> bool:
    return bool(outcome["correct"])


def _abstained(outcome: Mapping[str, Any]) -> bool:
    return bool(outcome["abstained"])


# -- p-values, without scipy -------------------------------------------------------------------

def exact_binomial_two_sided(successes: int, n: int) -> float:
    """Two-sided exact binomial p-value against p=0.5, from `math.comb` alone.

    Under McNemar's null the b+c discordant pairs are b successes from a fair coin, so the
    two-sided p-value is twice the smaller tail (the null is symmetric, which is what makes
    doubling legitimate here and not a shortcut). Clamped at 1.0 for the b == c case, where
    doubling a tail that already contains the centre otherwise exceeds 1.

    Computed as an exact `Fraction` and converted once, so the answer does not depend on the
    order the binomial coefficients happen to be summed in.
    """
    if n <= 0:
        return 1.0
    tail = min(successes, n - successes)
    cumulative = sum(math.comb(n, i) for i in range(tail + 1))
    return min(1.0, float(Fraction(2 * cumulative, 2**n)))


def _chi2_1df_sf(statistic: float) -> float:
    """Upper-tail probability of a chi-square with 1 degree of freedom.

    A chi-square on 1 df is the square of a standard normal, so its survival function is
    ``2 * (1 - Phi(sqrt(x)))``, which `math.erfc` gives exactly. No scipy, no table.
    """
    return math.erfc(math.sqrt(statistic / 2.0))


def _mcnemar_from_discordant(a_only: int, b_only: int) -> tuple[float, float, str]:
    """(statistic, p_value, test) for the discordant counts of a McNemar table."""
    discordant = a_only + b_only
    if discordant < EXACT_MAX_DISCORDANT:
        return float(a_only), exact_binomial_two_sided(a_only, discordant), _EXACT
    statistic = (abs(a_only - b_only) - 1.0) ** 2 / discordant
    return statistic, _chi2_1df_sf(statistic), _CHI2


# -- pairing -----------------------------------------------------------------------------------

def _by_question_id(
    outcomes: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for outcome in outcomes:
        question_id = str(outcome["question_id"])
        if question_id in indexed:
            raise ValueError(
                f"arm {label} has duplicate question_id {question_id!r}; pairing would silently "
                "drop one of them"
            )
        indexed[question_id] = outcome
    return indexed


def _aligned(
    a_outcomes: Sequence[Mapping[str, Any]], b_outcomes: Sequence[Mapping[str, Any]]
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Pair the two arms question-by-question, or refuse.

    A paired test is only valid if the arms answered the same questions, so a mismatch is an
    error and not a silent intersection: quietly dropping the unmatched questions would produce a
    number that looks like a paired comparison of the full set and is not one.
    """
    a_index = _by_question_id(a_outcomes, "A")
    b_index = _by_question_id(b_outcomes, "B")
    if a_index.keys() != b_index.keys():
        only_a = sorted(a_index.keys() - b_index.keys())
        only_b = sorted(b_index.keys() - a_index.keys())
        raise ValueError(
            "the two arms did not answer the same questions, so they cannot be paired: "
            f"{len(only_a)} only in A (e.g. {only_a[:3]}), "
            f"{len(only_b)} only in B (e.g. {only_b[:3]})"
        )
    pairs = []
    for question_id in sorted(a_index):
        a_outcome, b_outcome = a_index[question_id], b_index[question_id]
        if bool(a_outcome["is_adversarial"]) != bool(b_outcome["is_adversarial"]):
            raise ValueError(
                f"question {question_id!r} is adversarial in one arm and answerable in the "
                "other; the arms were scored against different question labels"
            )
        pairs.append((a_outcome, b_outcome))
    return pairs


def _paired_test(
    a_outcomes: Sequence[Mapping[str, Any]],
    b_outcomes: Sequence[Mapping[str, Any]],
    select: Callable[[Mapping[str, Any]], bool],
    scored: Callable[[Mapping[str, Any]], bool],
) -> dict[str, Any]:
    """McNemar over the pairs `select` keeps, scoring each side with `scored` (True = better)."""
    both_correct = both_wrong = a_only = b_only = 0
    for a_outcome, b_outcome in _aligned(a_outcomes, b_outcomes):
        if not select(a_outcome):
            continue
        a_good, b_good = scored(a_outcome), scored(b_outcome)
        if a_good and b_good:
            both_correct += 1
        elif a_good:
            a_only += 1
        elif b_good:
            b_only += 1
        else:
            both_wrong += 1

    statistic, p_value, test = _mcnemar_from_discordant(a_only, b_only)
    return {
        "n_paired": both_correct + both_wrong + a_only + b_only,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "a_only": a_only,
        "b_only": b_only,
        "statistic": statistic,
        "p_value": p_value,
        "test": test,
    }


def paired_mcnemar(
    a_outcomes: Sequence[Mapping[str, Any]], b_outcomes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """McNemar on ACCURACY over the answerable questions both arms answered.

    ``a_only`` is the classical *b* cell (A right, B wrong) and ``b_only`` the *c* cell; the
    concordant cells carry no information about a difference and are reported only so a reader
    can reconstruct the full 2x2 table. Adversarial questions are excluded because they have no
    gold answer to be correct about -- ``correct`` is ``None`` for them by construction
    (`benchmarks.pipeline.Outcome`), and counting ``None`` as wrong on both arms would inflate
    the concordant cells with questions that were never scored for accuracy at all.
    """
    return _paired_test(a_outcomes, b_outcomes, _is_answerable, _was_correct)


def paired_abstention_mcnemar(
    a_outcomes: Sequence[Mapping[str, Any]], b_outcomes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """McNemar on ABSTENTION over the adversarial questions -- the refusal axis, paired.

    Same table, different question subset and different scored event: abstaining on an
    unanswerable question is the good outcome, so ``a_only`` counts questions where A refused and
    B answered. Read it beside `paired_mcnemar`: winning here while losing there is exactly the
    blanket-conservatism trade `discrimination` exists to price.
    """
    return _paired_test(a_outcomes, b_outcomes, _is_adversarial, _abstained)


# -- discrimination ----------------------------------------------------------------------------

def discrimination(aggregate: Mapping[str, Any]) -> float | None:
    """Youden's J for the abstention decision: abstain-when-right minus abstain-when-wrong.

    ``adversarial_abstention.rate - answerable_false_abstain.rate``. A system that abstains on
    everything gets 1.0 - 1.0 = 0.0; a system that abstains on nothing gets 0.0 - 0.0 = 0.0. Only
    a system that TELLS THE TWO CASES APART scores above zero, which is what makes J and not the
    raw abstention rate the number to publish.

    ``None`` when either rate is ``None`` -- the artifact writes that for an empty subset (a
    per-category block with no adversarial questions, say), and inventing 0.0 there would report
    a measured non-discriminating system where there is simply no measurement.
    """
    adversarial = aggregate.get("adversarial_abstention") or {}
    answerable = aggregate.get("answerable_false_abstain") or {}
    abstain_rate: float | None = adversarial.get("rate")
    false_abstain_rate: float | None = answerable.get("rate")
    if abstain_rate is None or false_abstain_rate is None:
        return None
    return round(abstain_rate - false_abstain_rate, 4)


# -- the curve ---------------------------------------------------------------------------------

def _k_of(doc: Mapping[str, Any]) -> int | None:
    """Retrieval budget of a run: top-level ``k`` if present, else the recorded ``config.k``."""
    top_level: int | None = doc.get("k")
    if top_level is not None:
        return top_level
    config = doc.get("config") or {}
    budget: int | None = config.get("k")
    return budget


def _rate_of(aggregate: Mapping[str, Any], key: str) -> float | None:
    block = aggregate.get(key) or {}
    rate: float | None = block.get("rate")
    return rate


def _n_of(aggregate: Mapping[str, Any], key: str) -> int:
    block = aggregate.get(key) or {}
    n: int = int(block.get("n", 0))
    return n


def _ctx_tokens_mean(aggregate: Mapping[str, Any]) -> float | None:
    context = aggregate.get("retrieved_context") or {}
    tokens = context.get("tokens_approx") or {}
    mean: float | None = tokens.get("mean")
    return mean


def curve_points(paths: Iterable[Path | str]) -> list[dict[str, Any]]:
    """One row per results artifact, sorted by arm then by mean retrieved-context tokens.

    The x axis is CONTEXT TOKENS, not ``k``. The two arms return different amounts of text per
    retrieved item -- RE-call joins verbatim dialogue turns, Mem0 joins LLM-compressed facts --
    so equal k is not equal budget, and an accuracy-vs-k plot silently compares a system at one
    context size against a system at another. Plotting against what the generator was actually
    handed is the comparison the reader thinks they are looking at.
    """
    points: list[dict[str, Any]] = []
    for path in paths:
        path = Path(path)
        doc: dict[str, Any] = _load_published(path)
        aggregate: dict[str, Any] = doc.get("aggregate") or {}
        points.append(
            {
                "arm": str(doc.get("arm", "?")),
                "k": _k_of(doc),
                "ctx_tokens_mean": _ctx_tokens_mean(aggregate),
                "accuracy": _rate_of(aggregate, "answerable_accuracy"),
                "adversarial_abstention": _rate_of(aggregate, "adversarial_abstention"),
                "false_abstain": _rate_of(aggregate, "answerable_false_abstain"),
                "discrimination": discrimination(aggregate),
                "n_answerable": _n_of(aggregate, "answerable_accuracy"),
                "n_adversarial": _n_of(aggregate, "adversarial_abstention"),
                "path": str(path),
            }
        )
    points.sort(key=_curve_sort_key)
    return points


def _curve_sort_key(point: Mapping[str, Any]) -> tuple[str, float]:
    tokens = point["ctx_tokens_mean"]
    return (str(point["arm"]), math.inf if tokens is None else float(tokens))


def _series(
    points: Sequence[Mapping[str, Any]], arm: str, key: str
) -> tuple[list[float], list[float]]:
    """The (x, y) pairs of one arm's `key` series, dropping points where either is missing."""
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        if point["arm"] != arm:
            continue
        x, y = point["ctx_tokens_mean"], point[key]
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    return xs, ys


def plot_curve(points: Sequence[Mapping[str, Any]], out_path: Path | str) -> Path:
    """Two stacked panels: accuracy, then the abstention pair, both against context tokens.

    matplotlib is imported here rather than at module scope: it lives in the ``eval`` extra, and
    every other function in this module must keep working in an environment that does not have
    it. Failing at call time with an install hint beats failing at import time for a user who
    only wanted a p-value.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
        raise RuntimeError(
            "plot_curve needs matplotlib, which ships in the 'eval' extra: "
            "pip install 'recall-rag[eval]'"
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = sorted({str(p["arm"]) for p in points})
    fig, (ax_acc, ax_abs) = plt.subplots(2, 1, figsize=(7.5, 8), sharex=True)

    for arm in arms:
        n_answerable = max((int(p["n_answerable"]) for p in points if p["arm"] == arm), default=0)
        xs, ys = _series(points, arm, "accuracy")
        ax_acc.plot(xs, ys, marker="o", label=f"{arm} (n={n_answerable} answerable)")
    ax_acc.set_ylabel("answerable accuracy (fraction correct)")
    ax_acc.set_ylim(0, 1)
    ax_acc.set_title("Accuracy vs retrieved-context budget")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    for arm in arms:
        n_adversarial = max((int(p["n_adversarial"]) for p in points if p["arm"] == arm), default=0)
        x_abstain, y_abstain = _series(points, arm, "adversarial_abstention")
        line, = ax_abs.plot(
            x_abstain, y_abstain, marker="o",
            label=f"{arm}: adversarial abstention (n={n_adversarial})",
        )
        x_false, y_false = _series(points, arm, "false_abstain")
        ax_abs.plot(
            x_false, y_false, marker="s", linestyle="--", color=line.get_color(),
            label=f"{arm}: false abstention on answerable",
        )
    ax_abs.set_ylabel("abstention rate (fraction of questions refused)")
    ax_abs.set_ylim(0, 1)
    ax_abs.set_xlabel("mean retrieved context handed to the generator (approx. tokens/question)")
    ax_abs.set_title("Refusal behaviour: the gap between the two lines is Youden's J")
    ax_abs.legend(fontsize=8)
    ax_abs.grid(alpha=0.3)

    fig.tight_layout()
    path = Path(out_path)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# -- CLI ---------------------------------------------------------------------------------------

def _fmt(value: float | None, places: int = 4) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def _format_test(title: str, result: Mapping[str, Any]) -> list[str]:
    return [
        title,
        f"  n_paired={result['n_paired']}  both_good={result['both_correct']}  "
        f"both_bad={result['both_wrong']}  A_only={result['a_only']}  B_only={result['b_only']}",
        f"  test={result['test']}  statistic={result['statistic']:.4f}  "
        f"p_value={result['p_value']:.6f}",
    ]


def _compare_report(a_path: Path, b_path: Path) -> list[str]:
    a_doc: dict[str, Any] = _load_published(a_path)
    b_doc: dict[str, Any] = _load_published(b_path)
    a_arm, b_arm = str(a_doc.get("arm", "A")), str(b_doc.get("arm", "B"))

    lines = [
        f"A = {a_arm}  k={_k_of(a_doc)}  {a_path}",
        f"B = {b_arm}  k={_k_of(b_doc)}  {b_path}",
        "",
    ]
    lines += _format_test(
        "paired accuracy (McNemar, answerable questions only)",
        paired_mcnemar(a_doc["outcomes"], b_doc["outcomes"]),
    )
    lines.append("")
    lines += _format_test(
        "paired abstention (McNemar, adversarial questions only)",
        paired_abstention_mcnemar(a_doc["outcomes"], b_doc["outcomes"]),
    )
    lines.append("")
    lines.append("discrimination (Youden's J = adversarial abstention - false abstention)")
    for arm, doc in ((a_arm, a_doc), (b_arm, b_doc)):
        aggregate: dict[str, Any] = doc.get("aggregate") or {}
        lines.append(
            f"  {arm}: J={_fmt(discrimination(aggregate))}  "
            f"abstention={_fmt(_rate_of(aggregate, 'adversarial_abstention'))}  "
            f"false_abstention={_fmt(_rate_of(aggregate, 'answerable_false_abstain'))}"
        )
    return lines


def _curve_report(points: Sequence[Mapping[str, Any]]) -> list[str]:
    header = (
        f"{'arm':<14}{'k':>4}{'ctx_tok':>10}{'acc':>8}{'adv_abst':>10}"
        f"{'false_abst':>12}{'J':>8}{'n_ans':>7}{'n_adv':>7}"
    )
    lines = [header, "-" * len(header)]
    for point in points:
        k = point["k"]
        lines.append(
            f"{str(point['arm']):<14}{('-' if k is None else str(k)):>4}"
            f"{_fmt(point['ctx_tokens_mean'], 1):>10}{_fmt(point['accuracy']):>8}"
            f"{_fmt(point['adversarial_abstention']):>10}{_fmt(point['false_abstain']):>12}"
            f"{_fmt(point['discrimination']):>8}{point['n_answerable']:>7}"
            f"{point['n_adversarial']:>7}"
        )
    return lines



def _load_published(path: Path) -> dict[str, Any]:
    """Load a results artifact, refusing one `benchmarks.run` marked as never published.

    A refused artifact is quarantined outside the `results/*.json` glob AND marked in band, and
    this is the half that makes the mark mean something. Without it the mark is a comment: a
    quarantined file handed here directly, or reached by a `results/**/*.json` glob, is byte
    identical to a real measurement and gets tabulated as one.
    """

    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("unpublished"):
        raise SystemExit(
            f"{path} was REFUSED publication by benchmarks.run and is not a measurement: "
            f"{doc.get('unpublished_reason', 'no reason recorded')}"
        )
    return doc


def _expand(patterns: Sequence[str]) -> list[Path]:
    matched: list[Path] = []
    for pattern in patterns:
        hits = sorted(glob(pattern))
        if not hits:
            raise SystemExit(f"no results file matched {pattern!r}")
        matched += [Path(h) for h in hits]
    return matched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.analyze",
        description="Paired significance tests and the accuracy-vs-context curve, offline.",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("A.json", "B.json"),
        help="two results artifacts over the SAME question set, to test pairwise",
    )
    parser.add_argument(
        "--curve", nargs="+", metavar="GLOB",
        help="results artifacts (globs allowed) to tabulate as a curve",
    )
    parser.add_argument("--plot", type=Path, help="write the curve figure here (needs --curve)")
    args = parser.parse_args(argv)

    if not args.compare and not args.curve:
        parser.error("nothing to do: pass --compare and/or --curve")
    if args.plot and not args.curve:
        parser.error("--plot needs --curve")

    out: list[str] = []
    if args.compare:
        out += _compare_report(Path(args.compare[0]), Path(args.compare[1]))
    if args.curve:
        points = curve_points(_expand(args.curve))
        if out:
            out.append("")
        out += _curve_report(points)
        if args.plot:
            out.append(f"\nfigure written to {plot_curve(points, args.plot)}")
    print("\n".join(out))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
