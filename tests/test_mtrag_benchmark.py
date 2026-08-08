import json

from benchmarks.mtrag import run
from benchmarks.mtrag.run import ndcg_at, query_text, recall_at, score_predictions


def test_query_modes() -> None:
    task = {
        "task_id": "q",
        "input": [
            {"speaker": "user", "text": "first"},
            {"speaker": "agent", "text": "reply"},
            {"speaker": "user", "text": "second"},
            {"speaker": "user", "text": "third"},
            {"speaker": "user", "text": "fourth"},
        ],
    }
    assert query_text(task, "last") == "fourth"
    assert query_text(task, "recent3") == "second\nthird\nfourth"


def test_binary_metrics() -> None:
    relevant = {"a", "b"}
    assert recall_at(["a", "x", "b"], relevant, 1) == 0.5
    assert recall_at(["a", "x", "b"], relevant, 3) == 1.0
    assert ndcg_at(["a", "b"], relevant, 2) == 1.0
    assert 0.0 < ndcg_at(["x", "a", "b"], relevant, 3) < 1.0


def test_score_predictions_overall_is_pooled_not_domain_macro() -> None:
    """`overall` weights domains by judged-query count, and this fixture proves which one it is.

    The domains are deliberately UNBALANCED: three judged queries in clapnq (all hits) against
    one in cloud (a miss). Pooled gives 3/4 = 0.75; an unweighted mean of the two non-empty
    domain figures gives (1.0 + 0.0)/2 = 0.5. A balanced fixture cannot tell those apart, which
    is how the previous version of this test passed under either definition while its name
    asserted one of them.
    """
    predictions = [
        {"task_id": "q1", "contexts": [{"document_id": "a"}]},
        {"task_id": "q2", "contexts": [{"document_id": "b"}]},
        {"task_id": "q3", "contexts": [{"document_id": "c"}]},
        {"task_id": "q4", "contexts": [{"document_id": "x"}]},
    ]
    qrels = {
        "clapnq": {"q1": {"a"}, "q2": {"b"}, "q3": {"c"}},
        "cloud": {"q4": {"d"}},
        "fiqa": {},
        "govt": {},
    }
    scores = score_predictions(predictions, qrels)

    assert scores["overall"]["count"] == 4
    assert scores["overall"]["nDCG@5"] == 0.75
    assert scores["overall"]["Recall@5"] == 0.75

    # The per-domain figures stay unweighted within a domain.
    assert scores["domains"]["clapnq"]["nDCG@5"] == 1.0
    assert scores["domains"]["cloud"]["nDCG@5"] == 0.0
    assert scores["domains"]["fiqa"]["nDCG@5"] is None

    # Pin the choice itself: the pooled figure must NOT equal the domain macro. Without this,
    # switching the aggregation to a macro average would still satisfy every assertion above
    # that happens to hold under both.
    populated = [
        scores["domains"][d]["nDCG@5"]
        for d in ("clapnq", "cloud", "fiqa", "govt")
        if scores["domains"][d]["nDCG@5"] is not None
    ]
    domain_macro = sum(populated) / len(populated)
    assert domain_macro == 0.5
    assert scores["overall"]["nDCG@5"] != domain_macro


def test_the_default_split_is_dev_not_the_sealed_test_set() -> None:
    """The held-out set must be chosen deliberately, never arrived at by default.

    MTRAG-UN is what the official leaderboard scored and what the archived 2026-08-04 Task A
    baseline already used. A new arm comparison that lands on it by default silently spends the
    held-out set, and no error is raised when that happens — which is why this is asserted rather
    than left to a docstring.
    """
    args = run.parse_args(["--mtrag-root", ".", "--output-dir", "."])

    assert args.split == "dev"


def test_every_learned_sparse_arm_can_actually_reach_depth_100() -> None:
    """Recall@100 is only a depth measurement if the fused pool can hold 100 candidates.

    Each leg contributes at most `candidate_k`, so an arm at candidate_k=20 tops out at a pool of
    40 and its Recall@100 would silently be a statement about the POOL. These arms are declared
    at candidate_k=100 precisely so the number means what it says; if anyone lowers it, this
    fails instead of the metric quietly changing meaning.
    """
    undersized = {arm.name: arm.pool_bound() for arm in run.SPARSE_ARMS if arm.pool_bound() < 100}

    assert undersized == {}


def test_the_sparse_arms_vary_only_the_sparse_backend() -> None:
    """The comparison is only clean if nothing else moves between control and primary.

    `hybrid_lexical` and `hybrid_splade` must differ in exactly one field. If a future edit also
    changed candidate_k or the reranker, the measured delta would no longer be attributable to
    the learned sparse leg, and the result would look identical.
    """
    control = next(a for a in run.SPARSE_ARMS if a.name == "hybrid_lexical")
    primary = next(a for a in run.SPARSE_ARMS if a.name == "hybrid_splade")
    differing = {
        field for field in ("query_mode", "candidate_k", "use_dense", "use_sparse", "rerank",
                            "sparse_backend")
        if getattr(control, field) != getattr(primary, field)
    }

    assert differing == {"sparse_backend"}


def test_speaker_prefix_is_stripped_from_dev_queries() -> None:
    """MTRAG-human ships every turn prefixed with '|user|: '. It is not part of the question.

    Leaving it in feeds the literal token into both the embedder and the sparse encoder on EVERY
    dev query, which depresses the whole run and makes the numbers incomparable with the
    established baseline. Nothing errors; the scores are just quietly worse.

    Implementation matches benchmarks/mtrag/probe/fix_retrieval_2x2.py on bench/mtrag-arm-r, so
    normalisation is IDENTICAL to the measured baseline rather than merely similar.
    """
    assert run.strip_speaker("|user|: Do the Arizona Cardinals play outside the US?") == (
        "Do the Arizona Cardinals play outside the US?"
    )


def test_speaker_stripping_keeps_colons_inside_the_question() -> None:
    """Only the leading speaker tag goes. A colon in the question body is content."""
    assert run.strip_speaker("|user|: What is X: a thing?") == "What is X: a thing?"


def test_speaker_stripping_handles_the_multi_turn_concatenation() -> None:
    """The `_questions` file concatenates turns, one per line, each with its own prefix."""
    raw = "|user|: where do the arizona cardinals play\n|user|: Do they play outside the US?"

    assert run.strip_speaker(raw) == (
        "where do the arizona cardinals play\nDo they play outside the US?"
    )


def test_dev_queries_reach_the_retriever_without_their_speaker_prefix() -> None:
    """The end the bug actually lives at: what `query_text` hands the retriever.

    Testing `strip_speaker` alone would pass while the loader never called it, which is exactly
    the shape of the original defect.
    """
    task = {"task_id": "q1", "_domain": "clapnq", "_text": "|user|: How many teams are in the NFL?"}

    assert run.query_text(task, "last") == "How many teams are in the NFL?"


def _fake_release(
    root, dev_queries: int, test_queries: int, query_modes: tuple[str, ...] = ("lastturn",)
) -> None:
    """A minimal MTRAG root carrying BOTH layouts, with deliberately different sizes.

    The sizes differ so a validation that reads the wrong split cannot accidentally agree with
    one that reads the right one. Equal fixtures are how a split bug hides.
    """
    import json as _json
    import zipfile

    for domain in run.DOMAINS:
        corpus = root / "corpora" / "passage_level" / f"{domain}.jsonl.zip"
        corpus.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(corpus, "w") as zf:
            zf.writestr(f"{domain}.jsonl", "")

        dev_dir = root / "mtrag-human" / "retrieval_tasks" / domain
        (dev_dir / "qrels").mkdir(parents=True, exist_ok=True)
        for mode_file in query_modes:
            (dev_dir / f"{domain}_{mode_file}.jsonl").write_text(
                "".join(
                    _json.dumps({"_id": f"{domain}conv{i}<::>1", "text": f"q {mode_file}"}) + "\n"
                    for i in range(dev_queries)
                ),
                encoding="utf-8",
            )
        (dev_dir / "qrels" / "dev.tsv").write_text(
            "query\tcorpus\tscore\n"
            + "".join(f"{domain}conv{i}<::>1\tc{i}\t1\n" for i in range(dev_queries)),
            encoding="utf-8",
        )

    sealed_qrels = root / "mtragun-human" / "retrieval_tasks" / "qrels"
    sealed_qrels.mkdir(parents=True, exist_ok=True)
    sealed_tasks = root / "mtragun-human" / "generation_tasks"
    sealed_tasks.mkdir(parents=True, exist_ok=True)
    rows = []
    for domain in run.DOMAINS:
        (sealed_qrels / f"{domain}.tsv").write_text(
            "query\tcorpus\tscore\n"
            + "".join(f"{domain}-test-{i}\tc{i}\t1\n" for i in range(test_queries)),
            encoding="utf-8",
        )
        rows += [
            _json.dumps({"task_id": f"{domain}-test-{i}", "Collection": domain})
            for i in range(test_queries)
        ]
    (sealed_tasks / "reference.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_validation_describes_the_split_it_was_asked_for(tmp_path) -> None:
    """The manifest's `release` block must describe the split the run actually scores.

    `main()` writes `{"split": args.split, "release": validate_release(...)}` into one manifest.
    If validation ignores the split, those two fields contradict each other inside a single
    object: the run says `dev` and the provenance describes the SEALED set, down to the sha256 of
    files the run never opened. The scores would be right and the record of what they are would
    be wrong, which is the harder error to catch later.

    The pre-existing guard asserts `args.split == "dev"`, the argparse DEFAULT. The flag was never
    the problem; nothing downstream read it.
    """
    _fake_release(tmp_path, dev_queries=7, test_queries=3)

    dev = run.validate_release(tmp_path, "dev")
    test = run.validate_release(tmp_path, "test")

    assert dev["scored_query_count"] == 7 * len(run.DOMAINS)
    assert test["scored_query_count"] == 3 * len(run.DOMAINS)
    assert dev["split"] == "dev"
    assert test["split"] == "test"
    # The hashes must come from different files, or the block is describing one split for both.
    assert dev["input_sha256"] != test["input_sha256"]


def test_dev_validation_hashes_the_query_mode_the_selected_arms_use(tmp_path) -> None:
    """The provenance must name the files the ARMS open, not a hard-coded default.

    `validate_release` takes a `query_mode`, and on the dev split that argument is what selects
    WHICH per-domain file gets hashed. `main()` used to omit it, so the hashes were always the
    `last` files no matter which arms ran. That is the same defect as the split bug one axis over:
    every declared arm happens to use `last` today, so it was unobservable, but `DEV_QUERY_FILES`
    also defines `full` and `rewrite`.

    A run whose arms disagree about query mode has no single correct answer for one hash block, so
    it is refused rather than silently labelled with one of them.
    """
    _fake_release(tmp_path, dev_queries=4, test_queries=2, query_modes=("lastturn", "questions"))

    by_last = run.validate_release(tmp_path, "dev", "last")
    by_full = run.validate_release(tmp_path, "dev", "full")

    assert by_last["query_mode_validated"] == "last"
    assert by_full["query_mode_validated"] == "full"
    assert by_last["input_sha256"] != by_full["input_sha256"], (
        "the two query modes read different files, so their hash blocks must differ; "
        "equal hashes mean the mode is not reaching the file selection"
    )


def test_a_dev_run_whose_arms_disagree_on_query_mode_is_refused() -> None:
    """One manifest carries one `input_sha256` block, so it can describe only one query mode."""
    import pytest as _pytest

    with _pytest.raises(ValueError, match="query mode"):
        run.dev_query_mode_for(["recall_default_last", "recall_default_recent3"], "dev")


def test_the_dev_query_mode_comes_from_the_selected_arms() -> None:
    assert run.dev_query_mode_for(["hybrid_splade", "hybrid_lexical"], "dev") == "last"


def test_dev_tasks_carry_the_fields_the_prediction_writer_reads(tmp_path) -> None:
    """`run.py --split dev` had never completed a run: the writer reads fields dev lacked.

    `run_arm` builds each prediction from `task["conversation_id"]`, `task["Collection"]` and
    `task["input"]`. `load_dev_tasks` normalised only `task_id`, `_domain` and `_text`, so the
    first arm died with KeyError: 'conversation_id' after validation had already passed. The dev
    baseline recorded in memory (nDCG@5 0.2849 / R@100 0.6865) came from a separate probe script,
    which is why nothing had noticed.

    The missing fields are DERIVABLE, not absent. MTRAG uses one id convention across both splits:
    the sealed release ships `conversation_id` "18ef26..." beside `task_id` "18ef26...<::>7", and
    a dev `_id` reads "dd6b6f...<::>2". The conversation id is the part before the separator.
    """
    domain = run.DOMAINS[0]
    dev_dir = tmp_path / "mtrag-human" / "retrieval_tasks" / domain
    dev_dir.mkdir(parents=True, exist_ok=True)
    for other in run.DOMAINS:
        d = tmp_path / "mtrag-human" / "retrieval_tasks" / other
        d.mkdir(parents=True, exist_ok=True)
        rows = ""
        if other == domain:
            rows = json.dumps(
                {"_id": "conv123<::>4", "text": "|user|: Do the Cardinals play outside the US?"}
            ) + "\n"
        (d / f"{other}_lastturn.jsonl").write_text(rows, encoding="utf-8")

    tasks = run.load_dev_tasks(tmp_path, "last")

    assert len(tasks) == 1
    task = tasks[0]
    assert task["task_id"] == "conv123<::>4"
    assert task["conversation_id"] == "conv123"
    assert task["Collection"] == domain
    assert task["input"] == [
        {"speaker": "user", "text": "Do the Cardinals play outside the US?"}
    ], "input must be the turn WITHOUT the speaker prefix the release ships"


def _write_dev(root, domain: str, mode_file: str, rows: list[dict]) -> None:
    import json as _json

    d = root / "mtrag-human" / "retrieval_tasks" / domain
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{domain}_{mode_file}.jsonl").write_text(
        "".join(_json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


def test_full_mode_records_one_input_entry_per_turn(tmp_path) -> None:
    """The `_questions` file concatenates the conversation, one turn per line.

    Collapsing it into a single `{"speaker": "user", "text": ...}` records one fabricated turn
    whose body contains several real ones, which is not what the submission format means by
    `input`. Dormant today, since every declared dev arm uses `last`, but `DEV_QUERY_FILES`
    exposes `full` and the CLI can select it.
    """
    for domain in run.DOMAINS:
        rows = []
        if domain == run.DOMAINS[0]:
            rows = [{"_id": "conv9<::>2", "text": "|user|: where do they play\n|user|: outside the US?"}]
        _write_dev(tmp_path, domain, "questions", rows)

    tasks = run.load_dev_tasks(tmp_path, "full")

    assert tasks[0]["input"] == [
        {"speaker": "user", "text": "where do they play"},
        {"speaker": "user", "text": "outside the US?"},
    ]


def test_an_id_without_the_separator_is_refused(tmp_path) -> None:
    """The conversation id is derived by splitting; a different convention must not pass silently.

    `.split(sep, 1)[0]` on an id lacking the separator returns the whole id, so conversation_id
    would equal task_id and look plausible.
    """
    import pytest as _pytest

    for domain in run.DOMAINS:
        rows = [{"_id": "no-separator-here", "text": "|user|: hi"}] if domain == run.DOMAINS[0] else []
        _write_dev(tmp_path, domain, "lastturn", rows)

    with _pytest.raises(RuntimeError, match="carries no"):
        run.load_dev_tasks(tmp_path, "last")


def test_the_recorded_input_is_what_the_retriever_was_given(tmp_path) -> None:
    """`_text` and `input` are computed separately from one source; pin that they agree.

    If they drift, the prediction records an `input` that is not what was searched, and no
    existing test compares them.
    """
    for domain in run.DOMAINS:
        rows = [{"_id": "c1<::>1", "text": "|user|: how many teams"}] if domain == run.DOMAINS[0] else []
        _write_dev(tmp_path, domain, "lastturn", rows)

    task = run.load_dev_tasks(tmp_path, "last")[0]

    assert run.query_text(task, "last") == task["input"][-1]["text"]


def test_the_default_arm_selection_is_runnable_on_the_default_split() -> None:
    """`--split dev` is the default so the sealed set must be typed out. It has to work.

    The default ARMS tuple mixes `last` and `recent3`, and `recent3` has no dev file, so the
    default selection on the default split used to refuse before writing anything.
    """
    names = run.select_arm_names(None, "dev")

    assert names, "the default dev selection must not be empty"
    assert run.dev_query_mode_for(names, "dev") == "last"
    assert "recall_default_recent3" not in names


def test_an_explicit_arm_request_is_still_honoured_into_the_refusal() -> None:
    """Narrowing applies to the DEFAULT only. Naming two incompatible arms must still raise."""
    import pytest as _pytest

    explicit = ["recall_default_last", "recall_default_recent3"]

    assert run.select_arm_names(explicit, "dev") == explicit
    with _pytest.raises(ValueError, match="query mode"):
        run.dev_query_mode_for(run.select_arm_names(explicit, "dev"), "dev")


def test_the_two_reranked_arms_vary_only_the_reranker() -> None:
    """Local against hosted, with nothing else moving.

    If they also differed in candidate_k or backend, the measured gap would not be attributable to
    the reranker, and the article's two rows would be comparing two different systems.
    """
    local = next(a for a in run.SPARSE_ARMS if a.name == "hybrid_splade_rerank")
    hosted = next(a for a in run.SPARSE_ARMS if a.name == "hybrid_splade_voyage")
    differing = {
        f for f in ("query_mode", "candidate_k", "use_dense", "use_sparse", "rerank",
                    "sparse_backend")
        if getattr(local, f) != getattr(hosted, f)
    }

    assert differing == set()
    assert (local.reranker, hosted.reranker) == ("minilm", "voyage")


def test_an_unknown_reranker_name_is_refused() -> None:
    """A typo must not silently fall back to the local model and be reported as the hosted one."""
    import pytest as _pytest

    bogus = run.Arm("x", "last", 100, rerank=True, reranker="nope")
    with _pytest.raises(ValueError, match="not built"):
        run.build_reranker(bogus)


def test_the_unreranked_primary_is_the_matched_control_for_both() -> None:
    """hybrid_splade must differ from each reranked arm ONLY in `rerank`."""
    base = next(a for a in run.SPARSE_ARMS if a.name == "hybrid_splade")
    for name in ("hybrid_splade_rerank", "hybrid_splade_voyage"):
        arm = next(a for a in run.SPARSE_ARMS if a.name == name)
        differing = {
            f for f in ("query_mode", "candidate_k", "use_dense", "use_sparse", "sparse_backend")
            if getattr(base, f) != getattr(arm, f)
        }
        assert differing == set(), name
        assert base.rerank is False and arm.rerank is True


def test_a_transient_failure_is_retried_not_fatal(monkeypatch) -> None:
    """A hosted reranker turns one 429 into a dead arm unless the search is retried."""
    monkeypatch.setattr(run.time, "sleep", lambda *_: None)

    class Flaky:
        def __init__(self):
            self.calls = 0

        def search(self, query, k):
            self.calls += 1
            if self.calls < 3:
                raise ConnectionError("transient")
            return "ok"

    arm = next(a for a in run.SPARSE_ARMS if a.name == "hybrid_splade_voyage")
    flaky = Flaky()
    result, elapsed_ms, retries = run.search_with_retry(flaky, "q", arm)

    assert result == "ok"
    assert flaky.calls == 3
    assert retries == 2
    # The elapsed time must cover ONLY the successful attempt. Timing from before the first would
    # include 2s + 4s of backoff, and one retried query would put 6000ms into a published p95.
    assert elapsed_ms < 1000.0


def test_retries_are_bounded_and_name_the_partial_file(monkeypatch) -> None:
    """Give up eventually, and say where the already-paid work was flushed."""
    import pytest as _pytest

    monkeypatch.setattr(run.time, "sleep", lambda *_: None)

    class Dead:
        def search(self, query, k):
            raise ConnectionError("always")

    arm = next(a for a in run.SPARSE_ARMS if a.name == "hybrid_splade_voyage")
    with _pytest.raises(RuntimeError, match="partial.jsonl"):
        run.search_with_retry(Dead(), "q", arm)


def test_build_reranker_refuses_an_arm_that_does_not_rerank() -> None:
    """Constructing a PAID client for an arm that never reranks would demand a key for nothing."""
    import pytest as _pytest

    arm = run.Arm("x", "last", 100, rerank=False, reranker="voyage")
    with _pytest.raises(ValueError, match="rerank=False"):
        run.build_reranker(arm)


def test_a_permanent_error_is_not_retried(monkeypatch) -> None:
    """A bad key is not a flaky network. Retrying it burns 14s and reads like one.

    The final message must carry the underlying error, so someone skimming output sees "bad
    credential" rather than having to open a chained traceback to find out.
    """
    import pytest as _pytest

    monkeypatch.setattr(run.time, "sleep", lambda *_: None)

    class AuthenticationError(Exception):
        pass

    class Unauthorized:
        def __init__(self):
            self.calls = 0

        def search(self, query, k):
            self.calls += 1
            raise AuthenticationError("invalid api key")

    arm = next(a for a in run.SPARSE_ARMS if a.name == "hybrid_splade_voyage")
    bad = Unauthorized()
    with _pytest.raises(RuntimeError, match="invalid api key"):
        run.search_with_retry(bad, "q", arm)

    assert bad.calls == 1, "a permanent error must not be retried"


def test_the_failure_message_says_the_partial_is_not_resumed(monkeypatch) -> None:
    """The partial file is forensic, not a checkpoint. Nothing reads it back.

    The first version of this fix claimed in its commit message that a transient failure no longer
    discards an arm's paid work. That was only true of the bytes: there is no resume, so a restart
    re-issues every call including the billed ones. The message must not imply otherwise.
    """
    import pytest as _pytest

    monkeypatch.setattr(run.time, "sleep", lambda *_: None)

    class Dead:
        def search(self, query, k):
            raise ConnectionError("always")

    arm = next(a for a in run.SPARSE_ARMS if a.name == "hybrid_splade_voyage")
    with _pytest.raises(RuntimeError, match="NOT resumed from"):
        run.search_with_retry(Dead(), "q", arm)


def test_two_runs_into_one_directory_do_not_overwrite_each_others_manifest() -> None:
    """The manifest is the only record of which commit produced an arm's numbers.

    This is not hypothetical. Five dev arms completed 2026-08-07/08 and a later arm reused their
    output directory; because every run wrote `preregistered_manifest.json`, the record of what
    produced those five was destroyed, and the per-arm metrics carried no revision to fall back
    on. Their numbers survived review only because `retriever.py` happened to be byte-identical
    across the range, which is luck standing in for evidence.
    """
    from benchmarks.mtrag.run import manifest_filename

    a = manifest_filename("235666324ea534ff77ac05264aab40e3bc353e68", "2026-08-08T10:04:37+00:00")
    b = manifest_filename("c6f99b5aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "2026-08-08T13:11:02+00:00")
    assert a != b
    assert "2356663" in a, "the revision must be legible in the name, not only inside the file"

    # Same commit, different run: re-running one commit is normal, so the timestamp must separate
    # them too. Keying on revision alone would still overwrite.
    c = manifest_filename("235666324ea534ff77ac05264aab40e3bc353e68", "2026-08-08T18:30:00+00:00")
    assert a != c

    # A missing revision must still yield a usable, unique name rather than raising.
    d = manifest_filename(None, "2026-08-08T10:04:37+00:00")
    assert d.startswith("manifest_norev_") and d.endswith(".json")
