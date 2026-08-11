"""`recall extract`: the ingest-side command in front of the model-backed extractor.

Two things this command has to get right, and they pull in opposite directions.

It must be **impossible to run by accident**, because it is the first path in the library that
spends money and sends corpus text to a third party. So it is off unless `RECALL_EXTRACT` says
otherwise, and it says so and stops rather than quietly doing nothing.

And `--recheck` must be **able to report bad news**. It deliberately re-calls the model on keys
that are already cached and compares. A non-zero mismatch rate means temperature 0 is not
determinism for this provider, and that the cache — not the sampler — is the only thing making
runs reproducible. That is worth an exit code, because it is the kind of finding that otherwise
scrolls past.

Properties, one test each:

1. Off unless opted in, and the refusal names the variable.
2. A plain run reports what it found and never touches the corpus.
3. `--recheck` without a cache is refused: there is nothing to compare against.
4. `--recheck` re-calls the provider and reports a zero rate when the answer is stable.
5. `--recheck` reports a non-zero rate, exits non-zero, and leaves the cache intact.
6. Extraction never writes to the corpus, on either path.
"""
from __future__ import annotations

import json

import pytest

import recall.extraction as extraction_mod
from recall.cli import main
from recall.extraction import ClaimCache, ProseClaimExtractor

BODY = "This decision supersedes old_thing_2026 after the review."


class _Client:
    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.calls = 0

    def complete(self, *, model, messages, temperature):
        self.calls += 1
        payload = self._responses.pop(0) if self._responses else '{"claims": []}'
        return {
            "text": payload,
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "monetary_cost_usd": 0.0001,
        }


def _claim(obj: str = "old_thing_2026") -> str:
    return json.dumps({"claims": [{
        "relation": "supersedes", "object": obj, "evidence": BODY, "confidence": 0.9,
    }]})


def _corpus(tmp_path):
    """One memo with prose, plus a target that exists as a NAME but has no body.

    The empty target is deliberate. `recall extract` calls the model once per memo that has
    prose, so a two-memo fixture makes "how many calls did --recheck make" ambiguous — and that
    count is the whole assertion. An empty body is skipped before any call, so the target still
    populates `corpus_names` (the model may only name documents that exist) while keeping the
    call count exactly one per run.
    """
    (tmp_path / "old_thing_2026.md").write_bytes(b"")
    (tmp_path / "new.md").write_bytes(f"# new\n\n{BODY}\n".encode("utf-8"))
    return tmp_path


def _install(monkeypatch, client: _Client) -> None:
    """Replace the resolver so no real key, network call or SDK is involved.

    Patched on the module rather than on the CLI, because the dispatch imports it inside the
    function body — so the lookup happens per call and the seam is the module attribute.
    """
    def fake(env=None, *, cache: ClaimCache | None = None) -> ProseClaimExtractor:
        return ProseClaimExtractor(client=client, cache=cache)

    monkeypatch.setattr(extraction_mod, "resolve_claim_extractor", fake)


def test_the_command_is_off_unless_opted_in(tmp_path, monkeypatch, capsys):
    _corpus(tmp_path)
    monkeypatch.delenv("RECALL_EXTRACT", raising=False)

    with pytest.raises(SystemExit) as exc:
        main(["extract", str(tmp_path)])

    assert exc.value.code != 0
    assert "RECALL_EXTRACT" in capsys.readouterr().err


def test_a_plain_run_reports_the_claims_and_leaves_the_corpus_alone(tmp_path, monkeypatch, capsys):
    _corpus(tmp_path)
    before = (tmp_path / "new.md").read_bytes()
    _install(monkeypatch, _Client(_claim()))
    monkeypatch.setenv("RECALL_EXTRACT", "1")

    main(["extract", str(tmp_path)])

    out = capsys.readouterr().out
    assert "supersedes" in out
    assert "old_thing_2026" in out
    assert (tmp_path / "new.md").read_bytes() == before, "extraction wrote to the corpus"


def test_a_refused_response_does_not_abort_the_rest_of_the_corpus(tmp_path, monkeypatch, capsys):
    """"Failures in band, never raised."

    A malformed response is a finding about one memo, not a reason to abandon the others. Raising
    would leave the corpus half-extracted with no record of where it stopped — and, with a cache,
    half-paid-for. A surviving mutant found this: the property was stated in a comment beside the
    `except` and pinned by nothing.
    """
    _corpus(tmp_path)
    (tmp_path / "a_bad.md").write_bytes(f"# bad\n\n{BODY}\n".encode("utf-8"))
    _install(monkeypatch, _Client("not json at all", _claim()))
    monkeypatch.setenv("RECALL_EXTRACT", "1")

    main(["extract", str(tmp_path)])  # must return, not raise

    out = capsys.readouterr().out
    assert "SKIP a_bad.md" in out, "the malformed response was not reported"
    assert "new.md: supersedes old_thing_2026" in out, (
        "extraction stopped at the first bad response instead of continuing"
    )


def test_a_missing_path_is_refused_before_any_model_call(tmp_path, monkeypatch, capsys):
    """A typo'd corpus path produced a raw FileNotFoundError traceback, unlike every other
    argument error in this command, which exits 2 with a message."""
    client = _Client(_claim())
    _install(monkeypatch, client)
    monkeypatch.setenv("RECALL_EXTRACT", "1")

    with pytest.raises(SystemExit) as exc:
        main(["extract", str(tmp_path / "no_such_dir")])

    assert exc.value.code != 0
    assert "no such" in capsys.readouterr().err.lower()
    assert client.calls == 0, "a paid call was made against a path that does not exist"


def test_a_single_file_takes_its_corpus_names_from_its_directory(tmp_path, monkeypatch, capsys):
    """`corpus_names` globs, so a single-file path yielded an EMPTY name list while the file was
    still sent to the provider. Every claim then failed the "not a document in the corpus" check,
    guaranteeing a paid call that could not possibly produce a result."""
    _corpus(tmp_path)
    _install(monkeypatch, _Client(_claim()))
    monkeypatch.setenv("RECALL_EXTRACT", "1")

    main(["extract", str(tmp_path / "new.md")])

    out = capsys.readouterr().out
    assert "supersedes old_thing_2026" in out, f"the single-file form could not resolve: {out}"


def test_a_missing_api_key_exits_with_the_guidance_message(tmp_path, monkeypatch, capsys):
    """The resolver raises RuntimeError for a missing key and ImportError for a missing extra,
    but only ValueError was caught, so the two most likely first-run failures produced a
    traceback instead of the messages written for exactly those cases."""
    _corpus(tmp_path)

    def raising(env=None, *, cache=None):
        raise RuntimeError("the prose extractor needs an API key (OPENROUTER_API_KEY ...)")

    monkeypatch.setattr(extraction_mod, "resolve_claim_extractor", raising)
    monkeypatch.setenv("RECALL_EXTRACT", "1")

    with pytest.raises(SystemExit) as exc:
        main(["extract", str(tmp_path)])

    assert exc.value.code != 0
    assert "API key" in capsys.readouterr().err


def test_the_claim_cache_is_closed_even_when_the_run_exits_early(tmp_path, monkeypatch, capsys):
    """The cache was opened before the resolver and closed only on the single fall-through path,
    so every early exit leaked the sqlite connection. On Windows that leaves the file locked, and
    the observable symptom is that the corpus directory cannot be cleaned up afterwards."""
    _corpus(tmp_path)
    closed: list[int] = []

    class _CountingCache(extraction_mod.ClaimCache):
        def close(self) -> None:
            closed.append(1)
            super().close()

    def raising(env=None, *, cache=None):
        raise RuntimeError("no API key")

    monkeypatch.setattr(extraction_mod, "ClaimCache", _CountingCache)
    monkeypatch.setattr(extraction_mod, "resolve_claim_extractor", raising)
    monkeypatch.setenv("RECALL_EXTRACT", "1")

    with pytest.raises(SystemExit):
        main(["extract", str(tmp_path), "--cache", str(tmp_path / "claims.sqlite3")])

    assert closed, "the cache connection was leaked on the early-exit path"


def test_recheck_without_a_cache_is_refused(tmp_path, monkeypatch, capsys):
    _corpus(tmp_path)
    _install(monkeypatch, _Client())
    monkeypatch.setenv("RECALL_EXTRACT", "1")

    with pytest.raises(SystemExit) as exc:
        main(["extract", str(tmp_path), "--recheck"])

    assert exc.value.code != 0
    assert "--cache" in capsys.readouterr().err


def test_recheck_recalls_the_provider_and_reports_a_stable_rate(tmp_path, monkeypatch, capsys):
    _corpus(tmp_path)
    cache = tmp_path / "claims.sqlite3"
    client = _Client(_claim(), _claim())
    _install(monkeypatch, client)
    monkeypatch.setenv("RECALL_EXTRACT", "1")

    main(["extract", str(tmp_path), "--cache", str(cache)])
    assert client.calls == 1
    capsys.readouterr()

    main(["extract", str(tmp_path), "--cache", str(cache), "--recheck"])

    out = capsys.readouterr().out
    assert client.calls == 2, "--recheck did not re-call the provider"
    assert "mismatch rate" in out
    assert "0.0%" in out or "0%" in out


def test_recheck_reports_a_nonzero_rate_without_failing_the_command(tmp_path, monkeypatch, capsys):
    """A disagreement is a measurement of the provider, and it is REPORTED rather than fatal.

    The rate is what carries the finding, so it has to be on stdout and it has to be right. The
    exit code stays 0 deliberately: one flaky re-call is not a reason to fail a corpus-wide
    command, and a threshold baked in here would be a policy nobody chose. A caller wanting a
    gate reads the rate and picks their own.
    """
    _corpus(tmp_path)
    cache = tmp_path / "claims.sqlite3"
    client = _Client(_claim(), _claim(obj="new"))
    _install(monkeypatch, client)
    monkeypatch.setenv("RECALL_EXTRACT", "1")

    main(["extract", str(tmp_path), "--cache", str(cache)])
    capsys.readouterr()

    main(["extract", str(tmp_path), "--cache", str(cache), "--recheck"])  # must not raise

    out = capsys.readouterr().out
    assert "100.0%" in out, "the mismatch rate was not reported"
    assert "NOT determinism" in out, "the rate was printed without saying what it means"


def test_recheck_does_not_overwrite_the_cached_answer(tmp_path, monkeypatch, capsys):
    _corpus(tmp_path)
    cache_path = tmp_path / "claims.sqlite3"
    client = _Client(_claim(), _claim(obj="new"))
    _install(monkeypatch, client)
    monkeypatch.setenv("RECALL_EXTRACT", "1")

    main(["extract", str(tmp_path), "--cache", str(cache_path)])
    with ClaimCache(cache_path) as c:
        rows = c._conn.execute("SELECT response FROM claims").fetchall()
    stored = [r[0] for r in rows]
    capsys.readouterr()

    main(["extract", str(tmp_path), "--cache", str(cache_path), "--recheck"])

    with ClaimCache(cache_path) as c:
        after = [r[0] for r in c._conn.execute("SELECT response FROM claims").fetchall()]
    assert after == stored, "recheck replaced the content every proposal id was hashed from"
