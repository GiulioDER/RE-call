"""Fixes from the CCA hunt over the storage core (run 20261031T190444Z).

Each test names the finding it covers. Red proof (`python -m cca_checks tautology verify` for the
P1, a recorded production mutation for the others) is in the docstring of each behaviour test.
Two tests are deliberately nonbehavioural controls, labelled as such: they pin a refactor's
contract, which the pre-fix code also met, so red proof against it does not apply.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import get_args

import pytest

from recall.security_policy import (
    AccessContext,
    SourceRule,
    SourceSecurityPolicy,
    load_source_policy,
)

ROOT = Path(__file__).resolve().parent.parent


def _load(tmp_path: Path, payload: object) -> SourceSecurityPolicy | None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_source_policy({"RECALL_SOURCE_POLICY_FILE": str(path)})


# --- ENV-001 (P1): a misspelled policy key must not fall to the permissive default ------------


@pytest.mark.parametrize(
    "rule",
    [
        # Meant restricted; the typo used to load as `internal` and admit default clearance.
        {"prefix": "hr", "clasification": "restricted", "principals": [], "purposes": ["retrieval"],
         "redactions": []},
        # Meant egress gated; the typo used to drop the egress requirement.
        {"prefix": "vault", "principals": [], "purposes": ["retrieval"], "redactions": [],
         "allow_egres": True},
    ],
    ids=["classification", "allow_egress"],
)
def test_policy_file_refuses_an_unknown_rule_key(tmp_path: Path, rule: dict[str, object]) -> None:
    """ENV-001: `load_source_policy` refuses a rule key it does not know.

    Red proof: on 6eee13d87 both cases loaded, and `decide` admitted the default context
    (`internal`, egress denied) to the source the operator had meant to protect; the
    `pytest.raises` fails with DID NOT RAISE. Production line: the `_RULE_KEYS` check in
    `recall.security_policy.load_source_policy`.
    """
    with pytest.raises(ValueError, match="unknown keys"):
        _load(tmp_path, {"default_deny": True, "rules": [rule]})


def test_policy_file_refuses_an_unknown_top_level_key(tmp_path: Path) -> None:
    """ENV-001: the same refusal at the top level (`_POLICY_KEYS`).

    Red proof: on 6eee13d87 a stray `defualt_deny` loaded silently; DID NOT RAISE.
    """
    with pytest.raises(ValueError, match="unknown keys: defualt_deny"):
        _load(tmp_path, {"defualt_deny": False, "rules": []})


def test_policy_file_with_only_known_keys_still_loads(tmp_path: Path) -> None:
    """ENV-001 control: a correctly spelled policy loads and decides as before."""
    policy = _load(
        tmp_path,
        {
            "default_deny": True,
            "rules": [
                {"prefix": "hr", "classification": "restricted", "principals": [],
                 "purposes": ["retrieval"], "redactions": [], "allow_egress": True},
            ],
        },
    )
    assert policy is not None
    assert not policy.decide("hr/salaries.md", AccessContext("mallory", "t")).allowed


# --- BUG-001: a recall_graph line nested past the recursion limit is a marker, not a crash ----


def test_deeply_nested_recall_graph_is_a_parse_error_marker() -> None:
    """BUG-001: `parse_frontmatter` keeps a too-deep `recall_graph` as `__parse_error__`.

    Red proof: on 6eee13d87 (the GRAPH_KEY `except` caught `json.JSONDecodeError` only)
    RecursionError escaped parse_frontmatter, as it would stop an index run, and the explicit
    `pytest.fail` below reports it as this test's failure.
    """
    from recall.frontmatter import GRAPH_KEY, parse_frontmatter

    text = f"---\n{GRAPH_KEY}: " + "[" * 200_000 + "\n---\nbody\n"
    try:
        meta, _body = parse_frontmatter(text)
    except RecursionError:
        pytest.fail("a deeply nested recall_graph line raised RecursionError out of parse_frontmatter")
    assert "__parse_error__" in meta[GRAPH_KEY]


# --- SEC-001: the `secret` redaction catches the forms credentials are pasted in -------------


@pytest.mark.parametrize(
    ("text", "leaked"),
    [
        ("export OPENAI_API_KEY=sk-abc123", "sk-abc123"),
        ("DB_PASSWORD=hunter2", "hunter2"),
        ("GITHUB_TOKEN=ghp_xyz", "ghp_xyz"),
        ('{"password": "s3cr3t"}', "s3cr3t"),
        ("Authorization: Bearer eyJhbGciOi", "eyJhbGciOi"),
        ('{"Authorization": "Bearer abc123"}', "abc123"),
        ("headers={'Authorization': 'Bearer abc123'}", "abc123"),
        ("curl -H Bearer eyJhbGciOiJIUzI1NiJ9.e30.sig", "eyJhbGciOiJIUzI1NiJ9"),
        ("api_key=plain-value", "plain-value"),
    ],
)
def test_secret_redaction_covers_prefixed_quoted_and_bearer_forms(text: str, leaked: str) -> None:
    """SEC-001: none of these values survives a rule with `redactions=["secret"]`.

    Red proof: on 6eee13d87 every case but the last came back unchanged (a bare `\\b` cannot sit
    after `_`, and a closing quote blocked `[:=]`), so `leaked in redacted` fails. Production
    line: `_REDACTIONS["secret"]` in `recall/security_policy.py`.
    """
    policy = SourceSecurityPolicy((SourceRule("notes", redactions=("secret",)),))
    redacted, _ = policy.redact("notes/a.md", text, AccessContext("alice", "t"))
    assert leaked not in redacted
    assert "[REDACTED:secret]" in redacted


@pytest.mark.parametrize(
    ("text", "leaked"),
    [
        ("password=it'sx", "sx"),
        ('token: "a\'b c"', "b c"),
        ("secret='x\"y'", "y"),
        ('token="abc"def', "def"),
        ("TOKEN=\"abc\"'def'", "def"),
        ('password="ab"cd"ef"', "ef"),
    ],
)
def test_secret_redaction_takes_a_value_containing_quotes_whole(text: str, leaked: str) -> None:
    """SEC-001 follow-up: a value containing a quote is redacted whole, as before this hunt.

    Red proof: on 3daf7cf (this PR's first commit) the value stopped at the first quote, so
    `password=it'sx` came back as `[REDACTED:secret]'sx` and `leaked not in redacted` failed.
    The pre-hunt pattern (`[^\\s,;]+`) redacted these whole. Found by the post-commit security
    review; `token="abc"def` by the regression review of the first follow-up, and the two
    multi-segment cases by the security review of 11b14582e, against which each was red too.
    Production line: the value group of `_REDACTIONS["secret"]`.
    """
    policy = SourceSecurityPolicy((SourceRule("notes", redactions=("secret",)),))
    redacted, _ = policy.redact("notes/a.md", text, AccessContext("alice", "t"))
    assert leaked not in redacted


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        ('{"content": "set TOKEN=abc", "ts": "2026-10-31", "note": "public"}', '"ts": "2026-10-31"'),
        ('{"text": "password=hunter2", "author": "bob", "tags": ["a"]}', '"author": "bob"'),
        ("print('token=abc', 'the rest is public')", "'the rest is public'"),
        ('["token=x", "y"]', '"y"]'),
        ("password=it's a lovely day, isn't it", "a lovely day, isn't it"),
    ],
)
def test_secret_redaction_stops_at_the_end_of_the_value(text: str, kept: str) -> None:
    """SEC-001 upper bound: the text after a secret inside a JSON or code string survives.

    Red proof: on the second follow-up as first drafted (quoted segments allowed after an
    unquoted value) the closing quote of the string paired with the next field's quote and the
    following fields were erased; the regression review caught it before commit. Production line:
    the value group of `_REDACTIONS["secret"]`.
    """
    policy = SourceSecurityPolicy((SourceRule("notes", redactions=("secret",)),))
    redacted, _ = policy.redact("notes/a.md", text, AccessContext("alice", "t"))
    assert kept in redacted
    assert "[REDACTED:secret]" in redacted


#: The `secret` pattern as it stood before this hunt, kept only to assert the new one is no
#: narrower.
_PRE_HUNT_SECRET = re.compile(r"(?i)\b(?:api[_ -]?key|token|secret|password)\s*[:=]\s*[^\s,;]+")


@pytest.mark.parametrize(
    "text",
    [
        "password=it'sx",
        'token="abc"def',
        "TOKEN=\"abc\"'def'",
        'password="ab"cd"ef"',
        "token: \"unterminated value here",
        "secret='x\"y' tail",
        "api_key=a}b]c",
        'token: "line1\nline2"',
        "password=\"a b\"'c,d'e;f",
        "token=''",
        "the password: please",
    ],
)
def test_secret_redaction_is_no_narrower_than_before_the_hunt(text: str) -> None:
    """SEC-001 invariant: every span the pre-hunt pattern matched lies inside a new match.

    Checked on spans, not on the output: a narrower match can consume the key, and the pre-hunt
    pattern cannot find a leftover value whose key is gone. A value glued to a later credential
    assignment now stops there, so that assignment is matched on its own
    (`test_secret_redaction_closes_the_remaining_gaps`).
    Red proof: on 11b14582e `TOKEN="abc"'def'` and `password="ab"cd"ef"` ended at the second
    quote, inside the pre-hunt span. Production line: the value group of `_REDACTIONS["secret"]`.
    """
    from recall.security_policy import _REDACTIONS

    new_spans = [m.span() for m in _REDACTIONS["secret"].finditer(text)]
    for old in _PRE_HUNT_SECRET.finditer(text):
        assert any(start <= old.start() and old.end() <= end for start, end in new_spans), (
            f"pre-hunt span {old.span()} {old.group()!r} is no longer redacted whole in {text!r}"
        )


@pytest.mark.parametrize(
    ("text", "leaked"),
    [
        # The credential word is not the last part of the name.
        ("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI", "wJalrXUtnFEMI"),
        ("CLIENT_SECRET_ID: abc999", "abc999"),
        ("export STRIPE_SECRET_KEY=sk_live_1", "sk_live_1"),
        # A value glued to a later credential assignment.
        ("password=abc&token=xyz", "xyz"),
        ("password':a}token =xq", "xq"),
        ("secret:_password=btokenx", "btokenx"),
        ("Bearer sk-proj-abc123def456ghi789", "abc123def456ghi789"),
        # Long credential names: fourteen prefix segments, and a fifty two character segment.
        ("A_B_C_D_E_F_G_H_I_J_K_L_M_DB_PASSWORD=xq9", "xq9"),
        ("SUPERCALIFRAGILISTICEXPIALIDOCIOUSLYLONGSERVICENAMEX_TOKEN=xq9", "xq9"),
    ],
)
def test_secret_redaction_closes_the_remaining_gaps(text: str, leaked: str) -> None:
    """SEC-001 residuals the hunt left open, closed after PR 1108.

    Red proof (run 2026-10-31 on the remote test host): on 8f67f5c70 (PR 1108 merged) the three
    suffixed names matched no credential name, and in `password':a}token =xq` the later key was
    swallowed into the first value with its own value left after it, so `leaked not in
    redacted` fails. `password=abc&token=xyz` is a control: the old run already took it whole.
    `secret:_password=btokenx` and `Bearer sk-proj-...` are controls too, red against this
    change's first draft (a value stop that disagreed with `\b` on `_`, and a narrower bare bearer
    rule), which the regression review caught before commit. The two long names are red on
    fbffb0726, whose starting match bounded the prefix (found by the commit-time security review).
    Production lines: `_CREDENTIAL_NAME` and `_VALUE_CHAR` in `recall/security_policy.py`.
    """
    policy = SourceSecurityPolicy((SourceRule("notes", redactions=("secret",)),))
    redacted, _ = policy.redact("notes/a.md", text, AccessContext("alice", "t"))
    assert leaked not in redacted


@pytest.mark.parametrize(
    "text",
    [
        "the secret garden",
        "tokenizer settings",
        "my_secretary: bob",
        "a password policy",
        # Settings whose names merely contain a credential word, and must stay readable.
        "token_count: 5",
        "password_min_length: 8",
        "max_tokens=500",
        "token_id: 12",
        "password_auth: enabled",
        "TOKEN_ACCESS: read",
        "client_secret_hash: sha256",
        "token_key_id = kid-1",
        # The first version of the fix redacted these; the regression review caught it.
        "The bearer of bad news arrived.",
        "a bearer bond is a thing",
        "Bearer tokens expire",
        "the bearer instruments-and-securities market",
        "bearer self-identification/verification",
    ],
)
def test_secret_redaction_leaves_prose_alone(text: str) -> None:
    """SEC-001 control: words that merely contain a key name are not redacted.

    The five settings with a credential word inside their name were redacted by the first
    draft of the suffix list (an open list) before the regression review; they pin that the
    list stays closed. Production line: `_CREDENTIAL_NAME`.
    """
    policy = SourceSecurityPolicy((SourceRule("notes", redactions=("secret",)),))
    redacted, _ = policy.redact("notes/a.md", text, AccessContext("alice", "t"))
    assert redacted == text


# --- DAT-002: a reorder of overlapping redactions changes the text, so it changes the digests -


def test_redaction_order_is_part_of_both_policy_digests() -> None:
    """DAT-002: `digest` and `content_digest` change when a rule's redaction order changes.

    The two orders give different stored text (asserted first, so the premise is checked), and the
    indexer and the generation build reuse chunks on these digests. Red proof: on 6eee13d87 both
    digests sorted the redactions, so the digest assertions fail with equal values. Production
    lines: the two `"redactions"` entries in `SourceSecurityPolicy.__init__`.
    """
    phone_first = SourceSecurityPolicy((SourceRule("n", redactions=("phone", "secret")),))
    secret_first = SourceSecurityPolicy((SourceRule("n", redactions=("secret", "phone")),))
    context = AccessContext("alice", "t")
    text = "token: 555 123 4567 x"
    assert phone_first.redact("n/a.md", text, context)[0] != secret_first.redact("n/a.md", text, context)[0]
    assert phone_first.content_digest != secret_first.content_digest
    assert phone_first.digest != secret_first.digest


# --- ENV-004: a blank HNSW setting means unset -------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "RECALL_HNSW_EF_SEARCH_FILTERED",
        "RECALL_HNSW_ITERATIVE_SCAN_FILTERED",
        "RECALL_HNSW_EF_SEARCH_MULTIPLIER",
    ],
)
@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_hnsw_setting_takes_the_default(
    monkeypatch: pytest.MonkeyPatch, name: str, blank: str
) -> None:
    """ENV-004: `NAME=` (the generated template's form) applies the default instead of raising.

    Red proof: on 6eee13d87 each case raised ValueError from `int('')` or the vocabulary check,
    the failure every filtered dense search hit. Production line: `_env_setting` in
    `recall/store.py`, called by `_hnsw_filtered_tuning` and `_ef_search_multiplier`.
    """
    from recall import store

    for other in (
        "RECALL_HNSW_EF_SEARCH_FILTERED",
        "RECALL_HNSW_ITERATIVE_SCAN_FILTERED",
        "RECALL_HNSW_EF_SEARCH_MULTIPLIER",
    ):
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(name, blank)
    assert store.PgVectorStore._hnsw_filtered_tuning(None, 10) == (  # type: ignore[arg-type]
        max(store.DEFAULT_HNSW_EF_SEARCH_FILTERED, 10 * store.DEFAULT_HNSW_EF_SEARCH_MULTIPLIER),
        store.DEFAULT_HNSW_ITERATIVE_SCAN_FILTERED,
    )


def test_padded_iterative_scan_value_is_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    """ENV-004: `Relaxed_Order ` is read as `relaxed_order`, as the package's mode keys are.

    Red proof: on 6eee13d87 it raised "is not one of"; the equality assertion is never reached.
    """
    from recall import store

    monkeypatch.setenv("RECALL_HNSW_ITERATIVE_SCAN_FILTERED", " Relaxed_Order ")
    assert store.PgVectorStore._hnsw_filtered_tuning(None)[1] == "relaxed_order"  # type: ignore[arg-type]


# --- CODE-002: one bound for both dense arms (nonbehavioural control) -------------------------


def test_both_dense_arms_share_one_ef_search_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """CODE-002, NONBEHAVIOURAL CONTROL: the refactor kept each arm's refusal and warning.

    Red proof does not apply: the pre-fix code met this contract on both copies, which is the
    point of the refactor. It pins that the shared helper still refuses k past the maximum,
    naming the arm, and still warns when only the margin is cut.
    """
    from recall import store

    monkeypatch.delenv("RECALL_HNSW_EF_SEARCH_MULTIPLIER", raising=False)
    with pytest.raises(ValueError, match="a filtered HNSW scan"):
        store.PgVectorStore._hnsw_filtered_tuning(None, 1001)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="an unfiltered HNSW scan"):
        store._bounded_ef_search(1001, scan="an unfiltered", stacklevel=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert store._bounded_ef_search(400, scan="an unfiltered", stacklevel=2) == 1000
    assert any("over-fetch margin is reduced" in str(w.message) for w in caught)


# --- CODE-003: the redaction marker shapes have one definition --------------------------------


def _migration_arrays() -> list[tuple[str, ...]]:
    sql = (ROOT / "recall" / "migrations" / "sql" / "0029_fact_value_redaction.sql").read_text(
        encoding="utf-8"
    )
    return [
        tuple(re.findall(r"'([a-z_]+)'", body))
        for body in re.findall(r"=\s*ARRAY\[([^\]]+)\]", sql)
    ]


def test_redaction_marker_constants_match_migration_0029_and_every_writer() -> None:
    """CODE-003: REDACTED_CARD_KEYS and REDACTED_FACT_KEYS are the shape 0029 admits and writes.

    Red proof (mutation, run 2026-10-31 on the remote test host): adding `"extra": None` to the dict
    `recall.fact_redaction.redacted_card` returns failed this test (1 failed), and restoring it
    passed. The constants and the migration agree at 6eee13d87, so this guards drift.
    """
    from recall.fact_redaction import redacted_card, redacted_fact
    from recall.types import REDACTED_CARD_KEYS, REDACTED_FACT_KEYS, RedactedEvidenceCard

    arrays = _migration_arrays()
    assert REDACTED_CARD_KEYS in arrays
    assert REDACTED_FACT_KEYS in arrays
    assert tuple(sorted(redacted_fact("f", "r"))) == REDACTED_FACT_KEYS
    assert tuple(sorted(redacted_card({}, "c", "r"))) == REDACTED_CARD_KEYS
    card = RedactedEvidenceCard(
        card_id="c", generation_id="g", tenant_id="t", source="s.md", redaction_id="r",
        **_card_defaults(),
    )
    assert tuple(sorted(card.to_payload())) == REDACTED_CARD_KEYS


def _card_defaults() -> dict[str, object]:
    """Whatever else `EvidenceCard` requires, filled with empty values (the marker drops them)."""
    import dataclasses

    from recall.types import RedactedEvidenceCard

    given = {"card_id", "generation_id", "tenant_id", "source", "redaction_id"}
    filled: dict[str, object] = {}
    for f in dataclasses.fields(RedactedEvidenceCard):
        if f.name in given or not f.init:
            continue
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            filled[f.name] = ""
    return filled


# --- CODE-004: one authority vocabulary -------------------------------------------------------


def test_authority_vocabulary_has_one_definition() -> None:
    """CODE-004: the three copies agree, and `unknown` is refused by name, not by position.

    Red proof (mutation, run 2026-10-31 on the remote test host): adding a tier `"extra_tier"` to the Authority
    Literal in `recall/frontmatter.py` only failed this test (1 failed) on the set equality with
    `recall.types.Authority`, and restoring it passed. Before the fix, the same drift (or a tier
    added after `"unknown"`) would have passed silently through the positional `[:-1]` slice.
    """
    from recall import dependency_invalidation, frontmatter, types

    assert set(get_args(frontmatter.Authority)) == set(get_args(types.Authority))
    assert set(dependency_invalidation.AUTHORITY_VALUES) == set(get_args(types.Authority))
    with pytest.raises(ValueError, match="authority must be one of"):
        frontmatter.authority_from_metadata({frontmatter.GRAPH_KEY: {"authority": "unknown"}})
    for value in ("policy", "user_confirmed_decision", "tool_observation", "model_inference"):
        assert frontmatter.authority_from_metadata({frontmatter.GRAPH_KEY: {"authority": value}}) == value


# --- CODE-005: the raw context identity is named once (nonbehavioural control) ---------------


def test_raw_context_identity_constants_keep_the_fingerprint_shape() -> None:
    """CODE-005, NONBEHAVIOURAL CONTROL: the constants are the defaults, and a raw identity still
    serializes without context keys, so no live pipeline fingerprint moved.

    Red proof does not apply: the literals they replace held the same values.
    """
    import dataclasses

    from recall import lineage

    fields = {f.name: f.default for f in dataclasses.fields(lineage.EmbedderIdentity)}
    assert fields["context_mode"] == lineage.DEFAULT_CONTEXT_MODE == "none"
    assert fields["context_version"] == lineage.DEFAULT_CONTEXT_VERSION == "raw-v1"


# --- PERF-001: the learned-sparse refusal asks existence, not a full count --------------------


def test_learned_sparse_query_probes_existence_not_a_full_count() -> None:
    """PERF-001: `_query_learned_sparse` refuses an unencoded corpus without `sparse_row_count`.

    Red proof: on 6eee13d87 the query called `sparse_row_count`, an O(encoded rows) count, so the
    stub below raised AssertionError instead of the expected LookupError. Production line: the
    probe at the top of `PgVectorStore._query_learned_sparse`.
    """
    from recall.store import PgVectorStore

    class _Probe(PgVectorStore):
        def __init__(self) -> None:  # no connection: the probe answers before any SQL
            self._table = "chunks"
            self._tenant = "t"

        def sparse_row_count(self, profile_id: str) -> int:
            raise AssertionError("a full count ran on the query path")

        def _has_sparse_rows(self, profile_id: str) -> bool:
            return False

    with pytest.raises(LookupError, match="not indexed for learned sparse"):
        _Probe()._query_learned_sparse({1: 0.5}, 5, "p")


# --- NUM-001: a thousands separator is not a decimal comma ------------------------------------


@pytest.mark.parametrize(
    ("text", "terms"),
    [
        ("growth of 1,500 units", ["1500"]),
        ("revenue of 1,234,567 in 2023", ["1234567", "2023"]),
        ("1,500.25 and -2,000", ["1500.25", "-2000"]),
        ("a rate of 1,5% and 3.25", ["1.5%", "3.25"]),
        ("1,2345 is not grouped", ["1.2345"]),
        ("0,500 and -0,750 are decimals", ["0.500", "-0.750"]),
    ],
)
def test_numeric_terms_read_thousands_grouping(text: str, terms: list[str]) -> None:
    """NUM-001: `1,500` is fifteen hundred, and `1,234,567` is one number, not two.

    Red proof: on 8f67f5c70 every comma became a decimal point, so `1,500` gave `1.500` (and
    boosted a cell holding one and a half) and `1,234,567` gave `['1.234', '567']`; the
    equality fails. Production line: `recall.store._normalize_numeric` and `_NUMERIC_TOKEN_RE`.
    """
    from recall.store import _numeric_query_terms

    assert _numeric_query_terms(text) == terms


@pytest.mark.parametrize(
    "text",
    ["growth of 1,500 units", "1,234,567 and 1,5", "1,500.25 -2,000 +3", "v1.2 3,50 4", "none"],
)
def test_the_index_query_and_lite_sides_normalise_alike(text: str) -> None:
    """NUM-001: the boost matches only when the table side and the query side agree.

    Red proof (mutation, run 2026-10-31 on the remote test host): reverting
    `recall.extraction._table_numeric_values` alone to `replace(",", ".")` failed 3 of 5 cases
    (`['1.500'] == ['1500']`), and restoring it passed. On 8f67f5c70 it passes, because all three
    copies were wrong alike; `test_numeric_terms_read_thousands_grouping` is the red one there.
    Production lines: the three copies of the rule.
    """
    from recall import store as pg
    from recall.extraction import _table_numeric_values
    from recall.lite import store as lite

    query = pg._numeric_query_terms(text)
    assert lite._numeric_terms(text) == query
    assert _table_numeric_values([f"| {text} |"]) == query
