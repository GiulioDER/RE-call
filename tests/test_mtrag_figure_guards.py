"""Guards for docs/make_mtrag_abstention.py, the generator behind docs/mtrag_abstention.png.

That figure is used in outreach, so the failure that matters is not a crash. It is a rebuild
that quietly produces a *different* image: a badge at the wrong size, a callout restating a
percentage its own plotted dot no longer has, or a mark drawn outside the plot. Each test here
drives one such path and requires the generator to either refuse or self-correct.

No network and no database. The badge fetch is stubbed in every test, so CI never depends on
Glama being reachable, and a change to the live rating cannot turn this red.
"""

from __future__ import annotations

import http.client
import importlib.util
import io
import pathlib
import shutil
import urllib.error
import urllib.request

import pytest

DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"
GENERATOR = DOCS / "make_mtrag_abstention.py"
COMMITTED_BADGE = (DOCS / "glama_score_badge.svg").read_text(encoding="utf-8")


def load_generator(directory: pathlib.Path):
    """Import the generator with HERE pointed at `directory`, so tests never touch docs/."""
    copied = directory / "make_mtrag_abstention.py"
    shutil.copy(GENERATOR, copied)
    spec = importlib.util.spec_from_file_location(f"gen_{directory.name}", copied)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def generator(tmp_path):
    """A generator instance with its own writable copy of the committed badge cache."""
    (tmp_path / "glama_score_badge.svg").write_text(
        COMMITTED_BADGE, encoding="utf-8", newline="\n"
    )
    return load_generator(tmp_path), tmp_path


class FakeResponse(io.BytesIO):
    """Minimal stand-in for the urlopen context manager."""

    def __init__(self, body: bytes, content_type: str = "image/svg+xml", raise_on_read=None):
        super().__init__(body)
        self._content_type = content_type
        self._raise = raise_on_read
        self.headers = self

    def get_content_type(self) -> str:
        return self._content_type

    def read(self, *args):
        if self._raise is not None:
            raise self._raise
        return super().read(*args)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def serve(monkeypatch):
    """Serve a chosen body, or raise a chosen error, from urlopen."""

    def _serve(body=None, error=None, content_type="image/svg+xml", raise_on_read=None):
        def fake_urlopen(*args, **kwargs):
            if error is not None:
                raise error
            return FakeResponse(body.encode(), content_type, raise_on_read)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    return _serve


# --- the anti-drift guard, the reason this file exists ------------------------------


def test_regenerating_reproduces_the_committed_html(tmp_path, monkeypatch):
    """A rebuild must reproduce docs/mtrag_abstention.html byte for byte.

    Offline, so this asserts the generator is deterministic rather than that Glama is up.
    """
    (tmp_path / "glama_score_badge.svg").write_text(
        COMMITTED_BADGE, encoding="utf-8", newline="\n"
    )
    module = load_generator(tmp_path)
    monkeypatch.setattr(
        module, "_fetch_badge", lambda: (_ for _ in ()).throw(urllib.error.URLError("offline"))
    )
    committed = (DOCS / "mtrag_abstention.html").read_text(encoding="utf-8")
    assert module.build() == committed


# --- annotations must be derived, never restated ------------------------------------


def test_annotations_are_derived_from_points(generator):
    module, _ = generator
    assert module.label_for("llama-3.1-8b") == "32.7%"
    assert module.label_for("llama-3.1-70b") == "29.1%"
    assert module.label_for("llama-3.1-405b") == "5.5%"
    assert module.label_for("RE-call") == "29.1%"
    # RE-call ties llama-3.1-70b at 16 of 55, which docs/MTRAG_BENCHMARK.md reads as second.
    assert module.rank_of("RE-call") == "second of ten"
    assert module.rank_of("llama-3.1-8b") == "first of ten"


def test_editing_points_moves_the_annotation(generator):
    """The callout must follow POINTS, or the figure can contradict its own dots."""
    module, _ = generator
    module.POINTS = [
        (name, x, 20 if name == "llama-3.1-8b" else hits, *rest)
        for name, x, hits, *rest in module.POINTS
    ]
    assert module.label_for("llama-3.1-8b") == "36.4%"


def test_duplicate_system_name_is_refused(generator):
    module, _ = generator
    module.POINTS = list(module.POINTS) + [("gpt-4o", 0.5591, 2, "start", 17, 5, "field")]
    with pytest.raises(SystemExit, match="duplicate"):
        module.rank_of("RE-call")


def test_unknown_system_is_refused(generator):
    module, _ = generator
    with pytest.raises(SystemExit, match="not in POINTS"):
        module.label_for("gpt-5")


# --- marks must never be drawn outside the plot -------------------------------------


@pytest.mark.parametrize(
    ("axis", "value"),
    [("y", 36.36), ("y", -0.1), ("x", 0.60), ("x", 0.40)],
)
def test_out_of_domain_coordinates_are_refused(generator, axis, value):
    """Unclamped maps plus overflow:hidden means an out-of-range point vanishes silently."""
    module, _ = generator
    mapper = module.py if axis == "y" else module.px
    with pytest.raises(SystemExit, match="domain"):
        mapper(value)


# --- the badge: resize correctly, or fall back without destroying the cache ----------


def test_committed_badge_keeps_its_size(generator, serve):
    """110x20 must scale to exactly 176x32, or the committed PNG changes underneath us."""
    module, _ = generator
    serve(body=COMMITTED_BADGE)
    assert 'width="176" height="32"' in module.glama_badge()


@pytest.mark.parametrize(
    ("case", "body"),
    [
        ("wider badge", COMMITTED_BADGE.replace('width="110" height="20"', 'width="124" height="20"')),
        ("single quotes", "<svg xmlns='x' width='110' height='20'><g/></svg>"),
        ("px units", '<svg width="110px" height="20px"><g/></svg>'),
        ("doctype first", '<?xml version="1.0"?>\n<!DOCTYPE svg><svg width="110" height="20"><g/></svg>'),
        ("reordered attrs", COMMITTED_BADGE.replace('width="110" height="20"', 'height="20" width="110"')),
        ("gt inside a value", '<svg aria-label="score > 8" width="110" height="20"><g/></svg>'),
    ],
    # Name the cases: the bodies are whole SVG documents and would otherwise be the test ids.
    ids=lambda value: value if isinstance(value, str) and len(value) < 30 else "",
)
def test_badge_variants_resize_on_aspect(generator, serve, case, body):
    """A served badge must be scaled from its own size, never matched against a fixed pair."""
    module, _ = generator
    serve(body=body)
    result = module.glama_badge()
    expected = 'width="198"' if case == "wider badge" else 'width="176"'
    assert expected in result, case
    assert 'height="32"' in result, case


def test_stroke_width_is_not_mistaken_for_the_badge_width(generator, serve):
    """`\\b` fires after a hyphen, so an unanchored pattern reads `stroke-width` instead."""
    module, _ = generator
    serve(body='<svg xmlns="x" stroke-width="2" width="110" height="20"><g/></svg>')
    result = module.glama_badge()
    assert 'width="176"' in result
    assert 'stroke-width="2"' in result


@pytest.mark.parametrize(
    ("case", "kwargs"),
    [
        ("cloudflare interstitial", {"body": "<!DOCTYPE html><html>Just a moment</html>",
                                     "content_type": "text/html"}),
        ("no size attributes", {"body": '<svg viewBox="0 0 110 20"><g/></svg>'}),
        ("truncated body", {"body": COMMITTED_BADGE[:400]}),
        ("connection dropped", {"body": "", "raise_on_read": http.client.IncompleteRead(b"1234")}),
        ("network down", {"error": urllib.error.URLError("offline")}),
    ],
)
def test_bad_response_falls_back_without_poisoning_the_cache(generator, serve, case, kwargs):
    """The committed cache is the offline fallback, so a bad body must never overwrite it."""
    module, tmp_path = generator
    serve(**kwargs)
    result = module.glama_badge()
    assert 'width="176" height="32"' in result, case
    survived = (tmp_path / "glama_score_badge.svg").read_text(encoding="utf-8")
    assert survived == COMMITTED_BADGE, f"{case} overwrote the fallback"


def test_missing_cache_and_failed_fetch_names_the_cause(tmp_path, serve):
    """Without a cache there is nothing to render, so say that rather than raising IOError."""
    module = load_generator(tmp_path)
    serve(error=urllib.error.URLError("offline"))
    with pytest.raises(SystemExit, match="no cached badge"):
        module.glama_badge()


def test_corrupt_cache_offers_the_remedy(generator, serve):
    """The bad bytes are local, so the message must point at the file, not the URL."""
    module, tmp_path = generator
    (tmp_path / "glama_score_badge.svg").write_text(
        COMMITTED_BADGE[:400], encoding="utf-8", newline="\n"
    )
    serve(error=urllib.error.URLError("offline"))
    with pytest.raises(SystemExit, match="git checkout"):
        module.glama_badge()


def test_cache_write_failure_still_renders(generator, serve, monkeypatch):
    """A full disk must not discard a good badge that is already in hand."""
    module, _ = generator
    serve(body=COMMITTED_BADGE)
    real_write = pathlib.Path.write_text

    def exploding_write(self, *args, **kwargs):
        if self.name.endswith(".tmp"):
            raise OSError(28, "No space left on device")
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "write_text", exploding_write)
    assert 'width="176" height="32"' in module.glama_badge()


# --- importing must stay inert ------------------------------------------------------


def test_import_does_not_touch_the_network_or_the_cache(tmp_path, monkeypatch):
    """Everything effectful belongs behind main(), as in docs/make_banner_graph.py."""
    (tmp_path / "glama_score_badge.svg").write_text(
        COMMITTED_BADGE, encoding="utf-8", newline="\n"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("import reached the network")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    module = load_generator(tmp_path)
    assert not hasattr(module, "HTML")
    assert (tmp_path / "glama_score_badge.svg").read_text(encoding="utf-8") == COMMITTED_BADGE


# --- markup integrity ---------------------------------------------------------------


def test_labels_are_escaped(generator):
    """A model name carrying a metacharacter must not terminate its own <text> element."""
    module, _ = generator
    module.POINTS = [
        ("a<b&c" if name == "gpt-4o" else name, x, hits, *rest)
        for name, x, hits, *rest in module.POINTS
    ]
    body = module.chart()
    assert "a&lt;b&amp;c" in body
    assert "<text" in body and "a<b&c</text>" not in body
