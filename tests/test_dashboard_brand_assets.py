"""Regression coverage for the dashboard's generated canonical logo."""

from pathlib import Path

from PIL import Image


def test_sync_logos_replaces_stale_dashboard_webp(tmp_path, monkeypatch):
    """DEPLOY-001: sync_logos must refresh the dashboard from the canonical PNG.

    Red proof: this exact test against the pre-fix docs.make_brand_assets.sync_logos
    fails at the stale dashboard bytes assertion. The prior PNG outputs still exist.
    Node: tests/test_dashboard_brand_assets.py::test_sync_logos_replaces_stale_dashboard_webp
    """
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "docs"))
    import make_brand_assets

    docs = tmp_path / "docs"
    canonical = docs / "assets/re_call_logo.png"
    canonical.parent.mkdir(parents=True)
    source = Image.new("RGBA", (384, 192), (0, 0, 0, 0))
    source.paste((215, 165, 42, 255), (96, 32, 288, 160))
    source.paste((20, 90, 160, 128), (128, 64, 256, 128))
    source.save(canonical)

    outputs = [
        tmp_path / "site/assets/re_call_logo.png",
        tmp_path / "recall/desktop/assets/re_call_watermark.png",
        tmp_path / "plugin/.claude-plugin/icon.png",
    ]
    dashboard = tmp_path / "recall/dashboard/static/recall-logo.webp"
    for path in [*outputs, dashboard]:
        path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (8, 8), (200, 0, 0, 255)).save(dashboard, lossless=True)
    stale = dashboard.read_bytes()
    monkeypatch.setattr(make_brand_assets, "ROOT", docs)
    monkeypatch.setattr(make_brand_assets, "LOGO", canonical)

    make_brand_assets.sync_logos()

    for path in outputs:
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.convert("RGBA").tobytes() == source.tobytes()
    assert dashboard.read_bytes() != stale, "sync_logos left the dashboard WebP stale"
    expected = source.resize((192, 96), Image.Resampling.LANCZOS)
    with Image.open(dashboard) as image:
        assert image.format == "WEBP"
        assert image.size == (192, 96)
        assert image.mode == "RGBA"
        assert image.getchannel("A").getextrema() == (0, 255)
        assert image.tobytes() == expected.tobytes()
