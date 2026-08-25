from pathlib import Path

from PIL import Image


ROOT = Path(__file__).parents[1]


def test_repository_banner_has_no_top_gold_rule() -> None:
    image = Image.open(ROOT / "docs" / "banner.png").convert("RGB")
    top_band = image.crop((48, 42, 1233, 49))

    assert not any(
        red > 150 and green > 100 and blue < 100
        for red, green, blue in top_band.getdata()
    )
