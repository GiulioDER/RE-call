"""Compatibility entry point for rebuilding the RE-call social preview."""

from make_brand_assets import ROOT, save, social_card


if __name__ == "__main__":
    save(social_card(), ROOT / "social_card.png")
    print("wrote docs/social_card.png")
