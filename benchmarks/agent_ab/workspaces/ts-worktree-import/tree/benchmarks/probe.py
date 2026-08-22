"""Report the corpus chunk count."""

import recall


def main() -> int:
    print(f"chunks={recall.CHUNKS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
