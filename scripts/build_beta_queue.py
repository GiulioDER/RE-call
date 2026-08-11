from __future__ import annotations

import argparse
from pathlib import Path

from recall.beta_recruiting import input_field_guide, load_records, rank_leads, write_ranked_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rank public discussion exports into a beta outreach queue without collecting "
            "direct contact fields."
        )
    )
    parser.add_argument("input", type=Path, help="CSV or JSONL export of public discussions")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("beta_queue.csv"),
        help="Output CSV path, default: beta_queue.csv",
    )
    parser.add_argument(
        "--term",
        action="append",
        default=[],
        help="Topic term to match. Repeat the flag for multiple terms.",
    )
    parser.add_argument(
        "--show-fields",
        action="store_true",
        help="Print the allowed input fields and exit.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Maximum number of ranked leads to write, default: 50",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="Number of ranked leads to print to stdout, default: 10",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.show_fields:
        print("\n".join(input_field_guide()))
        return 0
    if not args.term:
        raise SystemExit("pass at least one --term to define the beta user profile")

    records = load_records(args.input)
    leads = rank_leads(records, args.term)[: args.top]
    write_ranked_csv(args.out, leads)
    print(f"wrote {args.out} with {len(leads)} ranked public-discussion leads")
    if leads and args.preview > 0:
        print("top links:")
        for index, lead in enumerate(leads[: args.preview], start=1):
            print(
                f"{index}. [{lead.action}] {lead.record.platform} | "
                f"{lead.record.title} | {lead.record.url}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
