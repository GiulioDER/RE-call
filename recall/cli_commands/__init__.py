"""Command families for the `recall` CLI.

Each module registers one family of subcommands on the root parser (`register(sub)`) and owns
the handlers those parsers dispatch to via `set_defaults(func=...)`. `recall.cli` builds the
root parser and keeps the DSN security gating; nothing here imports `recall.cli` back.
"""
