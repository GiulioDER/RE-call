"""`recall setup`, `recall wizard`, `recall uninstall` and `recall quickstart`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from recall.context import context_policy_for_profile
from recall.embeddings import embedding_profile_id
from recall.index import Indexer
from recall.store import (
    DEFAULT_TABLE,
    DEFAULT_TENANT,
    PgVectorStore,
    redacted_dsn,
    require_secure_dsn,
)
from recall.trust_policy import TrustPolicy

from recall.cli_commands._shared import _entailment_judge, _make_embedder, _run_queries


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    # Deliberately does NOT say "wizard": `recall wizard` is a different command, and `recall
    # --help` listed both describing themselves as the install wizard with no way to tell which.
    sub.add_parser(
        "setup",
        help="configure recall interactively and write a local .env file",
        # The canonical install, and the one the README and the epilogue in `recall.cli` both send
        # people to, so it says what it will DO to the machine before it starts asking. It had no
        # description at all, which meant `recall setup --help` printed four lines and none of them
        # mentioned that it writes `.env`, touches the schema, or edits the Claude Code config.
        description=(
            "The guided install. Asks which embedder, reranker and entailment judge you need, "
            "prepares the database schema at that embedder's width, offers to fit an abstention "
            "threshold against your own corpus, and offers to register the MCP server and session "
            "hooks with Claude Code. Writes a local .env with the answers; every step that touches "
            "something outside this directory is asked for first. Run `recall doctor` afterwards "
            "to check the result. If you want this scripted rather than asked, `recall wizard` "
            "runs the same engine from a JSON config."
        ),
    ).set_defaults(
        _opens_db=True,  # the wizard connects when the operator accepts the calibrate prompt
        func=_cmd_setup,
    )

    # The headless installer path. Separate from `setup`, which is the interactive configuration
    # interview: this one takes a corpus from a directory to a calibrated, promoted generation and
    # is what CI drives against a throwaway container.
    p_wizard = sub.add_parser(
        "wizard",
        help="install recall: ask what is needed and build it, or run a saved JSON config",
        # ⚠️ `help=` shows in `recall --help`; `description=` is what `recall wizard --help` shows,
        # and without one this subcommand printed a bare usage line and its flags. So the command
        # most likely to be reached by somebody unsure whether they wanted `setup` instead was the
        # one that explained itself least. It says how it DIFFERS from `setup` rather than what it
        # does, because a reader at this prompt has already seen both names and is choosing.
        description=(
            "Install recall as a repeatable pipeline: provision the database stack, build and "
            "calibrate every corpus, promote a generation, and register the MCP servers. Same "
            "engine as `recall setup`; the difference is that the answers become a JSON config "
            "which is what actually runs, so the install can be replayed and driven headlessly by "
            "CI. Prefer `recall setup` if you are installing this once, by hand, for yourself."
        ),
    )
    p_wizard.set_defaults(_opens_db=True, func=_cmd_wizard)
    p_wizard.add_argument(
        "--headless",
        action="store_true",
        help="do not ask anything; run --config unattended. Required whenever --config is given, "
        "because that path builds, calibrates and promotes without prompting.",
    )
    p_wizard.add_argument(
        "--config",
        required=False,
        help="JSON config to run. Omit it to be asked the questions instead, in which case the "
        "answers are WRITTEN to a config file and that file is what runs, so the install is "
        "repeatable. Required keys: dsn, migration_dsn, embedder, corpus_version, "
        "docs_root, code_root, memory_root (roots must be absolute). Optional: project, and "
        "serving_role, which is required when dsn and migration_dsn authenticate as different "
        "roles, because no migration emits a GRANT.",
    )
    p_wizard.add_argument(
        "--state",
        default=None,
        help="resumable state file (default: the config path with a .state.json suffix). A corpus a "
        "previous run PROMOTED under the same configuration is reused rather than rebuilt, and the "
        "file is written after every corpus. A degraded corpus is always retried.",
    )
    p_wizard.add_argument(
        "--fresh",
        action="store_true",
        help="ignore any recorded state and rebuild every corpus.",
    )
    p_wizard.add_argument(
        "--no-state",
        action="store_true",
        help="do not read or write a state file at all.",
    )
    p_wizard.add_argument(
        "--gui",
        action="store_true",
        help="ask the questions in a window instead of the terminal. Same questions, same config "
        "file, same engine; needs the desktop extra. `recall-install` opens it directly.",
    )

    p_uninstall = sub.add_parser(
        "uninstall",
        help="remove an install's containers, stack files and MCP registrations. Never removes the "
        "folders it was indexing.",
        description=(
            "Undo what an install created: its Docker containers and volumes, its generated stack "
            "files, the MCP server registrations and the session hooks. It never touches the "
            "folders you pointed it at, and never the memos in them. Requires --data-root, "
            "because the thing being removed is one install rather than every install on the "
            "machine, and guessing which would be the wrong way to be convenient."
        ),
    )
    p_uninstall.set_defaults(func=_cmd_uninstall)
    p_uninstall.add_argument(
        "--data-root",
        required=True,
        help="the data folder chosen during installation; it is recorded in that install's "
        "wizard.json.",
    )
    p_uninstall.add_argument(
        "--purge-data",
        action="store_true",
        help="also remove the database volume holding the built indexes. Off by default: they are "
        "reproducible by re-indexing and expensive to rebuild.",
    )
    p_uninstall.add_argument(
        "--yes",
        action="store_true",
        help="skip the confirmation. Without it the plan is printed and you are asked.",
    )
    p_uninstall.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be removed and stop.",
    )


def register_quickstart(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    _quickstart_blurb = (
        "start a throwaway database, index the bundled demo corpus and answer three queries"
    )
    p_quickstart = sub.add_parser(
        "quickstart",
        help=_quickstart_blurb,
        description=(
            f"{_quickstart_blurb}. One command from a fresh `pip install` to a real answer, "
            "including the PostgreSQL it needs. Nothing is calibrated and nothing is registered "
            "with an agent: this exists to show what the retrieval layer does, and prints the "
            "next step for each. Remove everything it created with `recall quickstart --remove`."
        ),
    )
    # ⚠️ **No `_opens_db=True`, and that is the point.** The flag drives `main()`'s DSN resolution,
    # its secure-DSN guard and its .env refusal, all of which act on a DSN this command does not
    # have yet: it PROVISIONS the database it is about to use. Declaring the flag would refuse the
    # run over a configuration problem the quickstart exists to bypass, which is exactly the dead
    # end the `setup` carve-out below already had to be written for once.
    p_quickstart.set_defaults(func=_cmd_quickstart)
    # ⚠️ `--existing-dsn`, NOT `--dsn`. The parent parser already owns `--dsn` (aliased from
    # `--serving-dsn`, with a default), so a subparser option of the same name would be a second
    # `--dsn` whose meaning depended on which side of the word `quickstart` the reader typed it,
    # and only one of the two would be read. A distinct name cannot be given by accident.
    p_quickstart.add_argument(
        "--existing-dsn",
        dest="existing_dsn",
        default=None,
        help=(
            "use a PostgreSQL you already have instead of starting one. Needs the pgvector "
            "extension available and a role that may create tables. Skips Docker entirely."
        ),
    )
    p_quickstart.add_argument(
        "--remove",
        action="store_true",
        help="stop the quickstart database and destroy its volume, then exit",
    )
    # No `--embedder` here: the parent parser's flag already covers it and already honours
    # RECALL_EMBEDDER. A second one would shadow the first depending on argument order.


def _cmd_setup(args: argparse.Namespace) -> None:
    from recall.setup import run_setup_wizard

    # Pass the caller's table through: the wizard checks the chosen embedder's width against
    # it, and checking a different table than the one in use is worse than not checking.
    run_setup_wizard(
        dsn=args.dsn,
        migration_dsn=args.migration_dsn,
        tenant=args.tenant,
        table=args.table,
    )


def _cmd_quickstart(args: argparse.Namespace) -> None:
    _quickstart(args)


def _quickstart(args: argparse.Namespace) -> None:
    """`recall quickstart`: provision, index, answer, and say what to do next.

    Kept apart from the shared resolved-DSN/embedder/store flow deliberately: this command
    resolves its own DSN by creating the database, so folding it into that flow would mean
    making three shared preconditions conditional. It is also the one command whose whole value
    is the ORDER of what the reader sees, and that order is much easier to keep right when it is
    readable in one screen.

    Nothing here is calibrated. `_cli_trust` therefore prints its uncertified-threshold notice, and
    that notice is a feature of this command rather than noise to be suppressed: the first thing a
    reader should learn is that the threshold answering their queries is not a calibration, because
    the second thing they will want is the command that fits a real one.
    """
    from recall.quickstart import (
        DEMO_QUERIES,
        QUICKSTART_TABLE,
        QUICKSTART_TENANT,
        demo_corpus,
        docker_unavailable_reason,
        next_steps,
        provision,
        remove_stack,
    )
    from recall.schema import SchemaTooOld, apply_migrations

    if args.remove:
        target = remove_stack()
        print(
            "removed the quickstart database and its volume"
            if target
            else "nothing to remove: no quickstart stack has been created on this machine"
        )
        return

    provisioned = args.existing_dsn is None
    if provisioned:
        # Checked BEFORE anything is written. A reader without Docker should get advice, not a
        # half-provisioned directory they then have to know to delete.
        reason = docker_unavailable_reason()
        if reason:
            raise SystemExit(reason)

    # Resolved before the database is touched: an unusable `--embedder` is a mistake the reader can
    # fix in a second, and discovering it after a PostgreSQL image has been pulled wastes minutes
    # on their first impression of this project. `dim` is also needed to apply the schema, and the
    # schema's vector width is welded to the table.
    embedder = _make_embedder(args.embedder)

    if provisioned:
        print("starting a throwaway PostgreSQL (first run pulls an image, later runs reuse it)")
        dsn, port, compose_path, reused = provision()
        print(f"  {'reused' if reused else 'started'} the quickstart database on 127.0.0.1:{port}")
    else:
        # `redacted_dsn`, because this one is the reader's and may carry a real password. The
        # quickstart's own DSN is printed in full a few lines up, deliberately: see `next_steps`.
        dsn, compose_path = args.existing_dsn, None
        print(f"using the database you supplied: {redacted_dsn(dsn)}")

    # ⛔ **Own table first, default target ONLY if the database demands it.**
    #
    # Two hazards pull in opposite directions and the order below is what satisfies both.
    #
    # On a FRESH database, `quickstart_chunks` alone raises `SchemaTooOld`: global generation
    # migrations are recorded against `chunks` and no other table may be migrated before them.
    # That is documented in the README and it is the state every new reader is in. Applying the
    # default target unconditionally fixes it and was the first attempt.
    #
    # ⚠️ It was wrong, and `--existing-dsn` is where it bites. A reader pointing this at a database
    # they already use has a `chunks` table at THEIR embedder's width, and an unconditional call
    # asks for it at this command's width: `SchemaIncompatible: table 'chunks' uses vector(64),
    # requested dimension is 384`. The quickstart would refuse to run against exactly the database
    # its own flag invites, and its whole isolation story is that it never touches `chunks`.
    #
    # Attempting the quickstart table first distinguishes the two by asking the database rather
    # than guessing: a `SchemaTooOld` means the globals are genuinely absent, which only happens on
    # a database with no recall install to damage. Where `chunks` already exists at another width
    # the globals are already applied, the first call succeeds, and `chunks` is never opened.
    try:
        apply_migrations(dsn, table=QUICKSTART_TABLE, dim=embedder.dim)
    except SchemaTooOld:
        apply_migrations(dsn, table=DEFAULT_TABLE, dim=embedder.dim)
        apply_migrations(dsn, table=QUICKSTART_TABLE, dim=embedder.dim)
    print(f"  applied the schema to {QUICKSTART_TABLE} at dim={embedder.dim}")

    # ⚠️ Its own table AND its own tenant. These 22 documents are fiction about a fictional
    # service; indexed into `chunks`/`default` they would be retrieved beside a reader's real
    # memory the first time they pointed anything real at the same database.
    with PgVectorStore(
        dsn, dim=embedder.dim, table=QUICKSTART_TABLE, tenant=QUICKSTART_TENANT
    ) as store:
        store.check_schema()
        stats = Indexer(
            store,
            embedder,
            context_policy=context_policy_for_profile(embedding_profile_id(embedder)),
        ).index_path(demo_corpus())
        # ⚠️ Report what the STORE holds, not only what this run wrote. Re-indexing skips a file
        # whose content hash is unchanged, so the second run of the quickstart wrote nothing and
        # printed "indexed 0 chunks from 0 files", which reads as a failed index rather than as a
        # corpus that was already there. `stats` is still shown when it is non-zero, because
        # "wrote 22" and "found 22 already present" are genuinely different events.
        held = store.count()
        if stats.chunks:
            print(f"  indexed {stats.chunks} chunks from {stats.files} files\n")
        else:
            print(f"  corpus already indexed: {held} chunks present, nothing to re-read\n")
        print("Three queries. The second one is the reason this project exists.\n")
        # ⚠️ **Development trust, stated here rather than inherited, and this was a real crash.**
        # `TrustPolicy.from_env` defaults to STRICT, and a strict policy refuses an uncalibrated
        # corpus outright: every one of these three queries died with
        # `TrustRefusal: INDEX_NOT_READY` and the reader's first run ended in a traceback. That is
        # correct behaviour for the library and wrong for this command, because this corpus is
        # uncalibrated BY CONSTRUCTION: calibration needs labelled queries the reader has not
        # written yet, and demanding them here would rebuild the seven-step install this command
        # exists to replace.
        #
        # Scoped to these three queries and never exported, so nothing the reader runs afterwards
        # inherits a relaxed posture. `_cli_trust` still prints its uncertified-threshold notice,
        # which is the honest half: the reader is told the number is not a calibration, and the
        # next-steps block below tells them which command fits a real one.
        _run_queries(
            store,
            embedder,
            list(DEMO_QUERIES),
            None,
            _entailment_judge(),
            policy=TrustPolicy.development(),
        )

    print("What just happened, in order:")
    print("  1. an ordinary question, answered from the corpus;")
    print(
        "  2. a question whose nearest match is a RETRACTED claim. `cache_ttl_v2.md` supersedes "
        "`cache_ttl_v1.md`, so the 15 minute answer is marked superseded and points at the 60 "
        "second one that replaced it. A plain vector index returns the retracted answer here;"
    )
    print("  3. a question the corpus cannot answer, refused rather than answered.\n")
    # Explained rather than hidden. Every result above carries `DEGRADED:INDEX_NOT_READY`, which a
    # first-time reader reads as breakage and which is in fact the system declining to overstate
    # what it knows. Suppressing the flag for the demo would be the exact dishonesty this project
    # is about; leaving it unexplained loses the reader instead. So: say what it means, and name
    # the command that clears it.
    print(
        "DEGRADED:INDEX_NOT_READY on every result above is not an error. It is this corpus telling "
        "you its threshold was never fitted to it, so no verdict here is certified. That is the "
        "state a strict deployment REFUSES to answer from, and it is why the demo had to ask for "
        "development trust explicitly. `recall setup` fits a real one.\n"
    )
    for line in next_steps(dsn, provisioned=provisioned, compose_path=compose_path):
        print(line)


def _cmd_uninstall(args: argparse.Namespace) -> None:
    from recall.wizard.uninstall import UninstallRefusal, execute, plan_uninstall

    try:
        plan = plan_uninstall(
            data_root=Path(args.data_root).expanduser(), purge_data=args.purge_data
        )
    except UninstallRefusal as exc:
        raise SystemExit(str(exc)) from exc
    print(plan.render())
    if args.dry_run:
        return
    if not args.yes:
        # ⚠️ Asked AFTER the plan is printed, never before. A confirmation offered ahead of the
        # list is a confirmation of nothing, and this removes containers and rewrites the MCP
        # client's config.
        if not sys.stdin.isatty():
            raise SystemExit(
                "\nrefusing to uninstall without confirmation, and this session has no "
                "terminal to ask in. Re-run with --yes if you meant it, or --dry-run to see "
                "the plan without acting on it."
            )
        answer = input("\nProceed? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            raise SystemExit("cancelled; nothing was removed")
    uninstall_report = execute(plan, purge_data=args.purge_data)
    print(uninstall_report.render())


def _cmd_wizard(args: argparse.Namespace) -> None:
    from recall.wizard.headless import PipelineRefusal, load_config, run_headless

    # ⚠️ The interactive path WRITES a config and then runs that file, rather than running the
    # answers directly. One engine, and the user keeps an artefact they can re-run, hand to
    # somebody else, or put in CI. An interactive flow that installed from memory would be a
    # second installer that drifts from the first.
    if args.gui:
        # ⚠️ Returns rather than falling through. The graphical installer asks the questions,
        # writes the config AND runs it, so continuing into the terminal flow below would ask
        # every question a second time and install twice.
        if args.config is not None:
            raise SystemExit(
                "`--gui` asks the questions itself, so there is nothing for `--config` to do. "
                "Run `recall wizard --headless --config <file>` to replay a saved config, or "
                "`recall wizard --gui` to be asked."
            )
        from recall.desktop.main import install_main

        # ⚠️ Neither `return install_main(...)` nor an unconditional `raise SystemExit(...)`.
        # `main()` is typed to return None, so returning the status is a type error; raising it
        # unconditionally makes a SUCCESSFUL run exit through an exception, which no other
        # branch of this function does and which broke the test asserting the terminal flow does
        # not also run. A non-zero status still has to reach the shell.
        install_status = install_main([])
        if install_status:
            raise SystemExit(install_status)
        return

    if args.config is None:
        from recall.wizard.interactive import (
            InteractiveRefusal,
            ask_config,
            stdin_prompter,
            write_config,
        )

        if args.headless:
            raise SystemExit(
                "`--headless` needs `--config <file>`: there is nothing to run unattended "
                "without one. Omit --headless to be asked the questions."
            )
        default_root = Path.home() / ".recall"
        try:
            prompter = stdin_prompter()
            document = ask_config(prompter, default_root=default_root)
            target = Path(
                prompter.ask(
                    "\nWhere should this configuration be saved?",
                    default=str(default_root / "wizard.json"),
                )
            ).expanduser()
            config_written = write_config(document, target)
        except InteractiveRefusal as exc:
            raise SystemExit(str(exc)) from exc
        except (KeyboardInterrupt, EOFError):
            # Ctrl-C during an interview is a person changing their mind, not a crash. Nothing
            # has been built at this point, so say so rather than printing a traceback.
            raise SystemExit("\ncancelled; nothing was installed") from None
        print(f"\nsaved {config_written}\nrunning it now; re-run with:")
        print(f"  recall wizard --headless --config {config_written}\n")
        args.config = str(config_written)
        # The person just answered every question, so running is precisely what they asked for.
        # Setting it here rather than relaxing the guard below keeps that guard meaning exactly
        # what it always meant: a config supplied on the command line still needs `--headless`.
        args.headless = True

    # ⚠️ **Preserved deliberately, and I removed it once by accident.** A `--config` the user
    # typed still requires `--headless`, because `recall wizard --config x` builds, calibrates
    # and PROMOTES, and doing that without the word "headless" anywhere is the wrong surprise
    # for an installer. Implying it from `--config` looked like a convenience and quietly
    # overturned a decision that had a test guarding it.
    if not args.headless:
        raise SystemExit(
            "`recall wizard --config <file>` runs unattended: it builds, calibrates and "
            "promotes. Say so with `--headless`, or run `recall wizard` with no arguments to "
            "be asked the questions instead."
        )
    # Accepted and silently discarded before. The tenants come from the plan and the table from
    # the migrator's default, so `recall --tenant myproject --table probe wizard ...` promoted
    # into `docs` and migrated the real `chunks` while looking like it had done neither.
    for flag, value, default in (
        ("--tenant", args.tenant, DEFAULT_TENANT),
        ("--table", args.table, DEFAULT_TABLE),
    ):
        if value != default:
            raise SystemExit(
                f"`recall wizard` does not accept {flag}: the tenants come from the corpus "
                f"plan and the table from the migrator, so {flag} {value!r} would be ignored "
                "rather than applied. Remove it."
            )
    try:
        wizard_config = load_config(args.config)
        # The DSNs this command ACTUALLY uses, which the global guard in `main` cannot see. Both,
        # because `migration_dsn` is the DDL owner and is the more privileged of the two.
        # Only the DSNs actually PRESENT. With `data_root` the wizard provisions the database
        # itself and there is no address yet to check; the one it creates is on 127.0.0.1, which
        # is the case `require_secure_dsn` exists to permit. Checking a value that is None here
        # would refuse every desktop install for having no remote credentials to object to.
        for key, value in (
            ("dsn", wizard_config.dsn),
            ("migration_dsn", wizard_config.migration_dsn),
        ):
            if not value:
                continue
            try:
                require_secure_dsn(value)
            except PermissionError as exc:
                raise SystemExit(f"the wizard config's {key} is refused: {exc}") from exc
        # Derived from the config path rather than required, so resumability is on by default
        # without a new mandatory key. `--no-state` opts out entirely; a run that cannot write
        # beside its config should say so rather than have the driver guess.
        wizard_state = (
            None
            if args.no_state
            else Path(args.state)
            if args.state
            else Path(args.config).with_suffix(".state.json")
        )
        wizard_report = run_headless(
            wizard_config,
            progress=lambda step: print(step),
            state_path=wizard_state,
            fresh=args.fresh,
        )
    except PipelineRefusal as exc:
        # A refusal is an operator-actionable message, not a traceback. It is raised before
        # any corpus is built, so there is nothing to clean up either.
        raise SystemExit(str(exc)) from exc
    except ValueError as exc:
        # Anything the wizard's own modules did not spell as a refusal. `PipelineRefusal` is a
        # `ValueError`, so it is caught above; this is the backstop for a value error raised
        # deeper, which used to reach a first-run installer as a stack trace.
        #
        # `TypeError` is deliberately NOT caught. `load_config` now type-checks every key, so a
        # TypeError here is a programming error, and absorbing it into "cannot run with this
        # config" would point the operator at their file for a defect in ours. An earlier
        # version did catch it, and immediately masked a real signature mismatch in a test
        # while returning the exit code that test expected.
        raise SystemExit(f"the wizard cannot run with this config: {exc}") from exc
    print(wizard_report.render())
    # Exit 1 when a corpus was REFUSED or FAILED, and when a corpus is degraded with nothing
    # serving behind it: that tenant answers nothing, so it is not a working install. A
    # degraded corpus over an existing generation IS exit 0, because the predecessor keeps
    # answering and the limitation is named. `HeadlessReport.ok` holds that distinction; CI
    # reads the exit code and nothing else.
    raise SystemExit(0 if wizard_report.ok else 1)
