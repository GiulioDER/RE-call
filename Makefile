.PHONY: db-up db-down db-status open close demo test test-serial lint eval

# `db-up` starts this checkout's container, but make CANNOT export into your shell, so it only
# prints the line. Run the eval form yourself to actually get a DSN:
#
#     eval "$$(scripts/session-db.sh up)"
#
# `test` below refuses to run without one, so `make db-up && make test` fails loudly instead of
# reporting a green run in which every DB test silently skipped.
db-up:
	@scripts/session-db.sh up
	@echo
	@echo "NOTE: make cannot export into your shell. Run this to set RECALL_TEST_DSN:"
	@echo '      eval "$$(scripts/session-db.sh up)"'
db-down:
	@scripts/session-db.sh down
db-status:
	@scripts/session-db.sh status
open:
	@scripts/session-open.sh
close:
	@scripts/session-close.sh
demo:
	python -m recall.cli demo

# `test` runs in PARALLEL, with four workers, and that is the whole reason the suite is usable
# several times a day. It is LARGE rather than slow: 6,563 tests, no hotspot worth removing, so
# the only lever on its wall clock is `pytest-xdist`. What makes that safe against one shared
# database is `tests/conftest.py::_isolate_xdist_worker`, which gives every worker a database of
# its own; read that docstring before raising or lowering the worker count.
#
# Measured 2026-08-23 on this workstation, same commit, same container, nothing else of mine
# running: serial 49:58, `-n 4` 22:16, `-n 6` 21:08 with one worker killed for memory. Four is
# the ceiling because six bought roughly a minute and cost a crashed worker on a 12 GB machine.
# Override on a bigger box with `make test N=8`.
#
# The worker count is sized at LAUNCH TIME by `scripts/suite-preflight.sh`, from the memory the
# machine actually has free, because the numbers above are from an idle box and this machine is
# routinely not idle: several sessions, Docker, and other sessions' embedding runs share its
# 12 GB, and four workers on a loaded box do not finish slower, they get OOM-killed
# (`node down: Not properly terminated`) and the retries turn 14 minutes into 40+. `N=<n>` still
# wins over the preflight, taken verbatim.
#
# Deliberately the DEFAULT `--dist load`, one test at a time to whichever worker is free, and not
# `--dist loadfile`. Keeping a file's tests together would hide, rather than fix, a module that
# owns a fixed-name database for the whole file: `test_beam_transfer_index_guards.py` was exactly
# that, nine of its ten tests erroring at once, and it now names its throwaway database after the
# worker instead. Scheduling is not isolation.
test:
	@if [ -z "$$RECALL_TEST_DSN" ]; then \
		echo "RECALL_TEST_DSN is not set: every DB test would skip and this would still exit 0."; \
		echo 'Run: eval "$$(scripts/session-db.sh up)"'; \
		exit 1; \
	fi
	@W=$$(scripts/suite-preflight.sh nworkers); \
	if [ "$$W" = 1 ]; then \
		echo "preflight picked 1 worker: running serially (readable output, ~50 min)."; \
		pytest -q; \
	else \
		pytest -q -n "$$W"; \
	fi

# The serial form, for when a failure needs an ordered, readable report rather than four workers
# interleaving theirs. The SAME tests: `-n` changes scheduling, never selection.
test-serial:
	@if [ -z "$$RECALL_TEST_DSN" ]; then \
		echo "RECALL_TEST_DSN is not set: every DB test would skip and this would still exit 0."; \
		echo 'Run: eval "$$(scripts/session-db.sh up)"'; \
		exit 1; \
	fi
	pytest -q

# `python -m ruff`, not bare `ruff`: on some machines the bare name resolves to an older ruff than
# the pinned one, and the two disagree about what passes.
lint:
	python -m ruff check .
eval:
	python -m recall.eval
