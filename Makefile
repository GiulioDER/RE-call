.PHONY: db-up db-down db-status open close demo test lint eval

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
test:
	@if [ -z "$$RECALL_TEST_DSN" ]; then \
		echo "RECALL_TEST_DSN is not set: every DB test would skip and this would still exit 0."; \
		echo 'Run: eval "$$(scripts/session-db.sh up)"'; \
		exit 1; \
	fi
	pytest -v
# `python -m ruff`, not bare `ruff`: on some machines the bare name resolves to an older ruff than
# the pinned one, and the two disagree about what passes.
lint:
	python -m ruff check .
eval:
	python -m recall.eval
