.PHONY: db-up db-down db-status open close demo test lint eval
# `db-up` cannot export into your shell from inside make, so it prints the line to eval.
# Run `eval "$$(scripts/session-db.sh up)"` directly to get RECALL_TEST_DSN set.
db-up:
	@scripts/session-db.sh up
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
	pytest -v
lint:
	python -m ruff check .
eval:
	python -m recall.eval
