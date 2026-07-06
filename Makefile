.PHONY: db-up demo test lint
db-up:
	docker compose up -d
demo:
	python -m recall.cli demo
test:
	pytest -v
lint:
	ruff check .
