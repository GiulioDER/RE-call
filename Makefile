.PHONY: db-up demo test lint
db-up:
	docker compose up -d --wait
demo:
	python -m recall.cli demo
test:
	pytest -v
lint:
	ruff check .
