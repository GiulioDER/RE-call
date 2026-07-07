.PHONY: db-up demo test lint eval
db-up:
	docker compose up -d --wait
demo:
	python -m recall.cli demo
test:
	pytest -v
lint:
	ruff check .
eval:
	python -m recall.eval
