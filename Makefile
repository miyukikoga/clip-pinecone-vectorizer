.PHONY: lint format typecheck run test check

lint:
	uv run ruff check

format:
	uv run ruff format

typecheck:
	uv run mypy src tests

run:
	uv run src/main.py

test:
	uv run pytest tests

check: format typecheck test
