lint:
	uv run ruff check

format:
	uv run ruff format

run:
	uv run src/main.py

test:
	uv run pytest tests