.PHONY: build lint format test clean

build:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

clean:
	rm -rf .venv .pytest_cache .ruff_cache
