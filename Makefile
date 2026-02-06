.PHONY: install lint format test check all clean

install:
	uv sync --all-groups

lint:
	uv run ruff check src tests examples

format:
	uv run ruff format src tests examples
	uv run ruff check --fix src tests examples

test:
	uv run pytest

check: lint test

all: format check

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .pytest_cache .ruff_cache dist build *.egg-info
