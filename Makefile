.PHONY: help install test test-all lint lint-architecture typecheck check doctor clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-20s\033[0m %s\n",$$1,$$2}'

install:  ## Create the venv and install everything, from the lockfile
	uv sync --locked --all-groups

test:  ## Deterministic tier only. Hard gate: must be 100% green (playbook 2.2)
	uv run pytest -m deterministic

test-all:  ## Every tier, including tests that need real credentials
	uv run pytest

lint:  ## ruff
	uv run ruff check .
	uv run ruff format --check .

lint-architecture:  ## ADR-010: no provider SDK outside the gateway
	uv run python scripts/lint_architecture.py

typecheck:  ## mypy, strict
	uv run mypy

check: lint typecheck lint-architecture test  ## Everything CI runs, in CI's order

doctor:  ## Verify the environment (needs .env)
	uv run dynaflows doctor

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
