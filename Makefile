.DEFAULT_GOAL := help
UV := uv

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies (including dev and extras)
	$(UV) sync --all-extras

serve:  ## Run the proxy on http://127.0.0.1:8080
	$(UV) run cachellm serve

dev:  ## Run with auto-reload
	$(UV) run cachellm serve --reload

test:  ## Run the test suite
	$(UV) run pytest

test-cov:  ## Run tests with a coverage report
	$(UV) run pytest --cov --cov-report=term-missing

lint:  ## Lint and type-check
	$(UV) run ruff check src tests bench eval
	$(UV) run ruff format --check src tests bench eval
	$(UV) run mypy

fix:  ## Auto-fix lint and formatting
	$(UV) run ruff check --fix src tests bench eval
	$(UV) run ruff format src tests bench eval

bench:  ## Replay 2000 requests against a running proxy
	$(UV) run python -m bench.replay --requests 2000 --concurrency 8 --reset

tune:  ## Sweep similarity thresholds on the labelled corpus
	$(UV) run python -m bench.tune_threshold

compare-models:  ## Score embedding models on safe recall
	$(UV) run python -m bench.compare_models

stats:  ## Print cache statistics
	$(UV) run cachellm stats

flush:  ## Drop every cache entry
	$(UV) run cachellm invalidate --all

up:  ## Start the full stack (proxy, Redis, Prometheus, Grafana)
	docker compose up --build -d
	@echo "proxy    http://localhost:8080/docs"
	@echo "grafana  http://localhost:3000"

down:  ## Stop the stack
	docker compose down

.PHONY: help install serve dev test test-cov lint fix bench tune compare-models stats flush up down
