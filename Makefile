# Shortcuts for the commands you actually run. `make help` lists them.
#
# Nothing here is required - every target is a thin wrapper around docker
# compose, pytest or the ppg CLI, and the equivalent long form is in the
# README. It exists so you do not have to remember that the compose service is
# called `api` while the project is called `ppg`.

COMPOSE  := docker compose
SERVICE  := api
VENV     := .venv
PY       := $(VENV)/bin/python
PPG      := $(VENV)/bin/ppg
URL      := http://localhost:$(shell grep -E '^PPG_PORT=' .env 2>/dev/null | cut -d= -f2 | grep -E '^[0-9]+$$' || echo 8000)

# Extra flags for the generate/batch targets:
#   make generate ARGS="--sex male --age 62 --profession fisherman"
ARGS ?=

.DEFAULT_GOAL := help
.PHONY: help up down restart logs ps health open shell build rebuild \
        install install-gpu models setup serve doctor warmup generate batch \
        clear test lint fmt typecheck check clean clean-docker nuke-data

##@ Running (Docker - the normal way)

up: ## Start the service in the background
	@test -f .env || cp .env.example .env
	$(COMPOSE) up -d
	@echo "  -> $(URL)  (readiness: make health)"

down: ## Stop the service. Generated images and the database are kept
	$(COMPOSE) down

restart: ## Restart without rebuilding
	$(COMPOSE) restart $(SERVICE)

rebuild: ## Rebuild the image and restart - run this after changing code
	$(COMPOSE) up -d --build
	@echo "  -> $(URL)"

logs: ## Follow the logs (Ctrl-C to stop watching, service keeps running)
	$(COMPOSE) logs -f $(SERVICE)

ps: ## Show whether it is running
	@$(COMPOSE) ps

health: ## Wait until the model is loaded, then print the status
	@echo "waiting for the model to load..."
	@until curl -fsS -m 3 $(URL)/readyz 2>/dev/null | grep -q '"model_loaded":true'; do sleep 3; done
	@curl -fsS $(URL)/readyz; echo

open: ## Open the gallery in a browser
	@xdg-open $(URL) 2>/dev/null || open $(URL) 2>/dev/null || echo "$(URL)"

shell: ## Open a shell inside the running container
	$(COMPOSE) exec $(SERVICE) /bin/bash

build: ## Build the image without starting anything
	$(COMPOSE) build $(SERVICE)

##@ First-time setup

setup: ## Host prep: GPU support for Docker, disk, Ollama (asks before each step)
	./scripts/setup-host.sh

install: ## Create the local virtualenv for the CLI and the tests
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	@echo "  installed. GPU extras (torch, ~3GB): make install-gpu"

install-gpu: install ## Also install torch and diffusers, for running without Docker
	$(PY) -m pip install -e ".[gpu]"

models: ## Download the model weights (~7.3GB, once)
	$(PY) scripts/download-models.py

##@ Using it from the command line

doctor: ## Check GPU, disk, weights and Ollama, and say what is wrong
	$(PPG) doctor

warmup: ## Load the model and time one image
	$(PPG) warmup

generate: ## Generate one avatar. make generate ARGS="--sex male --age 62"
	$(PPG) generate $(ARGS)

batch: ## Generate several. make batch ARGS="-n 20"
	$(PPG) batch $(ARGS)

clear: ## Delete every generated avatar (asks first). Weights are untouched
	$(PPG) clear

serve: ## Run the server directly, without Docker (needs make install-gpu)
	$(PPG) serve

##@ Development

test: ## Run the test suite (no GPU, no weights, no network)
	$(PY) -m pytest -q

lint: ## Check formatting and lint
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check src tests

fmt: ## Reformat and auto-fix what can be fixed
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/ruff format src tests

typecheck: ## Run mypy
	$(VENV)/bin/mypy src/

check: lint typecheck test ## Everything CI runs

##@ Housekeeping

clean: ## Remove Python caches and build artefacts
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \
	  -o -name .mypy_cache -o -name '*.egg-info' \) -prune -exec rm -rf {} + 2>/dev/null || true

clean-docker: ## Reclaim Docker build cache (often tens of GB)
	docker builder prune -f

nuke-data: ## DESTRUCTIVE: delete generated images, the database AND the weights
	@echo "This deletes ./data - every generated avatar and all $$(du -sh data 2>/dev/null | cut -f1) of model weights."
	@printf 'Type "yes" to continue: ' && read ans && [ "$$ans" = "yes" ] || (echo "cancelled."; exit 1)
	$(COMPOSE) down 2>/dev/null || true
	rm -rf data
	@echo "gone. 'make models' to download the weights again."

##@ Help

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nProfile Photo Generator\n"} \
	  /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2 } \
	  /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)
	@echo ""
