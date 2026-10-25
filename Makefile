PYTHON ?= python
NPM ?= npm
COMPOSE ?= docker compose

.PHONY: help install test eval data-quality build dev-backend dev-frontend docker-build docker-up docker-down

help:
	@echo "ElectriCast targets:"
	@echo "  make install       Install backend and frontend dependencies"
	@echo "  make test          Run the backend pytest suite"
	@echo "  make eval          Regenerate backend/results.json (full backtest)"
	@echo "  make data-quality  Regenerate backend/data_quality.json"
	@echo "  make build         Build the frontend bundle"
	@echo "  make dev-backend   Run FastAPI on http://localhost:8000"
	@echo "  make dev-frontend  Run Vite on http://localhost:3000"
	@echo "  make docker-build  Build both Docker images"
	@echo "  make docker-up     Build and start the Docker stack"
	@echo "  make docker-down   Stop the Docker stack"

install:
	$(PYTHON) -m pip install -r backend/requirements.txt
	cd frontend && $(NPM) ci

test:
	$(PYTHON) -m pytest backend/tests -q

eval:
	$(PYTHON) backend/evaluate.py

data-quality:
	$(PYTHON) backend/build_data_quality.py

build:
	cd frontend && $(NPM) run build

dev-backend:
	cd backend && $(PYTHON) -m uvicorn main:app --reload --port 8000

dev-frontend:
	cd frontend && $(NPM) run dev

docker-build:
	$(COMPOSE) build

docker-up:
	$(COMPOSE) up -d --build

docker-down:
	$(COMPOSE) down
