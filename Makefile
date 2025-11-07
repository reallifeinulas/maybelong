PYTHON=python3
VENV=.venv

.PHONY: init run test fmt lint

init:
python3 -m venv $(VENV)
$(VENV)/bin/pip install --upgrade pip
$(VENV)/bin/pip install -r requirements.txt

run:
$(PYTHON) -m src.main

test:
$(PYTHON) -m pytest

fmt:
$(PYTHON) -m black src tests
$(PYTHON) -m ruff check --fix src tests

lint:
$(PYTHON) -m ruff check src tests
