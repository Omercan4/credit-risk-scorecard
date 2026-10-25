PY ?= .venv/bin/python
export PYTHONPATH := src

.PHONY: install download data synthetic population test lint
install:
	python3.12 -m venv .venv && .venv/bin/pip install -e ".[download,dev]"
download:
	$(PY) -m crs.cli download
data:
	$(PY) -m crs.cli data
synthetic:
	$(PY) -m crs.cli data --synthetic 20000
population:
	$(PY) -m crs.cli population
test:
	.venv/bin/pytest -q
lint:
	.venv/bin/ruff check .
