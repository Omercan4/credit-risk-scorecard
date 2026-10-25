PY ?= .venv/bin/python
export PYTHONPATH := src

.PHONY: install download data synthetic population scorecard validate strategy monitor page serve test lint
install:
	python3.12 -m venv .venv && .venv/bin/pip install -e ".[download,report,dev]"
download:
	$(PY) -m crs.cli download
data:
	$(PY) -m crs.cli data
synthetic:
	$(PY) -m crs.cli data --synthetic 20000
population:
	$(PY) -m crs.cli population
scorecard:
	$(PY) -m crs.cli scorecard
validate:
	$(PY) -m crs.cli validate
strategy:
	$(PY) -m crs.cli strategy
monitor:
	$(PY) -m crs.cli monitor
page:
	$(PY) -m crs.cli page
serve:
	.venv/bin/uvicorn crs.serve:app --port 8030
test:
	.venv/bin/pytest -q
lint:
	.venv/bin/ruff check .
