PY ?= .venv/bin/python
export PYTHONPATH := src

.PHONY: install download data synthetic population scorecard validate strategy monitor regression policy page serve notebook test lint
install:
	python3.12 -m venv .venv && .venv/bin/pip install -e ".[download,report,notebook,dev]"
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
regression:
	$(PY) -m crs.cli regression
policy:
	$(PY) -m crs.cli policy
page:
	$(PY) -m crs.cli page
serve:
	.venv/bin/uvicorn crs.serve:app --port 8030
notebook:
	.venv/bin/jupyter lab notebooks/walkthrough.ipynb
test:
	.venv/bin/pytest -q
lint:
	.venv/bin/ruff check .
