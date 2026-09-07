.PHONY: help install dev test report report-quick web demo clean

PYTHON ?= python3
SAMPLES ?= 2000
SEED ?= 12345

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package in editable mode
	$(PYTHON) -m pip install -e .

dev:  ## Install with the test and web extras
	$(PYTHON) -m pip install -e ".[dev,web]"

test:  ## Run the full test suite
	$(PYTHON) -m pytest tests/ -q

report:  ## Run the experiments, write CSV/JSON, and refresh the README table
	$(PYTHON) benchmarks/run_experiments.py \
		--samples $(SAMPLES) --seed $(SEED) \
		--outdir benchmarks/results --update-readme

report-quick:  ## Same, but small and print-only (no files written)
	$(PYTHON) benchmarks/run_experiments.py --samples 200 --no-write

web:  ## Launch the full Streamlit simulator
	$(PYTHON) -m streamlit run webapp/app.py

demo:  ## Print the path of the standalone HTML demo
	@echo "Open this file in any browser (no server needed):"
	@echo "  file://$(CURDIR)/webapp/simulator.html"

clean:  ## Remove build artefacts and caches
	rm -rf build dist *.egg-info .pytest_cache benchmarks/results
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
