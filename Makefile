.PHONY: install test run lint

install:
	python -m pip install -r requirements.txt
	python -m pip install -e .

test:
	python -m pytest -q

run:
	python examples/train_and_screen.py

lint:
	python -m ruff check src tests examples data
