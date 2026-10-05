.PHONY: setup fetch preprocess index run test evaluate

setup:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt

fetch:
	python -m ingestion.fetch_stackoverflow --per-tag 40 --max-answers 2

preprocess:
	python -m ingestion.preprocess

index: preprocess
	python -m ingestion.build_index

run:
	streamlit run app/streamlit_app.py

test:
	python -m pytest -q

evaluate:
	python -m evaluation.run_retrieval
