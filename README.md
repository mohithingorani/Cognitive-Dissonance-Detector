# Cognitive Dissonance Detector (Text Mining Project)


This repository contains a multi-file prototype for a **Cognitive Dissonance Detector** — a text-mining system that identifies semantic-emotional contradictions in text (e.g., "I love my job" written with negative affective signals), producing a *dissonance score* and interpretable signals.


The project is designed as a research/portfolio piece and includes:
- data ingestion and simple annotation utilities
- preprocessing and linguistic feature extraction (semantic and affective)
- a hybrid model combining transformer sentence embeddings with affective-signal detectors
- interpretable output highlighting contradictory spans and a numeric dissonance score
- evaluation scripts and a Streamlit demo + FastAPI backend


## Quick start
1. Create a virtualenv: `python -m venv venv && source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Inspect sample data: `data/sample_texts.csv`
4. Train a demo model: `python -m app.train --demo`
5. Run API: `uvicorn app.api:app --reload --port 8000`
6. Run UI: `streamlit run app/ui_streamlit.py`