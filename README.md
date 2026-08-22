# 🎬 CINEIQ — Hybrid Movie Recommendation Engine

CINEIQ combines **content-based filtering** and **collaborative filtering** to generate personalized movie recommendations. A Streamlit dashboard provides recommendations and user taste visualizations.

## Architecture

```text
TMDB metadata ──> TF-IDF + cosine similarity ──> content candidates
                                                     │
MovieLens ratings ──> SVD ──> user-specific predicted rating
                                                     │
                                                     ▼
                                             hybrid ranking
                                                     │
                                                     ▼
                                                  FastAPI
                                                     │
                                                     ▼
                                                Streamlit
```

## Project structure

```text
CINEIQ/
├── ml_engine/
│   ├── __init__.py
│   ├── collaborative.py
│   ├── content.py
│   ├── data_loader.py
│   ├── ensemble.py
│   └── nlp_reranker.py
├── dashboard/
│   ├── __init__.py
│   ├── app.py
│   └── visuals.py
├── data/raw/                 # datasets are not included in the repository
├── models/                   # generated model files
├── api.py
├── config.py
├── run_pipeline.py
└── requirements.txt
```

## Data

Place these files under `data/raw/`:

- MovieLens 25M: `ratings.csv`, `links.csv`
- TMDB/The Movies Dataset: `movies_metadata.csv`, `credits.csv`, `keywords.csv`
- Optional: `IMDB Dataset.csv` for the standalone VADER sentiment module

`SAMPLE_SIZE = 100000` is used by default so local development does not attempt to load all 25M MovieLens ratings into memory. Set it to `None` when using the full dataset and sufficient compute.

## Run

```bash
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
uvicorn api:app --reload
streamlit run dashboard/app.py
```

Open the Streamlit URL shown in the terminal. The API also exposes `/health` and `/recommend`.

## Recommendation flow

1. **Content model:** combines cast, director, genres and keywords into a metadata representation, vectorizes it with TF-IDF and computes cosine similarity.
2. **Candidate generation:** retrieves the most similar movies to the input title.
3. **Collaborative model:** Surprise SVD predicts a rating for the selected user and each candidate.
4. **Hybrid ranking:** candidates are sorted by the user's predicted rating.
5. **Dashboard:** Streamlit displays recommendations and user-level genre, decade and cast/director statistics calculated from the user's rating history.

## Sentiment module

`ml_engine/nlp_reranker.py` implements VADER scoring and a bounded ±20% multiplier **when a movie-title-to-review mapping is supplied**. The standard IMDb 50K sentiment dataset contains reviews and sentiment labels but does not provide a reliable movie-title key, so the module is intentionally not connected to the main pipeline by default. This avoids assigning a review to the wrong movie.

## Experiment tracking

`run_pipeline.py` logs training configuration and dataset statistics to MLflow and stores the trained model files as MLflow artifacts.
