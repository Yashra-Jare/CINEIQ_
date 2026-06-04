# 🎬 CINEIQ — Explainable Hybrid Movie Recommendation Engine

> *Recommends movies you'll love — and tells you exactly why.*

---

## 📌 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
- [Dashboard](#dashboard)
- [MLOps Tracking](#mlops-tracking)

---

## Overview

CINEIQ is a hybrid movie recommendation engine that combines **Collaborative Filtering (SVD)** and **Content-Based Filtering (TF-IDF)** into a single ensemble, then re-ranks results using **NLP sentiment analysis** (NLTK VADER). Every recommendation comes with a plain-English explanation — no black boxes.

Built as a full ML pipeline with experiment tracking via MLflow and a live Streamlit dashboard.

---

## Architecture

```
MovieLens 25M Ratings          TMDB 45K Metadata (+ Credits + Keywords)
        │                                        │
        ▼                                        ▼
┌──────────────────┐                 ┌─────────────────────────┐
│ CollaborativeModel│                │      ContentModel        │
│  SVD via Surprise │                │ TF-IDF on metadata soup  │
│  (collaborative.py)               │  cast + director +       │
└────────┬─────────┘                │  genres + keywords       │
         │                          │  (content.py)            │
         │                          └────────────┬─────────────┘
         │                                       │
         └──────────────┬────────────────────────┘
                        ▼
               ┌─────────────────┐
               │  HybridEnsemble  │
               │  (ensemble.py)   │
               │  Content → CF    │
               │  re-ranking      │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │   NLPReranker    │
               │ (nlp_reranker.py)│
               │  VADER sentiment │
               │  ±20% score adj  │
               └────────┬────────┘
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
       ┌────────────┐     ┌─────────────────┐
       │  FastAPI   │     │   Streamlit App  │
       │  Backend   │     │    (app.py)      │
       └────────────┘     └─────────────────┘
```

---

## Project Structure

```
cineiq/
│
├── ml_engine/
│   ├── collaborative.py     # SVD model — train, save, load, predict
│   ├── content.py           # TF-IDF + cosine similarity content model
│   ├── ensemble.py          # Hybrid merger: content → CF re-ranking
│   ├── nlp_reranker.py      # VADER sentiment re-ranker (±20% boost/penalty)
│   └── data_loader.py       # Loads MovieLens ratings, TMDB metadata, links
│
├── data/
│   └── raw/                 # Place datasets here (gitignored)
│       ├── ratings.csv
│       ├── links.csv
│       ├── movies_metadata.csv
│       ├── credits.csv
│       ├── keywords.csv
│       └── IMDB Dataset.csv
│
├── models/                  # Saved .pkl models (gitignored)
│   ├── svd_model.pkl
│   └── content_model.pkl
│
├── app.py                   # Streamlit dashboard
├── visuals.py               # Plotly charts (radar, bar, affinities)
├── run_pipeline.py          # End-to-end training orchestrator + MLflow
├── config.py                # Paths, SAMPLE_SIZE, MODEL_DIR
└── requirements.txt
```

---

## Tech Stack

| Layer | Library | Purpose |
|-------|---------|---------|
| Collaborative Filtering | `scikit-surprise` (SVD) | Matrix factorization on user-item ratings |
| Content Filtering | `scikit-learn` (TF-IDF + cosine similarity) | Metadata soup similarity |
| Sentiment Analysis | `NLTK VADER` | Re-rank by audience review sentiment |
| Experiment Tracking | `MLflow` | Log params, metrics, and model artifacts |
| Dashboard | `Streamlit` + `Plotly` | Interactive UI and taste profile charts |
| Data | MovieLens 25M + TMDB 45K | Ratings and rich movie metadata |

---

## Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/cineiq.git
cd cineiq
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Datasets

Place these files in `data/raw/`:

| File | Source |
|------|--------|
| `ratings.csv`, `links.csv` | [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) |
| `movies_metadata.csv`, `credits.csv`, `keywords.csv` | [Kaggle TMDB 45K](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) |
| `IMDB Dataset.csv` | [Kaggle IMDB 50K Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |

### 3. Configure Sample Size

In `config.py`, `SAMPLE_SIZE = 100000` by default (safe for local machines).  
Set to `None` to train on the full 25M dataset.

### 4. Train the Models

```bash
python run_pipeline.py
```

Expected output:
```
--- Starting Full Pipeline Run with MLflow ---
Loaded 100000 MovieLens ratings.
Loaded and merged 45466 TMDB records with Cast/Crew/Keywords.
Training SVD Model...           ✓
Training Content Model...       ✓
✅ Training complete and models saved to disk!
✅ Run tracked in MLflow.
```

### 5. Run the Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` — enter any MovieLens User ID to explore recommendations and taste profile charts.

---

## How It Works

### Step 1 — Content-Based Filtering (`content.py`)

For each movie, a **metadata "soup"** is constructed by concatenating:
- Top 3 cast members (space-stripped: `"bradpitt"`)
- Director
- Top 5 keywords
- Genres

TF-IDF vectorizes these strings and **cosine similarity** finds the 50 most similar movies to the user's input title.

### Step 2 — Collaborative Filtering Re-Ranking (`collaborative.py` + `ensemble.py`)

For each of the 50 content-similar candidates, **SVD** predicts what rating *this specific user* would give that movie. The list is re-sorted by predicted rating — personalizing what would otherwise be identical results for every user.

The ID bridge in `ensemble.py` maps:
```
TMDB Title → TMDB ID → MovieLens ID → SVD predicted rating
```

### Step 3 — Sentiment Re-Ranking (`nlp_reranker.py`)

NLTK VADER scores each movie's audience reviews (-1 to +1). Converted to a multiplier:

```python
sentiment_multiplier = 1.0 + (sentiment_score * 0.2)
final_score = base_score * sentiment_multiplier
```

Highly praised movies get up to a **+20% boost**; poorly reviewed ones get up to a **-20% penalty**.

### Step 4 — Explainability

The FastAPI layer generates a plain-English reason for each recommendation based on the signals that drove it — director match, genre overlap, collaborative taste similarity, and sentiment score.

---

## Dashboard

The Streamlit dashboard (`app.py`) has two tabs:

**Discover Movies** — Enter a movie title + User ID, hit generate, and the API returns ranked recommendations with explanations.

**My Taste Profile** — Three Plotly visualizations derived from your rating history:
- Genre Affinity Radar Chart
- Decade Preference Bar Chart
- Top Actors & Directors Affinity Chart

---

## MLOps Tracking

All training runs are logged to MLflow automatically via `run_pipeline.py`.

```bash
mlflow ui
# Visit http://localhost:5000
```

Each run logs: `sample_size`, `collab_model_type`, `content_model_type`, and serialized model artifacts.

---

Built with ❤️ by [Yashraj] & [Aayush]
