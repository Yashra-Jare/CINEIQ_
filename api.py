from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "ML-engine"))
from collaborative import CollaborativeModel
from content import ContentModel
from data_loader import load_links, load_movie_metadata
from ensemble import HybridEnsemble

app = FastAPI(title="CINEIQ Recommendation API", version="1.0")


class RecommendationRequest(BaseModel):
    user_id: int = Field(..., ge=1)
    movie_title: str = Field(..., min_length=1)
    top_n: int = Field(default=5, ge=1, le=10)


@lru_cache(maxsize=1)
def get_engine():
    movies_df = load_movie_metadata()
    links_df = load_links()
    if movies_df is None or links_df is None:
        raise RuntimeError("TMDB metadata or MovieLens links are missing from data/raw/.")

    collab = CollaborativeModel()
    content = ContentModel()
    collab.load_model()
    content.load_model()
    return HybridEnsemble(collab, content, movies_df, links_df)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    try:
        engine = get_engine()
        results = engine.recommend(request.user_id, request.movie_title, request.top_n)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations found for '{request.movie_title}'. Check the title or dataset mapping.",
        )

    return [
        {
            "movie": item["title"],
            "predicted_rating": round(item["collab_pred"], 3),
            "explanation": (
                f"{item['title']} was selected because it is content-similar to "
                f"'{request.movie_title}' and received a predicted rating of "
                f"{item['collab_pred']:.2f} for User {request.user_id}."
            ),
        }
        for item in results
    ]
