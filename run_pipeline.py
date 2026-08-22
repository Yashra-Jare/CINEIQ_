import mlflow
import mlflow.sklearn

from config import MODEL_DIR, SAMPLE_SIZE
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "ML-engine"))
from collaborative import CollaborativeModel
from content import ContentModel
from data_loader import load_links, load_movie_metadata, load_movielens_ratings


def train_and_log():
    print("--- Starting CINEIQ training pipeline with MLflow ---")
    ratings_df = load_movielens_ratings()
    movies_df = load_movie_metadata()
    links_df = load_links()

    if any(df is None for df in (ratings_df, movies_df, links_df)):
        raise FileNotFoundError("Required datasets are missing from data/raw/. Pipeline aborted.")

    mlflow.set_experiment("CINEIQ_Hybrid_Engine")
    with mlflow.start_run():
        mlflow.log_param("sample_size", SAMPLE_SIZE if SAMPLE_SIZE is not None else "full")
        mlflow.log_param("collab_model_type", "SVD")
        mlflow.log_param("content_model_type", "TF-IDF + cosine similarity")
        mlflow.log_metric("ratings_loaded", len(ratings_df))
        mlflow.log_metric("unique_users", ratings_df["userId"].nunique())
        mlflow.log_metric("unique_movies", ratings_df["movieId"].nunique())
        mlflow.log_metric("tmdb_titles_loaded", len(movies_df))
        mlflow.log_metric("id_links_loaded", len(links_df))

        collab_model = CollaborativeModel()
        collab_model.train(ratings_df)
        collab_model.save_model()

        content_model = ContentModel()
        content_model.train(movies_df)
        content_model.save_model()

        mlflow.log_artifact(str(MODEL_DIR / "svd_model.pkl"), artifact_path="models")
        mlflow.log_artifact(str(MODEL_DIR / "content_model.pkl"), artifact_path="models")
        print("Training complete. Models and MLflow metadata saved.")


if __name__ == "__main__":
    train_and_log()
