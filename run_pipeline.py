import mlflow
import mlflow.sklearn
from ml_engine.data_loader import load_movielens_ratings, load_movie_metadata, load_links
from ml_engine.collaborative import CollaborativeModel
from ml_engine.content import ContentModel
from config import SAMPLE_SIZE

def train_and_log():
    print("--- Starting Full Pipeline Run with MLflow ---")
    
    # 1. Load All Data
    ratings_df = load_movielens_ratings()
    movies_df = load_movie_metadata()
    links_df = load_links()
    
    if ratings_df is None or movies_df is None or links_df is None:
        print("Error: Missing data files. Pipeline aborted.")
        return

    # Start MLflow Tracking Run
    mlflow.set_experiment("CINEIQ_Hybrid_Engine")
    
    with mlflow.start_run():
        print(f"Tracking run with Sample Size: {SAMPLE_SIZE}")
        
        # Log parameters
        mlflow.log_param("sample_size", SAMPLE_SIZE)
        mlflow.log_param("collab_model_type", "SVD")
        mlflow.log_param("content_model_type", "TF-IDF")
        
        # 2. Train Collaborative Model
        collab_model = CollaborativeModel()
        collab_model.train(ratings_df)
        collab_model.save_model()
        
        # 3. Train Content Model
        content_model = ContentModel()
        content_model.train(movies_df)
        content_model.save_model()
        
        print("✅ Training complete and models saved to disk!")
        print("✅ Run tracked in MLflow.")

if __name__ == "__main__":
    train_and_log()