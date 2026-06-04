import pandas as pd
from surprise import Reader, Dataset, SVD
import pickle
from config import MODEL_DIR

class CollaborativeModel:
    def __init__(self):
        self.model = SVD()
        # MovieLens ratings are on a 0.5 to 5.0 scale
        self.reader = Reader(rating_scale=(0.5, 5.0))

    def train(self, ratings_df):
        print("Preparing data for the Surprise library...")
        # Surprise strictly requires these three columns in order
        data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], self.reader)
        trainset = data.build_full_trainset()
        
        print("Training SVD Model (this might take a minute depending on SAMPLE_SIZE)...")
        self.model.fit(trainset)
        print("SVD Training complete!")

    def save_model(self, filename="svd_model.pkl"):
        path = MODEL_DIR / filename
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")

    def load_model(self, filename="svd_model.pkl"):
        path = MODEL_DIR / filename
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {path}")

    def predict_rating(self, user_id, movie_id):
        """Predicts what rating a specific user would give a specific movie."""
        prediction = self.model.predict(user_id, movie_id)
        return prediction.est