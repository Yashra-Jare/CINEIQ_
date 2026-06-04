import pandas as pd

class HybridEnsemble:
    def __init__(self, collab_model, content_model, movies_df, links_df):
        self.collab_model = collab_model
        self.content_model = content_model
        self.movies_df = movies_df
        
        # 1. Map TMDB Title -> TMDB ID (From TMDB Dataset)
        # Ensure 'id' column in movies_df is numeric and drop bad data
        self.movies_df['id'] = pd.to_numeric(self.movies_df['id'], errors='coerce')
        valid_movies = self.movies_df.dropna(subset=['id']).copy()
        valid_movies['id'] = valid_movies['id'].astype(int)
        
        self.title_to_tmdb_id = pd.Series(valid_movies['id'].values, index=valid_movies['title']).to_dict()
        
        # 2. Map TMDB ID -> MovieLens ID (From links.csv)
        self.tmdb_to_ml_id = pd.Series(links_df['movieId'].values, index=links_df['tmdbId']).to_dict()

    def recommend(self, user_id, movie_title, top_n=10):
        print(f"Generating Hybrid Recommendations for User {user_id} based on '{movie_title}'...")
        
        # Step 1: Get Content-Based Recommendations (TMDB Data)
        similar_titles = self.content_model.get_recommendations(movie_title, top_n=50)
        
        if isinstance(similar_titles, str):
            return [] # Return empty list if movie not found
            
        hybrid_scores = []
        
        # Step 2: Re-rank using Collaborative Filtering (MovieLens Data)
        for title in similar_titles:
            tmdb_id = self.title_to_tmdb_id.get(title)
            
            if tmdb_id:
                ml_id = self.tmdb_to_ml_id.get(tmdb_id)
                
                if ml_id:
                    # Predict what THIS user would rate THIS similar movie
                    predicted_rating = self.collab_model.predict_rating(user_id, ml_id)
                    hybrid_scores.append({
                        "title": title, 
                        "collab_pred": predicted_rating
                    })
                
        # Step 3: Sort by predicted rating (highest first)
        hybrid_scores.sort(key=lambda x: x["collab_pred"], reverse=True)
        
        # Return the top N dictionary objects
        return hybrid_scores[:top_n]