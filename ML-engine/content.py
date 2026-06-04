import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle
import ast
from config import MODEL_DIR

class ContentModel:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.cosine_sim = None
        self.indices = None
        self.movies_df = None

    def _get_list(self, x, limit=3):
        """Helper to extract top N names from TMDB's stringified JSON lists."""
        if isinstance(x, list):
            names = [i['name'] for i in x]
            if len(names) > limit:
                names = names[:limit]
            return [str.lower(i.replace(" ", "")) for i in names] # Strip spaces so 'Brad Pitt' becomes 'bradpitt'
        return []

    def _get_director(self, x):
        """Helper to extract the director from the crew list."""
        if isinstance(x, list):
            for i in x:
                if i['job'] == 'Director':
                    return [str.lower(i['name'].replace(" ", ""))]
        return []

    def train(self, movies_df):
        print("Training Advanced Content Model (Cast, Genres, Keywords)...")
        self.movies_df = movies_df.drop_duplicates(subset=['title']).reset_index(drop=True)
        
        # Parse the JSON strings into actual Python lists safely
        features = ['cast', 'crew', 'keywords', 'genres']
        for feature in features:
            self.movies_df[feature] = self.movies_df[feature].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

        # Extract the specific data we want
        self.movies_df['director'] = self.movies_df['crew'].apply(self._get_director)
        self.movies_df['cast'] = self.movies_df['cast'].apply(self._get_list)
        self.movies_df['keywords'] = self.movies_df['keywords'].apply(self._get_list, limit=5)
        self.movies_df['genres'] = self.movies_df['genres'].apply(self._get_list)

        # Create the Metadata "Soup" (Combine everything into one string per movie)
        def create_soup(x):
            return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + ' '.join(x['director']) + ' ' + ' '.join(x['genres'])
            
        self.movies_df['metadata_soup'] = self.movies_df.apply(create_soup, axis=1)
        
        # Fit TF-IDF on our new rich metadata
        tfidf_matrix = self.tfidf.fit_transform(self.movies_df['metadata_soup'])
        print("Computing Cosine Similarity Matrix...")
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df['title'])
        print("Content Model Training complete!")

    def save_model(self, filename="content_model.pkl"):
        path = MODEL_DIR / filename
        data_to_save = {
            'cosine_sim': self.cosine_sim,
            'indices': self.indices,
            'movies_df': self.movies_df[['title', 'id']]
        }
        with open(path, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"Content model saved to {path}")

    def load_model(self, filename="content_model.pkl"):
        path = MODEL_DIR / filename
        try:
            with open(path, 'rb') as f:
                data_loaded = pickle.load(f)
            self.cosine_sim = data_loaded['cosine_sim']
            self.indices = data_loaded['indices']
            self.movies_df = data_loaded['movies_df']
            print(f"Content model loaded from {path}")
        except FileNotFoundError:
            print(f"Error: Could not find {path}.")

    def get_recommendations(self, title, top_n=10):
        if title not in self.indices:
            return f"Movie '{title}' not found in the dataset."
        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]
        return self.movies_df['title'].iloc[movie_indices].tolist()