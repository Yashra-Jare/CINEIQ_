import pandas as pd
from config import RAW_DATA_DIR, SAMPLE_SIZE

def load_movielens_ratings():
    """Loads a subset of the MovieLens ratings dataset to save memory."""
    ratings_path = RAW_DATA_DIR / "ratings.csv"
    try:
        df = pd.read_csv(ratings_path, nrows=SAMPLE_SIZE)
        print(f"Loaded {len(df)} MovieLens ratings.")
        return df
    except FileNotFoundError:
        print(f"Error: ratings.csv not found at {ratings_path}")
        return None

def load_movie_metadata():
    """Loads TMDB metadata, credits, and keywords, and merges them for the Content Model."""
    metadata_path = RAW_DATA_DIR / "movies_metadata.csv"
    credits_path = RAW_DATA_DIR / "credits.csv"
    keywords_path = RAW_DATA_DIR / "keywords.csv"
    
    try:
        # Load the files
        df = pd.read_csv(metadata_path, low_memory=False)
        credits = pd.read_csv(credits_path)
        keywords = pd.read_csv(keywords_path)
        
        # Clean IDs (Kaggle TMDB has 3 corrupted rows we must drop)
        df = df.drop([19730, 29503, 35587], errors='ignore')
        df['id'] = df['id'].astype('int')
        credits['id'] = credits['id'].astype('int')
        keywords['id'] = keywords['id'].astype('int')
        
        # Merge all three datasets on the movie ID
        df = df.merge(credits, on='id')
        df = df.merge(keywords, on='id')
        
        df = df.dropna(subset=['title']) 
        print(f"Loaded and merged {len(df)} TMDB records with Cast/Crew/Keywords.")
        return df
    except FileNotFoundError:
        print("Error: Ensure movies_metadata.csv, credits.csv, and keywords.csv are in data/raw/")
        return None

def load_imdb_reviews():
    """Loads the IMDB 50K Reviews dataset for the NLP pipeline."""
    imdb_path = RAW_DATA_DIR / "IMDB Dataset.csv"
    try:
        df = pd.read_csv(imdb_path)
        print(f"Loaded {len(df)} IMDB reviews.")
        return df
    except FileNotFoundError:
        print(f"Error: IMDB Dataset.csv not found at {imdb_path}")
        return None
def load_links():
    """Loads the mapping between MovieLens IDs and TMDB IDs."""
    links_path = RAW_DATA_DIR / "links.csv"
    try:
        df = pd.read_csv(links_path)
        # Drop rows where tmdbId is missing
        df = df.dropna(subset=['tmdbId']) 
        # Convert tmdbId to integer for clean mapping
        df['tmdbId'] = df['tmdbId'].astype(int)
        print(f"Loaded {len(df)} ID links.")
        return df
    except FileNotFoundError:
        print(f"Error: links.csv not found at {links_path}")
        return None