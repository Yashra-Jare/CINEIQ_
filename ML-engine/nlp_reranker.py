import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class NLPReranker:
    def __init__(self):
        print("Initializing NLP VADER Sentiment Analyzer...")
        # Download the VADER lexicon automatically if it's not on your Mac
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')
            
        self.sia = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text_list):
        """Calculates the average sentiment score for a list of reviews."""
        if not text_list:
            return 0.0 # Neutral if no reviews exist
            
        scores = [self.sia.polarity_scores(text)['compound'] for text in text_list]
        avg_score = sum(scores) / len(scores)
        return avg_score

    def rerank_recommendations(self, hybrid_recommendations, movie_reviews_dict):
        """
        Takes the hybrid recommendations and re-ranks them based on sentiment.
        hybrid_recommendations: list of tuples -> [(movie_title, hybrid_score), ...]
        movie_reviews_dict: dict -> {'Movie Title': ['Review 1', 'Review 2']}
        """
        reranked_scores = []
        
        for title, base_score in hybrid_recommendations:
            # Get reviews for the movie (or an empty list if none are found)
            reviews = movie_reviews_dict.get(title, [])
            
            # Get sentiment score (-1 to 1)
            sentiment_score = self.analyze_sentiment(reviews)
            
            # Normalize sentiment to a multiplier between 0.8 and 1.2
            # A highly positive movie gets a 20% boost; a negative one gets penalized.
            sentiment_multiplier = 1.0 + (sentiment_score * 0.2)
            
            final_score = base_score * sentiment_multiplier
            reranked_scores.append((title, final_score))
            
        # Sort by the new sentiment-adjusted score
        reranked_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the titles in the new order
        return [item[0] for item in reranked_scores]