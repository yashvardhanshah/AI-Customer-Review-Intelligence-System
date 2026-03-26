from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_vectorizer():
    return TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2),
        min_df=5,
        max_df=0.9
    )