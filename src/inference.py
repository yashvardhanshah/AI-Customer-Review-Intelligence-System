import os
import json
import joblib
from src.preprocessing import preprocess_text

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load best model
model = joblib.load(os.path.join(BASE, "models", "linear_svm_best.pkl"))

# Load TF-IDF vectorizer
vectorizer = joblib.load(os.path.join(BASE, "data", "processed", "tfidf_vectorizer.pkl"))

# Load label mapping
with open(os.path.join(BASE, "models", "label_mapping.json"), "r") as f:
    label_mapping = json.load(f)


def predict(text: str) -> str:
    """
    Predict sentiment of a raw review string.
    Returns: 'Positive', 'Negative', or 'Neutral'
    """
    clean_text = preprocess_text(text)
    vector = vectorizer.transform([clean_text])
    pred = model.predict(vector)[0]
    return label_mapping[str(pred)] if isinstance(pred, int) else pred