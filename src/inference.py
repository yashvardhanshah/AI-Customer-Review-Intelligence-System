import joblib
from src.preprocessing import preprocess_text

# Load best model (change if needed later)
model = joblib.load("../models/linear_svm_best.pkl")

# Load TF-IDF
vectorizer = joblib.load("../data/processed/tfidf_vectorizer.pkl")

# Load label mapping
import json
with open("../models/label_mapping.json", "r") as f:
    label_mapping = json.load(f)

def predict(text):
    clean_text = preprocess_text(text)
    vector = vectorizer.transform([clean_text])
    pred = model.predict(vector)[0]
    
    return label_mapping[str(pred)] if isinstance(pred, int) else pred