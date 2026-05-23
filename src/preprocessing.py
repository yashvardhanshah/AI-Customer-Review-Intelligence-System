from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))

# Preserve negations so "not happy" ≠ "happy"
negations = {"no", "not", "never", "neither", "nor", "none", "nobody", "nothing", "nowhere", "hardly", "barely", "scarcely"}
stop_words -= negations

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = text.split()
    processed = []

    for word in words:
        if word not in stop_words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            processed.append(lemma)

    return " ".join(processed)