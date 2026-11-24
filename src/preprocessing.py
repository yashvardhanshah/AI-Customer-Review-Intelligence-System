import re
from functools import lru_cache

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.sentiment.util import mark_negation

# Negations carry sentiment ("not good" != "good"), so they are never dropped as stopwords
NEGATIONS = {"no", "not", "never", "neither", "nor", "none",
             "nobody", "nothing", "nowhere", "hardly", "barely", "scarcely"}
stop_words = set(stopwords.words("english")) - NEGATIONS

lemmatizer = WordNetLemmatizer()

# Words + sentence-ending punctuation (punctuation is kept so negation scope can end at it)
tokenizer = RegexpTokenizer(r"[a-z]+|[.:;!?]")

# Order matters: special cases first, then the generic "n't" rule
CONTRACTIONS = [
    (r"\bwon't\b|\bwont\b", "will not"),
    (r"\bcan't\b|\bcant\b|\bcannot\b", "can not"),
    (r"\bain't\b|\baint\b", "is not"),
    (r"n't\b", " not"),
    (r"\b(do|does|did|is|are|was|were|have|has|had|could|would|should|must|need)nt\b", r"\1 not"),
]


@lru_cache(maxsize=None)
def _lemma(word):
    return lemmatizer.lemmatize(word, pos="v")


def preprocess_text(text):
    """Raw review text -> cleaned, negation-marked tokens (one string).
    Used for BOTH training (notebook 02) and prediction (inference.py / app.py)."""
    text = str(text).lower().replace("\u2019", "'")
    text = re.sub(r"<.*?>", " ", text)            # HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)   # URLs
    for pattern, repl in CONTRACTIONS:            # didn't -> did not
        text = re.sub(pattern, repl, text)
    # commas and contrast words end a negation's scope, like a full stop does
    text = re.sub(r",|\b(?:but|however|although|though)\b", " . ", text)

    tokens = mark_negation(tokenizer.tokenize(text))   # good -> good_NEG after a negation

    out = []
    for tok in tokens:
        neg = tok.endswith("_NEG")
        base = tok[:-4] if neg else tok
        if not base.isalpha() or base in stop_words:
            continue
        out.append(_lemma(base) + ("_NEG" if neg else ""))
    return " ".join(out)