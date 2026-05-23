<div align="center">

# ⬡ ReviewIQ
### AI Customer Review Intelligence System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.8-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Deployed-Live-00f5d4?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<p align="center">
  <b>End-to-end classical NLP sentiment analysis system trained on 500K+ Amazon Fine Food Reviews.</b><br/>
  Benchmarks 5 ML models · Deploys best-in-class Linear SVM · Serves real-time predictions via a professional dark-mode web app.
</p>

<p align="center">
  <a href="https://review-iq-ai.streamlit.app">
    <img src="https://img.shields.io/badge/🚀 Live Demo-review--iq--ai.streamlit.app-00f5d4?style=for-the-badge" />
  </a>
</p>

---

</div>

## 📌 Overview

ReviewIQ is a production-grade NLP machine learning system that classifies customer reviews into **Positive**, **Neutral**, or **Negative** sentiment in real time. Built entirely on classical ML — no transformers, no black boxes — it demonstrates rigorous ML engineering from raw data ingestion to live deployment.

> **Why classical ML?** TF-IDF + Linear SVM on a 500K review corpus achieves **84.6% accuracy** with sub-millisecond inference, making it ideal for high-throughput production use cases where interpretability and speed matter.

---

## 🖥️ Live Demo

**[https://review-iq-ai.streamlit.app](https://review-iq-ai.streamlit.app)**

| Page | Description |
|------|-------------|
| **Home** | System overview, inference pipeline, tech stack |
| **Predict** | Real-time sentiment inference with token preview |
| **Dashboard** | Session analytics, sentiment distribution, prediction timeline |
| **Model Comparison** | Full benchmark table with interactive Plotly charts |
| **About** | Project architecture, dataset info, known limitations |

---

## 🏗️ System Architecture

```
ai_review_intelligence/
├── app.py                        # Streamlit deployment app (5-page professional UI)
├── requirements.txt              # Pinned dependencies
├── src/
│   ├── preprocessing.py          # NLP pipeline — cleaning, negation handling, lemmatization
│   ├── feature_engineering.py    # TF-IDF vectorization (unigrams + bigrams, 5k features)
│   ├── train_ml.py               # Model training & hyperparameter tuning
│   └── inference.py              # Production inference pipeline
├── notebooks/
│   ├── 01_data_eda.ipynb         # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # Feature pipeline construction
│   └── 03_ml_models.ipynb        # Model benchmarking & evaluation
├── models/
│   ├── linear_svm_best.pkl       # Deployed model (tuned LinearSVC)
│   └── label_mapping.json        # Class index → label mapping
├── data/
│   └── processed/
│       └── tfidf_vectorizer.pkl  # Fitted TF-IDF vectorizer
└── results/                      # Metrics JSONs, confusion matrices, learning curves
```

---

## 🔬 ML Pipeline

### Phase 1 — Data & Feature Engineering
- Loaded and cleaned the **Amazon Fine Food Reviews** dataset (568,454 reviews)
- Deduplicated, removed HTML tags, stripped noise
- Mapped 5-star ratings → 3 sentiment classes: **Positive** (4–5★), **Neutral** (3★), **Negative** (1–2★)
- Built reusable preprocessing pipeline:
  - Lowercasing & tokenization
  - **Negation preservation** (`not`, `never`, `no` retained as features)
  - WordNet lemmatization (verb-form)
  - Stopword removal with negation exceptions
- Applied **TF-IDF vectorization**: unigrams + bigrams, 5,000 features, stratified 80/20 split

### Phase 2 — Model Benchmarking

| Model | Accuracy | F1 Weighted | F1 Neutral |
|-------|----------|-------------|------------|
| Logistic Regression | 0.7651 | 0.7965 | 0.3293 |
| Logistic Regression (Tuned) | 0.7651 | 0.7963 | 0.3287 |
| Linear SVM | 0.8305 | 0.8362 | 0.3459 |
| **Linear SVM (Tuned) ✅ DEPLOYED** | **0.8463** | **0.8418** | **0.3599** |
| Naive Bayes | 0.8183 | 0.7643 | 0.0247 |
| Naive Bayes (Tuned) | 0.8196 | 0.7672 | 0.0343 |
| Random Forest | 0.7727 | 0.7965 | 0.3293 |
| XGBoost | 0.7917 | 0.7116 | 0.0050 |

**Key finding:** Linear models consistently outperform tree-based models on sparse high-dimensional TF-IDF feature spaces. The Neutral class is the hardest to classify across all models — a known challenge in 3-class sentiment analysis.

Every model was evaluated with: Confusion Matrix · ROC-AUC · Learning Curve · Per-class F1 · Misclassification Analysis.

---

## 🚀 Inference Pipeline

```
Raw Text Input
     ↓
Preprocessing (lowercase → negation handling → lemmatize → stopword removal)
     ↓
TF-IDF Vectorization (unigrams + bigrams · 5k features)
     ↓
Linear SVM Inference (tuned LinearSVC)
     ↓
Sentiment Output: Positive / Negative / Neutral
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| ML Framework | scikit-learn |
| Classifier | LinearSVC (tuned) |
| Boosting | XGBoost |
| NLP | NLTK (lemmatization, stopwords) |
| Vectorization | TF-IDF · ngram(1,2) · 5k features |
| Visualization | Plotly |
| Deployment | Streamlit |
| Hosting | Streamlit Community Cloud |
| Environment | Conda (review_ai) |
| Version Control | Git + GitHub |

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.10+
- Conda (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/yashvardhanshah/AI-Customer-Review-Intelligence-System.git
cd AI-Customer-Review-Intelligence-System

# Create and activate environment
conda create -n review_ai python=3.10
conda activate review_ai

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Run the App

```bash
streamlit run app.py
```

App will open at `http://localhost:8501`

### Run Inference Programmatically

```python
from src.inference import predict

result = predict("Absolutely love this product, would buy again!")
print(result)  # → Positive
```

---

## 📊 Dataset

**Amazon Fine Food Reviews** — sourced from Kaggle

| Property | Value |
|----------|-------|
| Total Reviews | 568,454 |
| After Deduplication | ~393,000 |
| Train Split | 80% (stratified) |
| Test Split | 20% (stratified) |
| Classes | Positive · Neutral · Negative |
| Domain | Food product reviews |

> **Note:** The raw dataset (`Reviews.csv`) is not included in this repository due to size (287MB). Download it from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) and place it at `data/raw/Reviews.csv`.

---

## ⚠️ Known Limitations

- **Neutral class F1 ~0.36** — 3-class sentiment with an ambiguous middle class is inherently hard
- **TF-IDF loses word order** — bag-of-words representation misses sequential context
- **Negation handling is lexical** — `not satisfied` works; complex negations like `I wouldn't say I disliked it` may not
- **Domain-specific** — trained on food reviews; performance may degrade on other domains
- **No deep learning (by design)** — this project focuses on classical NLP fundamentals

---

## 🗺️ Roadmap

- [ ] Phase 3 — FastAPI REST backend + React frontend
- [ ] Transformer-based model (DistilBERT) for comparison
- [ ] Batch prediction endpoint
- [ ] Confidence scores via probability calibration
- [ ] Docker containerization

---

## 📁 Results & Artifacts

All evaluation artifacts are saved in `results/`:
- Per-model metrics JSON
- Confusion matrices (PNG)
- ROC curves (PNG)
- Learning curves (PNG)
- Misclassification analysis reports

---

## 📄 License

This project is licensed under the **MIT License** — see below for details.

```
MIT License

Copyright (c) 2026 Yash Vardhan Shah

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgements

- [Amazon Fine Food Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) — J. McAuley and J. Leskovec, Stanford
- [scikit-learn](https://scikit-learn.org/) — ML framework
- [Streamlit](https://streamlit.io/) — deployment platform
- [NLTK](https://www.nltk.org/) — NLP toolkit

---

<div align="center">

**Built with precision by [Yash Vardhan Shah](https://github.com/yashvardhanshah)**

⭐ Star this repo if you found it useful

</div>