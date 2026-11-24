<div align="center">

# ⬡ ReviewIQ
### AI Customer Review Intelligence System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.7.2-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.54.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Deployed-Live-00f5d4?style=for-the-badge" />
</p>

<p align="center">
  <b>Classical NLP sentiment analysis on Amazon Fine Food Reviews.</b><br/>
  Benchmarks 5 ML models on TF-IDF features, deploys a tuned Linear SVM, and serves predictions in a dark-mode Streamlit app.
</p>

<p align="center">
  <a href="https://review-iq-ai.streamlit.app">
    <img src="https://img.shields.io/badge/🚀 Live Demo-review--iq--ai.streamlit.app-00f5d4?style=for-the-badge" />
  </a>
</p>

---

</div>

## 📌 Overview

ReviewIQ classifies a customer review as **Positive**, **Neutral** or **Negative**. It uses no transformers: text is cleaned, negation is handled explicitly, and a linear SVM is trained on TF-IDF features. A linear model on sparse text features is small and inspectable, so the app can show which terms drove each prediction.

The deployed model scores **86.3% accuracy and 0.68 macro F1** on a held-out test set of 31,487 reviews. Accuracy needs context here: 78% of the reviews are Positive, so a model that always answers "Positive" already scores 77.9%. Macro F1, which weights the three classes equally, is the fairer yardstick (0.68 against 0.29 for that baseline).

---

## 🖥️ Live Demo

**[https://review-iq-ai.streamlit.app](https://review-iq-ai.streamlit.app)**

| Page | Description |
|------|-------------|
| **Home** | System overview, inference pipeline, tech stack |
| **Predict** | Real-time prediction, the processed tokens, and the terms that drove the result |
| **Dashboard** | Session analytics, sentiment distribution, prediction timeline |
| **Model Comparison** | Full benchmark table, charts, and the highest-weight terms per class |
| **About** | Dataset funnel, architecture, known limitations |

The Home, Dashboard and Model Comparison numbers are read from `results/metrics/model_comparison.csv`, so the app always matches the notebooks.

---

## 📊 Results

Evaluated on the same held-out test set (31,487 reviews: 24,541 Positive, 4,566 Negative, 2,380 Neutral). Source: `results/metrics/model_comparison.csv`.

| Model | Accuracy | F1 (Macro) | F1 (Weighted) | F1 (Neutral) | ROC-AUC (Macro) |
|-------|----------|------------|---------------|--------------|-----------------|
| Logistic Regression | 0.8097 | 0.6516 | 0.8308 | 0.3679 | 0.9063 |
| Logistic Regression (Tuned) | 0.8125 | 0.6434 | 0.8305 | 0.3508 | 0.8948 |
| Linear SVM | 0.8467 | 0.6589 | 0.8503 | 0.3550 | 0.8948 |
| **Linear SVM (Tuned), deployed** | **0.8629** | **0.6825** | **0.8623** | **0.3959** | **0.9130** |
| Naive Bayes | 0.8428 | 0.5288 | 0.8037 | 0.0828 | 0.9025 |
| Naive Bayes (Tuned) | 0.8491 | 0.5641 | 0.8174 | 0.1530 | 0.9042 |
| Random Forest | 0.7786 | 0.5992 | 0.7953 | 0.3133 | 0.8557 |
| XGBoost | 0.7512 | 0.5982 | 0.7817 | 0.3309 | 0.8763 |
| *Always predict Positive* | *0.7794* | *0.2920* | *0.6828* | *0.0000* | *0.5000* |

Tuned models were selected by 5-fold cross-validated weighted F1 on the training data: Logistic Regression 0.8321, Linear SVM 0.8628, Naive Bayes 0.8170.

**What the results show**
- The tuned Linear SVM is best on every column, and it also has the best cross-validated score.
- Naive Bayes looks strong on accuracy but almost never predicts Neutral (Neutral F1 0.08 to 0.15), which is why macro F1 is the better comparison.
- Random Forest and XGBoost trained on the full training set with class balancing, yet trail the SVM by about 8 points of macro F1. Neither was cross-validation-tuned, and XGBoost used memory-limited settings (depth 5, 32 histogram bins, 30% column sampling), so this is a comparison under these settings, not a ceiling for tree models.
- **Error profile of the deployed model:** 4,317 of 31,487 test reviews (13.7%) are misclassified. About 65% of those involve a 3-star Neutral review in one direction or the other (e.g. Neutral predicted as Positive: 876). The remaining 1,521 are direct Positive/Negative swaps.

---

## 🔬 How It Works

### Data
Amazon Fine Food Reviews (Kaggle): 568,454 reviews, reduced to 393,579 after removing duplicate texts, then a stratified 40% sample of 157,432 reviews for compute reasons. Star ratings map to classes: 4-5★ Positive, 3★ Neutral, 1-2★ Negative. The sample is split 80/20 (stratified): **125,945 train / 31,487 test**.

### Preprocessing (`src/preprocessing.py`)
One function, `preprocess_text`, is used by the training notebooks, `inference.py` and the app, so training and prediction can never drift apart:
1. Lowercase; remove HTML tags and URLs.
2. Expand negative contractions: `didn't`, `didnt`, `can't`, `won't`, `ain't`... become `not`.
3. End negation scope at commas and contrast words (`but`, `however`, `although`, `though`).
4. Tokenize with NLTK's `RegexpTokenizer`, keeping sentence punctuation.
5. Mark negation with NLTK's `mark_negation`: words after a negation get a `_NEG` suffix (`not good` → `not good_NEG`), so "good" and "good_NEG" become different features.
6. Remove stopwords except negations, and lemmatize (verb form).

### Features and model
- **TF-IDF:** unigrams and bigrams, `min_df=5`, `max_df=0.9`, 20,000 features. The vectorizer is fit on training data only.
- **Classifier:** `LinearSVC`, `class_weight='balanced'`, `C=0.1` chosen by 5-fold `GridSearchCV` over C in {0.1, 1, 5}.
- **Other models:** Logistic Regression and Naive Bayes (grid-searched), Random Forest (80 trees, depth 20, balanced) and XGBoost (200 trees, balanced sample weights).

### Evaluation notes
- The split happens before preprocessing and the vectorizer never sees test data.
- Hyperparameters were tuned by cross-validation on the training set only. The models were then compared on the same test set, and there is no separate validation set, so the model choice has a small optimistic bias.
- Every number in `results/metrics/` is produced by one final cell in notebook 03 that re-evaluates each saved model from its own predictions.

---

## ⚠️ Known Limitations

- **Neutral is hard:** the deployed model's Neutral F1 is about 0.40. A 3-star review often mixes praise and complaint.
- **Negation is rule-based:** double negation (`not not happy`) and sarcasm still fail.
- **Unseen words are ignored:** a word outside the 20,000-feature vocabulary (e.g. `okayish`) contributes nothing.
- **TF-IDF discards word order** beyond bigrams.
- **Domain:** trained on food reviews and on a 157K sample of the de-duplicated data.
- **Tree models** were not cross-validation-tuned.
- Positive reviews are 78% of the data, so rare-class performance matters more than accuracy.

---

## 📁 Repository Layout

```
├── app.py                      # Streamlit app (5 pages)
├── requirements.txt            # pinned runtime dependencies for the app
├── src/
│   ├── preprocessing.py        # shared text pipeline (training + app)
│   ├── feature_engineering.py  # TF-IDF vectorizer definition
│   ├── train_ml.py             # model training helpers
│   └── inference.py            # predict(text)
├── notebooks/
│   ├── 01_data_eda.ipynb       # loading, cleaning, labels, sampling
│   ├── 02_feature_engineering.ipynb   # split, preprocessing, TF-IDF
│   └── 03_ml_models.ipynb      # training, tuning, evaluation (final cell writes results/)
├── models/                     # linear_svm_best.pkl (deployed), label_mapping.json
├── data/processed/tfidf_vectorizer.pkl
└── results/
    ├── metrics/                # model_comparison.csv, per-model metrics and reports
    └── figures/                # confusion matrices, ROC and learning curves
```

Other trained models, the raw data and the intermediate matrices are not tracked (they are large); the notebooks regenerate them.

---

## ⚙️ Run Locally

```bash
git clone https://github.com/yashvardhanshah/AI-Customer-Review-Intelligence-System.git
cd AI-Customer-Review-Intelligence-System

conda create -n review_ai python=3.10
conda activate review_ai
pip install -r requirements.txt

streamlit run app.py          # NLTK data is downloaded automatically on first run
```

Use the model directly:

```python
from src.inference import predict

print(predict("I didn't like it at all!"))   # Negative
```

**Reproducing the training** (notebooks) needs a few extra packages and the dataset:

```bash
pip install xgboost seaborn matplotlib jupyter
```

Download `Reviews.csv` from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews), place it at `data/raw/Reviews.csv`, then run notebooks 01, 02 and 03 in order. Notebook 03 trains every model and its last cell rewrites `results/`.

---

## 🗺️ Roadmap

- [ ] Cross-validation-tune Random Forest and XGBoost and add a separate validation split
- [ ] Error analysis on the saved misclassified reviews (negation, sarcasm, mixed reviews)
- [ ] Probability calibration for confidence scores
- [ ] FastAPI backend, and a transformer baseline (DistilBERT) for comparison

---

## 🙏 Acknowledgements

- [Amazon Fine Food Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews): J. McAuley and J. Leskovec, Stanford
- [scikit-learn](https://scikit-learn.org/), [NLTK](https://www.nltk.org/), [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/)

---

<div align="center">

**Built by [Yashvardhan Shah](https://github.com/yashvardhanshah)**

</div>