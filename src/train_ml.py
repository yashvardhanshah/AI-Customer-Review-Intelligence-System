# src/train_ml.py

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

#Logistic Regression
def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model
    """

    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


# Linear SVM
def train_linear_svm(X_train, y_train, C=1.0):
    model = LinearSVC(
        C=C,
        class_weight='balanced',
        max_iter=1000
    )
    model.fit(X_train, y_train)
    return model

# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB

def train_naive_bayes(X_train, y_train, alpha=1.0):
    """
    Train Multinomial Naive Bayes model for text classification
    """

    model = MultinomialNB(alpha=alpha)

    model.fit(X_train, y_train)

    return model

# Random Forest 
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=20,
        max_features='sqrt',
        class_weight='balanced',   # 🔥 ADD THIS
        n_jobs=1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# XGBoost Classifier
from xgboost import XGBClassifier

def train_xgboost(X_train, y_train, n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42):
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, path):
    """
    Save trained model to disk
    """
    joblib.dump(model, path)


def load_model(path):
    """
    Load trained model
    """
    return joblib.load(path)