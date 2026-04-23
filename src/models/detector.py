import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_curve, roc_auc_score, average_precision_score,
)
import joblib


NUMERIC_FEATURES = [
    "has_link", "num_attachments", "body_length", "subject_length",
    "num_special_chars", "num_urls", "has_urgency",
    "num_exclamation_marks", "sender_domain_length", "has_ip_in_url",
]

AVAILABLE_MODELS = {
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, class_weight="balanced",
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42,
    ),
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced",
    ),
    "linear_svc": lambda: LinearSVC(
        max_iter=2000, random_state=42, class_weight="balanced",
    ),
}


class PhishingDetector:
    """
    Multi-model phishing email detector using TF-IDF text features
    combined with engineered numeric features.
    """

    def __init__(self, model_name="random_forest"):
        """
        Parameters
        ----------
        model_name : str
            One of: 'random_forest', 'gradient_boosting', 'logistic_regression', 'linear_svc'.
        """
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(AVAILABLE_MODELS.keys())}")

        self.model_name = model_name
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        self.model = AVAILABLE_MODELS[model_name]()
        self._fitted = False

    def prepare_features(self, df, fit=True):
        """
        Build feature matrix from text (TF-IDF) and numeric columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing 'subject', 'body', and numeric feature columns.
        fit : bool
            If True, fit the vectorizer on the data (use for training).
            If False, only transform (use for test/inference).

        Returns
        -------
        X : sparse matrix
            Combined TF-IDF + numeric feature matrix.
        y : pd.Series
            Target labels.
        """
        text_data = df['subject'].fillna('') + " " + df['body'].fillna('')

        if fit:
            X_text = self.vectorizer.fit_transform(text_data)
            self._fitted = True
        else:
            X_text = self.vectorizer.transform(text_data)

        available_numeric = [col for col in NUMERIC_FEATURES if col in df.columns]
        if available_numeric:
            from scipy.sparse import csr_matrix
            X_numeric = csr_matrix(df[available_numeric].fillna(0).values.astype(float))
            X = hstack([X_text, X_numeric])
        else:
            X = X_text

        return X, df['label']

    def train(self, X, y):
        """Train the model on the given feature matrix and labels."""
        if issparse(X):
            X_dense = X
        else:
            X_dense = X
        self.model.fit(X_dense, y)

    def predict(self, X):
        """Generate predictions."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Generate probability predictions (if supported by the model)."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        elif hasattr(self.model, 'decision_function'):
            return self.model.decision_function(X)
        else:
            return self.predict(X).astype(float)

    def evaluate(self, X, y):
        """
        Evaluate the model and return a dictionary of metrics.

        Returns
        -------
        dict
            Dictionary containing classification report, confusion matrix,
            F1 scores, ROC-AUC, and average precision.
        """
        y_pred = self.predict(X)
        y_scores = self.predict_proba(X)

        report = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)
        f1_macro = f1_score(y, y_pred, average='macro')
        f1_phishing = f1_score(y, y_pred, pos_label=1)

        try:
            roc_auc = roc_auc_score(y, y_scores)
        except ValueError:
            roc_auc = None

        try:
            avg_precision = average_precision_score(y, y_scores)
        except ValueError:
            avg_precision = None

        results = {
            "model_name": self.model_name,
            "classification_report": report,
            "confusion_matrix": cm,
            "f1_macro": f1_macro,
            "f1_phishing": f1_phishing,
            "roc_auc": roc_auc,
            "average_precision": avg_precision,
        }

        print(f"\n{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"{'='*60}")
        print("Classification Report:")
        print(classification_report(y, y_pred))
        print(f"Confusion Matrix:\n{cm}")
        print(f"F1 (Macro): {f1_macro:.4f}")
        print(f"F1 (Phishing): {f1_phishing:.4f}")
        if roc_auc is not None:
            print(f"ROC-AUC: {roc_auc:.4f}")
        if avg_precision is not None:
            print(f"Average Precision: {avg_precision:.4f}")

        return results

    def save_model(self, path="results/latest_model.joblib"):
        """Save the trained model and vectorizer to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "vectorizer": self.vectorizer,
            "model_name": self.model_name,
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path="results/latest_model.joblib"):
        """Load a trained model from disk."""
        data = joblib.load(path)
        detector = cls(model_name=data["model_name"])
        detector.model = data["model"]
        detector.vectorizer = data["vectorizer"]
        detector._fitted = True
        return detector


if __name__ == "__main__":
    try:
        df = pd.read_csv("data/raw/phishing_dataset.csv")
        for name in AVAILABLE_MODELS:
            print(f"\nTraining {name}...")
            detector = PhishingDetector(model_name=name)
            X, y = detector.prepare_features(df)
            detector.train(X, y)
            detector.evaluate(X, y)
    except FileNotFoundError:
        print("Dataset not found. Run src/data/generator.py first.")
