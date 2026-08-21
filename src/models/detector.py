"""
Multi-model phishing email detector with feature engineering and evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from typing import Dict, Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')

# Available models configuration
AVAILABLE_MODELS = {
    'random_forest': {
        'model': RandomForestClassifier,
        'params': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1}
    },
    'logistic_regression': {
        'model': LogisticRegression,
        'params': {'max_iter': 1000, 'random_state': 42, 'class_weight': 'balanced'}
    },
    'gradient_boosting': {
        'model': GradientBoostingClassifier,
        'params': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}
    },
    'svm': {
        'model': SVC,
        'params': {'kernel': 'rbf', 'probability': True, 'random_state': 42, 'class_weight': 'balanced'}
    },
    'decision_tree': {
        'model': DecisionTreeClassifier,
        'params': {'max_depth': 10, 'random_state': 42, 'class_weight': 'balanced'}
    }
}


class PhishingDetector:
    """
    Phishing email detector with feature engineering and multiple model support.
    """

    def __init__(self, model_name: str = 'random_forest', **kwargs):
        """
        Initialize detector with specified model.

        Args:
            model_name: Name of the model to use (must be in AVAILABLE_MODELS)
            **kwargs: Additional parameters for the model
        """
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not available. Choose from: {list(AVAILABLE_MODELS.keys())}")

        self.model_name = model_name
        self.kwargs = kwargs

        # Initialize TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )

        # Initialize scaler for numeric features
        self.scaler = StandardScaler()

        # Numeric feature columns
        self.numeric_features = [
            'length', 'num_links', 'num_attachments', 'num_special_chars',
            'num_urgency_words', 'num_financial_words', 'num_suspicious_words',
            'has_misspellings', 'num_sentences', 'avg_word_length'
        ]

        # Boolean features (convert to int if needed)
        self.boolean_features = ['has_misspellings']

        # Initialize the chosen model
        self.model = self._create_model()
        self.is_fitted = False

        # Store feature names for interpretation
        self.feature_names = None

    def _create_model(self):
        """Create the specified model with parameters."""
        config = AVAILABLE_MODELS[self.model_name]
        model_class = config['model']
        params = config['params'].copy()

        # Update with any custom parameters
        params.update(self.kwargs)

        return model_class(**params)

    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for training or inference.

        Args:
            df: DataFrame with 'email' column and numeric features
            fit: Whether to fit transformers (True for training, False for inference)

        Returns:
            X: Feature matrix
            y: Target vector (if available, else None)

        Raises:
            KeyError: If required columns are missing
        """
        # Validate required columns
        if 'email' not in df.columns:
            raise KeyError("DataFrame must contain 'email' column")

        # Check for numeric features
        missing_numeric = [col for col in self.numeric_features if col not in df.columns]
        if missing_numeric:
            print(f"Warning: Missing numeric features: {missing_numeric}")
            # Try to extract from email if not present
            for feature in missing_numeric:
                if feature == 'length':
                    df[feature] = df['email'].str.len()
                elif feature == 'num_links':
                    df[feature] = df['email'].str.count(r'https?://|www\.')
                elif feature == 'num_special_chars':
                    df[feature] = df['email'].str.count(r'[!@#$%^&*(),.?":{}|<>]')
                elif feature == 'num_sentences':
                    df[feature] = df['email'].str.count(r'[.!?]+')
                else:
                    # Fill with default values
                    df[feature] = 0

        # Extract text
        texts = df['email'].values.astype(str)

        # Extract numeric features with proper handling
        X_numeric = np.zeros((len(df), len(self.numeric_features)), dtype=np.float64)
        for i, feature in enumerate(self.numeric_features):
            if feature in df.columns:
                # Convert boolean to int if needed
                if feature in self.boolean_features:
                    X_numeric[:, i] = df[feature].astype(int).values
                else:
                    X_numeric[:, i] = df[feature].fillna(0).values
            else:
                X_numeric[:, i] = 0

        # Extract target if available
        y = df['label'].values if 'label' in df.columns else None

        # TF-IDF transformation
        try:
            if fit:
                X_text = self.tfidf.fit_transform(texts).toarray()
            else:
                X_text = self.tfidf.transform(texts).toarray()
        except Exception as e:
            print(f"Warning: TF-IDF transformation failed: {e}")
            # Fallback: use empty array
            X_text = np.zeros((len(texts), 0))

        # Scale numeric features
        try:
            if fit:
                X_numeric_scaled = self.scaler.fit_transform(X_numeric)
            else:
                X_numeric_scaled = self.scaler.transform(X_numeric)
        except Exception as e:
            print(f"Warning: Scaling failed: {e}")
            X_numeric_scaled = X_numeric

        # Combine features
        X = np.hstack([X_text, X_numeric_scaled])

        # Store feature names for interpretation
        if fit:
            self.feature_names = (
                    list(self.tfidf.get_feature_names_out()) +
                    self.numeric_features
            )

        return X, y

    def train(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """
        Train the model on prepared features.

        Args:
            X: Feature matrix
            y: Target vector
            sample_weight: Optional sample weights for training
        """
        try:
            if sample_weight is not None:
                self.model.fit(X, y, sample_weight=sample_weight)
            else:
                self.model.fit(X, y)
            self.is_fitted = True
            print(f"✓ Trained {self.model_name} on {len(X)} samples")
            print(f"  Feature dimensions: {X.shape}")
        except Exception as e:
            print(f"Error training model: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for new data."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for new data."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            # For SVM without probability
            decision = self.model.decision_function(X)
            if len(decision.shape) == 1:
                # Binary classification
                proba = np.column_stack([
                    1 / (1 + np.exp(-decision)),
                    np.exp(-decision) / (1 + np.exp(-decision))
                ])
                return proba
            else:
                # Multi-class
                return np.exp(decision) / np.exp(decision).sum(axis=1, keepdims=True)
        else:
            raise ValueError("Model does not support probability prediction")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model performance.

        Returns dictionary with various metrics.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        try:
            y_pred = self.predict(X)
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Fallback: use dummy predictions
            y_pred = np.zeros_like(y)

        try:
            y_prob = self.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else None
        except:
            y_prob = None

        # Calculate metrics
        metrics = {
            'model_name': self.model_name,
        }

        # Calculate all metrics with error handling
        try:
            metrics['accuracy'] = accuracy_score(y, y_pred)
        except:
            metrics['accuracy'] = 0.0

        try:
            metrics['precision_phishing'] = precision_score(y, y_pred, pos_label=1, zero_division=0)
        except:
            metrics['precision_phishing'] = 0.0

        try:
            metrics['recall_phishing'] = recall_score(y, y_pred, pos_label=1, zero_division=0)
        except:
            metrics['recall_phishing'] = 0.0

        try:
            metrics['f1_phishing'] = f1_score(y, y_pred, pos_label=1, zero_division=0)
        except:
            metrics['f1_phishing'] = 0.0

        try:
            metrics['precision_legitimate'] = precision_score(y, y_pred, pos_label=0, zero_division=0)
        except:
            metrics['precision_legitimate'] = 0.0

        try:
            metrics['recall_legitimate'] = recall_score(y, y_pred, pos_label=0, zero_division=0)
        except:
            metrics['recall_legitimate'] = 0.0

        try:
            metrics['f1_legitimate'] = f1_score(y, y_pred, pos_label=0, zero_division=0)
        except:
            metrics['f1_legitimate'] = 0.0

        try:
            metrics['f1_macro'] = f1_score(y, y_pred, average='macro', zero_division=0)
        except:
            metrics['f1_macro'] = 0.0

        try:
            metrics['f1_weighted'] = f1_score(y, y_pred, average='weighted', zero_division=0)
        except:
            metrics['f1_weighted'] = 0.0

        # Add probability-based metrics if available
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y, y_prob)
            except:
                metrics['roc_auc'] = 0.0

            try:
                metrics['average_precision'] = average_precision_score(y, y_prob)
            except:
                metrics['average_precision'] = 0.0

        # Confusion matrix
        try:
            cm = confusion_matrix(y, y_pred)
            metrics['confusion_matrix'] = cm
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = tn
            metrics['false_positives'] = fp
            metrics['false_negatives'] = fn
            metrics['true_positives'] = tp
        except:
            metrics['confusion_matrix'] = np.zeros((2, 2))
            metrics['true_negatives'] = 0
            metrics['false_positives'] = 0
            metrics['false_negatives'] = 0
            metrics['true_positives'] = 0

        # Print results
        print(f"\n{self.model_name} Evaluation:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 (Phishing): {metrics['f1_phishing']:.4f}")
        print(f"  F1 (Legitimate): {metrics['f1_legitimate']:.4f}")
        print(f"  Precision (Phishing): {metrics['precision_phishing']:.4f}")
        print(f"  Recall (Phishing): {metrics['recall_phishing']:.4f}")

        if 'roc_auc' in metrics:
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  Avg Precision: {metrics['average_precision']:.4f}")

        print(f"  Confusion Matrix:")
        print(f"    TP: {metrics['true_positives']}, FP: {metrics['false_positives']}")
        print(f"    FN: {metrics['false_negatives']}, TN: {metrics['true_negatives']}")

        return metrics

    def save_model(self, path: str):
        """Save trained model and transformers."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        pipeline = {
            'model': self.model,
            'tfidf': self.tfidf,
            'scaler': self.scaler,
            'model_name': self.model_name,
            'numeric_features': self.numeric_features,
            'feature_names': self.feature_names,
        }

        joblib.dump(pipeline, path)
        print(f"✓ Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model and transformers."""
        pipeline = joblib.load(path)
        self.model = pipeline['model']
        self.tfidf = pipeline['tfidf']
        self.scaler = pipeline['scaler']
        self.model_name = pipeline['model_name']
        self.numeric_features = pipeline.get('numeric_features', self.numeric_features)
        self.feature_names = pipeline.get('feature_names', None)
        self.is_fitted = True
        print(f"✓ Model loaded from {path}")