#!/usr/bin/env python3
"""
Run TRTR, TSTR, and data-augmentation phishing/spam classification experiments.

Expected inputs:
  1) Real dataset: CSV containing a binary label column and an email-text column.
  2) Optional SeqGAN output CSV: synthetic phishing emails generated ONLY from the
     real training partition for the corresponding seed. See leakage note below.

Examples:
  python run_spamassassin_experiments.py \
      --data src/dataset/SpamAssasin.csv \
      --label-col label --text-col text

  python run_spamassassin_experiments.py \
      --data src/dataset/SpamAssasin.csv \
      --synthetic src/dataset/seqgan_phishing.csv \
      --label-col label --text-col text

Label convention:
  Numeric 1 = spam/phishing, 0 = ham/legitimate.
  Text labels such as spam/phishing/junk = positive; ham/legitimate/normal = negative.

Important:
  A synthetic CSV generated once from the FULL real dataset can leak information
  from the held-out test set. For a strict experiment, train SeqGAN separately
  on each seed's REAL TRAINING phishing partition, then generate that seed's
  synthetic phishing CSV and pass it using --synthetic. Do not train SeqGAN on
  validation or test messages.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


POSITIVE = {
    "1", "spam", "phishing", "phish", "junk", "malicious", "positive"
}
NEGATIVE = {
    "0", "ham", "legitimate", "legit", "normal", "benign", "non-spam",
    "nonspam", "negative"
}


def normalize_label(value):
    """Convert common spam/phishing labels to 1 and ham/legitimate to 0."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in POSITIVE:
        return 1
    if s in NEGATIVE:
        return 0
    # Handle labels such as "spam email" or "legitimate email".
    if any(word in s for word in ("phish", "spam", "malicious", "junk")):
        return 1
    if any(word in s for word in ("legitimate", "legit", "ham", "benign", "normal")):
        return 0
    raise ValueError(
        f"Cannot interpret label value {value!r}. "
        "Use 1 for spam/phishing and 0 for ham/legitimate, "
        "or provide recognizable text labels."
    )


def find_column(df, requested, candidates, kind):
    if requested:
        if requested not in df.columns:
            raise ValueError(
                f"{kind} column {requested!r} not found. "
                f"Available columns: {list(df.columns)}"
            )
        return requested

    lower_to_original = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    raise ValueError(
        f"Could not automatically identify the {kind} column. "
        f"Available columns: {list(df.columns)}. "
        f"Pass it explicitly with --{kind}-col."
    )


def load_email_csv(path, label_col=None, text_col=None, synthetic=False):
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    if df.empty:
        raise ValueError(f"No rows found in {path}")

    text_col = find_column(
        df, text_col,
        ["text", "body", "email", "message", "content", "v2", "mail"],
        "text"
    )

    if synthetic and label_col is None:
        labels = pd.Series(np.ones(len(df), dtype=int), index=df.index)
    else:
        label_col = find_column(
            df, label_col,
            ["label", "class", "category", "target", "v1", "type", "is_spam"],
            "label"
        )
        labels = df[label_col].map(normalize_label)

    out = pd.DataFrame({
        "text": df[text_col].fillna("").astype(str).str.strip(),
        "label": labels
    })
    out = out.dropna(subset=["label"])
    out = out[out["text"].str.len() > 0]
    out["label"] = out["label"].astype(int)
    out = out.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    if synthetic:
        # Synthetic input is intended to contain generated phishing/spam only.
        out = out[out["label"] == 1].copy()
        out["label"] = 1

    return out


def clean_text(text):
    """Lightweight identical cleaning for real and synthetic email bodies."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)       # remove HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " EMAIL ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_models(seed, max_features):
    # All models use the same TF-IDF preprocessing pipeline.
    common = dict(
        lowercase=False,
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        ),
        "LinearSVM": LinearSVC(
            class_weight="balanced", random_state=seed
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=seed, n_jobs=-1
        ),
        "MultinomialNB": MultinomialNB(alpha=0.5),
    }

    pipelines = {}
    for name, estimator in models.items():
        pipelines[name] = Pipeline([
            ("tfidf", TfidfVectorizer(**common)),
            ("classifier", estimator),
        ])
    return pipelines


def get_scores(model, texts):
    clf = model.named_steps["classifier"]
    if hasattr(clf, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    return model.decision_function(texts)


def choose_f1_threshold(y_true, scores):
    """Choose threshold on validation data only; never on the test set."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.0
    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )
    best = int(np.nanargmax(f1_values))
    return float(thresholds[best])


def make_training_data(real_train, synthetic, design, seed):
    real_phish = real_train[real_train.label == 1]
    real_ham = real_train[real_train.label == 0]

    if design == "TRTR":
        train = real_train.copy()

    elif design == "TSTR":
        if synthetic is None or synthetic.empty:
            raise ValueError("TSTR requires --synthetic with generated phishing emails.")
        train = pd.concat([synthetic[["text"]].assign(label=1),
                           real_ham[["text", "label"]]], ignore_index=True)

    elif design == "AUGMENTATION":
        if synthetic is None or synthetic.empty:
            raise ValueError("AUGMENTATION requires --synthetic with generated phishing emails.")
        train = pd.concat([
            real_train[["text", "label"]],
            synthetic[["text"]].assign(label=1)
        ], ignore_index=True)

    else:
        raise ValueError(f"Unknown design: {design}")

    train["text"] = train["text"].map(clean_text)
    train = train[train.text.str.len() > 0].drop_duplicates(
        subset=["text", "label"]
    )
    return train.sample(frac=1, random_state=seed).reset_index(drop=True)


def evaluate_one(model, X_val, y_val, X_test, y_test):
    val_scores = get_scores(model, X_val)
    threshold = choose_f1_threshold(y_val, val_scores)

    test_scores = get_scores(model, X_test)
    predictions = (test_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": threshold,
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "auprc": average_precision_score(y_test, test_scores),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run TRTR, TSTR, and augmentation experiments."
    )
    parser.add_argument("--data", required=True, help="Original real email CSV.")
    parser.add_argument("--synthetic", default=None,
                        help="Optional SeqGAN-generated phishing-only CSV.")
    parser.add_argument("--label-col", default=None,
                        help="Label column in real CSV (auto-detected if omitted).")
    parser.add_argument("--text-col", default=None,
                        help="Email text column in real CSV (auto-detected if omitted).")
    parser.add_argument("--synthetic-label-col", default=None,
                        help="Label column in synthetic CSV; omit if it contains only phishing.")
    parser.add_argument("--synthetic-text-col", default=None,
                        help="Text column in synthetic CSV; auto-detected if omitted.")
    parser.add_argument("--seeds", default="42,123,2024,2025,2026")
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.20,
                        help="Fraction of remaining train/validation pool.")
    parser.add_argument("--output-dir", default="results/spamassassin_experiments")
    args = parser.parse_args()

    real = load_email_csv(args.data, args.label_col, args.text_col)
    real["text"] = real["text"].map(clean_text)
    real = real[real.text.str.len() > 0].drop_duplicates(
        subset=["text", "label"]
    ).reset_index(drop=True)

    if real.label.nunique() != 2:
        raise ValueError("The real dataset must contain both classes (0 and 1).")

    synthetic = None
    if args.synthetic:
        synthetic = load_email_csv(
            args.synthetic,
            args.synthetic_label_col,
            args.synthetic_text_col or args.text_col,
            synthetic=True,
        )
        synthetic["text"] = synthetic["text"].map(clean_text)
        synthetic = synthetic[synthetic.text.str.len() > 0].reset_index(drop=True)
        if synthetic.empty:
            raise ValueError("No usable synthetic phishing emails found.")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nDataset summary")
    print("----------------")
    print(f"Usable real emails: {len(real)}")
    print(f"Real legitimate/ham (0): {(real.label == 0).sum()}")
    print(f"Real phishing/spam (1): {(real.label == 1).sum()}")
    if synthetic is not None:
        print(f"Synthetic phishing emails: {len(synthetic)}")
    print("\nNote: thresholds are selected on validation data only.")

    records = []
    designs = ["TRTR"]
    if synthetic is not None:
        designs += ["TSTR", "AUGMENTATION"]

    for seed in seeds:
        # First reserve a real-only held-out test set.
        train_val, test = train_test_split(
            real, test_size=args.test_size, random_state=seed,
            stratify=real.label
        )

        # Split the remaining real data into train and validation.
        train, validation = train_test_split(
            train_val, test_size=args.validation_size, random_state=seed,
            stratify=train_val.label
        )

        X_val = validation.text.tolist()
        y_val = validation.label.to_numpy()
        X_test = test.text.tolist()
        y_test = test.label.to_numpy()

        print(f"\n===== Seed {seed} =====")
        print(f"Train real: {len(train)} | Validation real: {len(validation)} "
              f"| Test real: {len(test)}")
        print(f"Test class counts: ham={(test.label == 0).sum()}, "
              f"spam={(test.label == 1).sum()}")

        # NOTE: The synthetic file must have been generated using this seed's
        # real TRAIN phishing partition only. A single fixed synthetic file
        # is convenient for an initial pilot, but is not a strict seed-wise
        # leakage-safe SeqGAN evaluation.
        for design in designs:
            if design == "TRTR":
                training_data = make_training_data(train, None, design, seed)
            else:
                training_data = make_training_data(train, synthetic, design, seed)

            if training_data.label.nunique() != 2:
                print(f"Skipping {design}: training data has only one class.")
                continue

            X_train = training_data.text.tolist()
            y_train = training_data.label.to_numpy()

            print(f"\n-- {design}: train={len(training_data)}, "
                  f"ham={(training_data.label == 0).sum()}, "
                  f"spam={(training_data.label == 1).sum()}")

            for model_name, model in build_models(seed, args.max_features).items():
                try:
                    model.fit(X_train, y_train)
                    metrics = evaluate_one(
                        model, X_val, y_val, X_test, y_test
                    )
                    row = {
                        "seed": seed,
                        "design": design,
                        "model": model_name,
                        "train_size": len(training_data),
                        "train_ham": int((training_data.label == 0).sum()),
                        "train_spam": int((training_data.label == 1).sum()),
                        "validation_size": len(validation),
                        "test_size": len(test),
                        **metrics,
                    }
                    records.append(row)
                    print(
                        f"{model_name:20s} "
                        f"P={metrics['precision']:.4f} "
                        f"R={metrics['recall']:.4f} "
                        f"F1={metrics['f1']:.4f} "
                        f"AUPRC={metrics['auprc']:.4f} "
                        f"threshold={metrics['threshold']:.4f}"
                    )
                except Exception as exc:
                    print(f"ERROR {design}/{model_name}: {exc}")

    if not records:
        raise RuntimeError("No experiments completed successfully.")

    results = pd.DataFrame(records)
    raw_path = output_dir / "experiment_results_by_seed.csv"
    results.to_csv(raw_path, index=False)

    metric_cols = ["precision", "recall", "f1", "auprc"]
    summary = (
        results.groupby(["design", "model"])[metric_cols]
        .agg(["mean", "std"])
        .round(4)
    )
    summary_path = output_dir / "experiment_summary_mean_std.csv"
    summary.to_csv(summary_path)

    print("\n===== SUMMARY: mean and standard deviation across seeds =====")
    print(summary)
    print(f"\nDetailed results saved to: {raw_path}")
    print(f"Summary saved to: {summary_path}")
    print("\nUse the held-out real test results for Chapter 4. "
          "Do not select thresholds using test labels.")


if __name__ == "__main__":
    main()
