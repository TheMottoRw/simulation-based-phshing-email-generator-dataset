"""
Simulation-Based Imbalanced Phishing Detection — Main Pipeline

This script orchestrates the full experimental pipeline:
1. Synthetic data generation with controlled imbalance
2. Feature engineering (TF-IDF + numeric features)
3. Imbalance handling (oversampling, undersampling, SMOTE, ADASYN, SMOTE-Tomek)
4. Multi-model training and evaluation
5. Adversarial robustness testing
6. Visualization and results export
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.generator import PhishingDataGenerator
from src.data.adversarial_gen import AdversarialPhishingGenerator
from src.data.imbalance_handler import (
    handle_imbalance_oversampling,
    handle_imbalance_undersampling,
    handle_imbalance_smote,
    handle_imbalance_adasyn,
    handle_imbalance_smote_tomek,
)
from src.models.detector import PhishingDetector, AVAILABLE_MODELS
from src.utils.visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_imbalance_method_comparison,
    plot_adversarial_robustness,
)


def generate_data(total_samples, phishing_ratio, seed=42):
    """Step 1: Generate synthetic imbalanced dataset."""
    print("\n" + "=" * 70)
    print("STEP 1: Data Generation")
    print("=" * 70)

    generator = PhishingDataGenerator()
    df = generator.create_imbalanced_dataset(total_samples, phishing_ratio, seed=seed)

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/phishing_dataset.csv", index=False)

    print(f"Generated {len(df)} emails ({df['label'].sum()} phishing, "
          f"{len(df) - df['label'].sum()} legitimate)")
    print(f"Imbalance ratio: {df['label'].mean():.2%} phishing")

    plot_class_distribution(df, title="Original Dataset — Class Distribution")
    return df


def split_data(df, test_size=0.2, seed=42):
    """Step 2: Stratified train/test split."""
    print("\n" + "=" * 70)
    print("STEP 2: Train/Test Split")
    print("=" * 70)

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df['label'],
    )

    print(f"Training set: {len(train_df)} samples "
          f"({train_df['label'].sum()} phishing)")
    print(f"Test set:     {len(test_df)} samples "
          f"({test_df['label'].sum()} phishing)")

    return train_df, test_df


def run_imbalance_experiment(train_df, test_df, model_name="random_forest", methods=None):
    """
    Step 3: Compare imbalance handling methods with a fixed model.

    Returns dict mapping method name -> evaluation results.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Imbalance Handling Comparison")
    print("=" * 70)

    if methods is None:
        methods = ["none", "oversample", "undersample", "smote", "smote_tomek"]

    dataframe_methods = {
        "oversample": handle_imbalance_oversampling,
        "undersample": handle_imbalance_undersampling,
    }
    feature_methods = {
        "smote": handle_imbalance_smote,
        "adasyn": handle_imbalance_adasyn,
        "smote_tomek": handle_imbalance_smote_tomek,
    }

    all_results = {}

    for method in methods:
        print(f"\n--- Method: {method} ---")
        detector = PhishingDetector(model_name=model_name)

        if method == "none":
            X_train, y_train = detector.prepare_features(train_df, fit=True)
        elif method in dataframe_methods:
            balanced_df = dataframe_methods[method](train_df)
            X_train, y_train = detector.prepare_features(balanced_df, fit=True)
        elif method in feature_methods:
            X_train, y_train = detector.prepare_features(train_df, fit=True)
            X_train, y_train = feature_methods[method](X_train, y_train)
        else:
            print(f"Unknown method: {method}, skipping.")
            continue

        detector.train(X_train, y_train)

        X_test, y_test = detector.prepare_features(test_df, fit=False)
        results = detector.evaluate(X_test, y_test)
        all_results[method] = results

    plot_imbalance_method_comparison(all_results)
    return all_results


def run_model_comparison(train_df, test_df, imbalance_method="smote"):
    """
    Step 4: Compare multiple ML models with a fixed imbalance method.

    Returns list of evaluation result dicts.
    """
    print("\n" + "=" * 70)
    print("STEP 4: Multi-Model Comparison")
    print("=" * 70)

    feature_methods = {
        "smote": handle_imbalance_smote,
        "adasyn": handle_imbalance_adasyn,
        "smote_tomek": handle_imbalance_smote_tomek,
    }

    all_results = []
    best_detector = None
    best_f1 = -1

    for model_name in AVAILABLE_MODELS:
        print(f"\n--- Model: {model_name} ---")
        detector = PhishingDetector(model_name=model_name)
        X_train, y_train = detector.prepare_features(train_df, fit=True)

        if imbalance_method in feature_methods:
            X_train, y_train = feature_methods[imbalance_method](X_train, y_train)

        detector.train(X_train, y_train)

        X_test, y_test = detector.prepare_features(test_df, fit=False)
        results = detector.evaluate(X_test, y_test)
        all_results.append(results)

        y_pred = detector.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, model_name=model_name)

        if results['f1_phishing'] > best_f1:
            best_f1 = results['f1_phishing']
            best_detector = detector

    plot_model_comparison(all_results)

    if best_detector:
        best_detector.save_model("results/best_model.joblib")
        print(f"\nBest model: {best_detector.model_name} (F1-phishing: {best_f1:.4f})")

    return all_results, best_detector


def run_adversarial_test(best_detector, test_df, n_adversarial=200, seed=42):
    """
    Step 5: Test model robustness against adversarial phishing emails.
    """
    print("\n" + "=" * 70)
    print("STEP 5: Adversarial Robustness Testing")
    print("=" * 70)

    adv_gen = AdversarialPhishingGenerator(seed=seed)
    adv_df = adv_gen.generate_adversarial_dataset(n_samples=n_adversarial, seed=seed)

    # Drop the adversarial_strategy column for feature preparation
    adv_eval_df = adv_df.drop(columns=['adversarial_strategy'], errors='ignore')

    # Standard test evaluation
    X_test, y_test = best_detector.prepare_features(test_df, fit=False)
    standard_results = best_detector.evaluate(X_test, y_test)

    # Adversarial evaluation
    X_adv, y_adv = best_detector.prepare_features(adv_eval_df, fit=False)
    print("\nAdversarial Test Results:")
    adversarial_results = best_detector.evaluate(X_adv, y_adv)

    # Per-strategy breakdown
    print("\n--- Per-Strategy Breakdown ---")
    for strategy in adv_df['adversarial_strategy'].unique():
        mask = adv_df['adversarial_strategy'] == strategy
        strategy_df = adv_eval_df[mask.values]
        X_s, y_s = best_detector.prepare_features(strategy_df, fit=False)
        y_pred = best_detector.predict(X_s)
        detection_rate = (y_pred == 1).mean()
        print(f"  {strategy}: detection rate = {detection_rate:.2%} ({(y_pred == 1).sum()}/{len(y_pred)})")

    plot_adversarial_robustness(standard_results, adversarial_results)
    return standard_results, adversarial_results


def save_summary(imbalance_results, model_results, adversarial_results=None):
    """Step 6: Save experiment summary to JSON."""
    print("\n" + "=" * 70)
    print("STEP 6: Saving Results Summary")
    print("=" * 70)

    os.makedirs("results", exist_ok=True)

    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    summary = {
        "imbalance_comparison": {
            method: {
                "f1_macro": r.get("f1_macro"),
                "f1_phishing": r.get("f1_phishing"),
                "roc_auc": r.get("roc_auc"),
                "average_precision": r.get("average_precision"),
            }
            for method, r in imbalance_results.items()
        },
        "model_comparison": [
            {
                "model_name": r.get("model_name"),
                "f1_macro": r.get("f1_macro"),
                "f1_phishing": r.get("f1_phishing"),
                "roc_auc": r.get("roc_auc"),
                "average_precision": r.get("average_precision"),
            }
            for r in model_results
        ],
    }

    if adversarial_results:
        std_r, adv_r = adversarial_results
        summary["adversarial_robustness"] = {
            "standard_test": {
                "f1_phishing": std_r.get("f1_phishing"),
                "roc_auc": std_r.get("roc_auc"),
            },
            "adversarial_test": {
                "f1_phishing": adv_r.get("f1_phishing"),
                "roc_auc": adv_r.get("roc_auc"),
            },
        }

    summary = make_serializable(summary)

    with open("results/experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Results saved to results/experiment_summary.json")
    print("Figures saved to results/figures/")


def main():
    parser = argparse.ArgumentParser(
        description="Simulation-Based Imbalanced Phishing Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --samples 5000 --ratio 0.05
  python main.py --samples 3000 --ratio 0.1 --model random_forest
  python main.py --samples 5000 --ratio 0.05 --skip-adversarial
        """,
    )
    parser.add_argument("--samples", type=int, default=5000,
                        help="Total samples to generate (default: 5000)")
    parser.add_argument("--ratio", type=float, default=0.05,
                        help="Phishing ratio in dataset (default: 0.05)")
    parser.add_argument("--model", choices=list(AVAILABLE_MODELS.keys()),
                        default="random_forest",
                        help="Model for imbalance comparison (default: random_forest)")
    parser.add_argument("--imbalance-method", choices=["none", "oversample", "undersample",
                                                        "smote", "adasyn", "smote_tomek"],
                        default="smote",
                        help="Imbalance method for model comparison (default: smote)")
    parser.add_argument("--adversarial-samples", type=int, default=200,
                        help="Number of adversarial samples (default: 200)")
    parser.add_argument("--skip-adversarial", action="store_true",
                        help="Skip adversarial robustness testing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    print("=" * 70)
    print("  Simulation-Based Imbalanced Phishing Detection")
    print("=" * 70)
    print(f"  Samples: {args.samples} | Phishing ratio: {args.ratio}")
    print(f"  Imbalance method: {args.imbalance_method} | Seed: {args.seed}")
    print("=" * 70)

    # Step 1: Generate data
    df = generate_data(args.samples, args.ratio, seed=args.seed)

    # Step 2: Split data
    train_df, test_df = split_data(df, seed=args.seed)

    # Step 3: Imbalance handling comparison
    imbalance_results = run_imbalance_experiment(
        train_df, test_df, model_name=args.model,
    )

    # Step 4: Multi-model comparison
    model_results, best_detector = run_model_comparison(
        train_df, test_df, imbalance_method=args.imbalance_method,
    )

    # Step 5: Adversarial robustness
    adversarial_out = None
    if not args.skip_adversarial and best_detector:
        adversarial_out = run_adversarial_test(
            best_detector, test_df,
            n_adversarial=args.adversarial_samples, seed=args.seed,
        )

    # Step 6: Save summary
    save_summary(imbalance_results, model_results, adversarial_out)

    print("\n" + "=" * 70)
    print("  Pipeline Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
