import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

plt.style.use('seaborn-v0_8-whitegrid')
RESULTS_DIR = "results/figures"


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_class_distribution(df, title="Class Distribution", filename="class_distribution.png"):
    """Plot the class distribution of the dataset."""
    ensure_results_dir()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df['label'].value_counts()
    bars = ax.bar(['Legitimate (0)', 'Phishing (1)'], [counts.get(0, 0), counts.get(1, 0)],
                  color=['#2ecc71', '#e74c3c'])
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(int(bar.get_height())), ha='center', fontweight='bold')
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name="Model", filename=None):
    """Plot a confusion matrix heatmap."""
    ensure_results_dir()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=13)
    plt.tight_layout()
    fname = filename or f"cm_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150)
    plt.close()


def plot_model_comparison(results_list, filename="model_comparison.png"):
    """
    Bar chart comparing F1-macro, F1-phishing, ROC-AUC across models.

    Parameters
    ----------
    results_list : list of dict
        Each dict must have keys: model_name, f1_macro, f1_phishing, roc_auc.
    """
    ensure_results_dir()
    models = [r['model_name'] for r in results_list]
    metrics = ['f1_macro', 'f1_phishing', 'roc_auc']
    labels = ['F1 (Macro)', 'F1 (Phishing)', 'ROC-AUC']

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        values = [r.get(metric, 0) or 0 for r in results_list]
        ax.bar(x + i * width, values, width, label=label)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close()


def plot_imbalance_method_comparison(results_dict, filename="imbalance_comparison.png"):
    """
    Compare performance across different imbalance handling methods.

    Parameters
    ----------
    results_dict : dict
        Keys are method names, values are result dicts with f1_phishing, roc_auc.
    """
    ensure_results_dir()
    methods = list(results_dict.keys())
    f1_vals = [results_dict[m].get('f1_phishing', 0) for m in methods]
    auc_vals = [results_dict[m].get('roc_auc', 0) or 0 for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, f1_vals, width, label='F1 (Phishing)', color='#3498db')
    ax.bar(x + width / 2, auc_vals, width, label='ROC-AUC', color='#e67e22')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Imbalance Handling Method Comparison', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close()


def plot_adversarial_robustness(standard_results, adversarial_results, filename="adversarial_robustness.png"):
    """
    Compare model performance on standard vs adversarial test sets.

    Parameters
    ----------
    standard_results : dict
        Results on standard test set.
    adversarial_results : dict
        Results on adversarial test set.
    """
    ensure_results_dir()
    metrics = ['f1_phishing', 'roc_auc']
    labels = ['F1 (Phishing)', 'ROC-AUC']

    std_vals = [standard_results.get(m, 0) or 0 for m in metrics]
    adv_vals = [adversarial_results.get(m, 0) or 0 for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, std_vals, width, label='Standard Test', color='#2ecc71')
    ax.bar(x + width / 2, adv_vals, width, label='Adversarial Test', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title(f"Adversarial Robustness — {standard_results.get('model_name', 'Model')}", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close()
