# Simulation-Based Imbalanced Phishing Detection

A research framework for evaluating machine learning–based phishing email detection under realistic class-imbalance conditions. The project generates synthetic email datasets, applies multiple imbalance-handling strategies, benchmarks several classifiers, and tests model robustness against adversarial phishing campaigns.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Steps](#pipeline-steps)
- [Imbalance Handling Methods](#imbalance-handling-methods)
- [Models](#models)
- [Adversarial Robustness Testing](#adversarial-robustness-testing)
- [Results](#results)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Overview

Phishing emails represent a small fraction of total email traffic, creating a severe class-imbalance problem for detection systems. This framework addresses the challenge through a simulation-based approach:

1. **Synthetic Data Generation** — Produces realistic phishing and legitimate emails with engineered features (TF-IDF text features, URL counts, urgency indicators, sender domain analysis, etc.).
2. **Imbalance Mitigation** — Implements and compares five strategies: random oversampling, random undersampling, SMOTE, ADASYN, and SMOTE-Tomek Links.
3. **Multi-Model Benchmarking** — Evaluates Random Forest, Gradient Boosting, Logistic Regression, and Linear SVC with metrics tailored for imbalanced classification (F1-phishing, ROC-AUC, Average Precision).
4. **Adversarial Testing** — Generates evasive phishing emails using three strategies (semantic camouflage, domain spoofing, content obfuscation) to assess detector robustness.

## Project Structure

```
simulation-based-imbalanced-phishing-detection/
├── main.py                          # Full experimental pipeline
├── requirements.txt                 # Python dependencies
├── README.md
├── data/
│   ├── raw/                         # Generated raw datasets
│   └── processed/                   # Processed/balanced datasets
├── src/
│   ├── data/
│   │   ├── generator.py             # Synthetic email data generator
│   │   ├── adversarial_gen.py       # Adversarial phishing generator
│   │   └── imbalance_handler.py     # Imbalance handling strategies
│   ├── models/
│   │   └── detector.py              # Multi-model phishing detector
│   └── utils/
│       └── visualization.py         # Plotting and result visualization
├── results/
│   ├── figures/                     # Generated plots and charts
│   ├── best_model.joblib            # Best trained model artifact
│   └── experiment_summary.json      # Experiment metrics summary
└── notebooks/                       # Exploratory analysis notebooks
```

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/simulation-based-imbalanced-phishing-detection.git
cd simulation-based-imbalanced-phishing-detection

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- Key libraries: scikit-learn, imbalanced-learn, pandas, matplotlib, seaborn, faker

## Usage

### Run the Full Pipeline

```bash
python main.py
```

This executes all six steps: data generation → train/test split → imbalance comparison → model comparison → adversarial testing → results export.

### Custom Configuration

```bash
# Generate 10,000 samples with 3% phishing ratio
python main.py --samples 10000 --ratio 0.03

# Use Gradient Boosting for imbalance comparison, ADASYN for model comparison
python main.py --model gradient_boosting --imbalance-method adasyn

# Skip adversarial testing for faster runs
python main.py --skip-adversarial

# Full custom run
python main.py --samples 5000 --ratio 0.05 --model random_forest \
               --imbalance-method smote --adversarial-samples 300 --seed 123
```

### Run Individual Components

```bash
# Generate dataset only
python -m src.data.generator

# Generate adversarial samples
python -m src.data.adversarial_gen

# Test imbalance handlers
python -m src.data.imbalance_handler

# Train and evaluate all models
python -m src.models.detector
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--samples` | `5000` | Total number of email samples to generate |
| `--ratio` | `0.05` | Proportion of phishing emails (minority class) |
| `--model` | `random_forest` | Model used for imbalance method comparison |
| `--imbalance-method` | `smote` | Imbalance strategy used for model comparison |
| `--adversarial-samples` | `200` | Number of adversarial phishing samples |
| `--skip-adversarial` | `false` | Skip adversarial robustness testing |
| `--seed` | `42` | Random seed for reproducibility |

## Pipeline Steps

### Step 1 — Data Generation

The `PhishingDataGenerator` creates synthetic emails with the following features:

| Feature | Type | Description |
|---|---|---|
| `sender` | text | Email sender address |
| `subject` | text | Email subject line |
| `body` | text | Email body content |
| `has_link` | binary | Whether the email contains a URL |
| `num_attachments` | int | Number of file attachments |
| `body_length` | int | Character count of the body |
| `subject_length` | int | Character count of the subject |
| `num_special_chars` | int | Count of special characters |
| `num_urls` | int | Number of URLs in the body |
| `has_urgency` | binary | Presence of urgency-related keywords |
| `num_exclamation_marks` | int | Count of exclamation marks |
| `sender_domain_length` | int | Length of the sender's domain |
| `has_ip_in_url` | binary | Whether any URL contains an IP address |
| `label` | binary | 0 = Legitimate, 1 = Phishing |

### Step 2 — Train/Test Split

Stratified 80/20 split preserving the original class distribution.

### Step 3 — Imbalance Handling Comparison

Trains the same model (default: Random Forest) with different imbalance strategies and compares performance on the held-out test set.

### Step 4 — Multi-Model Comparison

Trains all four classifiers using the selected imbalance method and ranks them by F1-phishing score. The best model is saved to `results/best_model.joblib`.

### Step 5 — Adversarial Robustness Testing

Generates adversarial phishing emails using three evasion strategies and measures detection rates per strategy:

- **Semantic Camouflage** — Professional language avoiding typical phishing keywords
- **Domain Spoofing** — Sender domains resembling legitimate services (e.g., `paypa1-support.com`)
- **Content Obfuscation** — Malicious links embedded within legitimate-looking content

### Step 6 — Results Export

Saves all metrics to `results/experiment_summary.json` and generates comparison charts in `results/figures/`.

## Imbalance Handling Methods

| Method | Type | Description |
|---|---|---|
| `none` | Baseline | No resampling applied |
| `oversample` | Over-sampling | Random duplication of minority class samples |
| `undersample` | Under-sampling | Random removal of majority class samples |
| `smote` | Synthetic | SMOTE — generates synthetic minority samples via interpolation |
| `adasyn` | Synthetic | ADASYN — adaptive synthetic sampling focusing on harder examples |
| `smote_tomek` | Hybrid | SMOTE + Tomek Links — over-samples then cleans boundary noise |

## Models

| Model | Key Hyperparameters |
|---|---|
| Random Forest | 200 trees, max_depth=20, balanced class weights |
| Gradient Boosting | 150 trees, max_depth=5, lr=0.1 |
| Logistic Regression | max_iter=1000, balanced class weights |
| Linear SVC | max_iter=2000, balanced class weights |

### Evaluation Metrics

- **F1 (Macro)** — Balanced F1 across both classes
- **F1 (Phishing)** — F1 score for the minority (phishing) class
- **ROC-AUC** — Area under the ROC curve
- **Average Precision** — Area under the Precision-Recall curve
- **Confusion Matrix** — Per-class prediction breakdown

## Adversarial Robustness Testing

The `AdversarialPhishingGenerator` produces phishing emails designed to evade detection:

```python
from src.data.adversarial_gen import AdversarialPhishingGenerator

gen = AdversarialPhishingGenerator(seed=42)
adv_df = gen.generate_adversarial_dataset(n_samples=200)
```

Results include per-strategy detection rates, enabling targeted analysis of model weaknesses.

## Results

After running the pipeline, results are available in:

- `results/experiment_summary.json` — All metrics in machine-readable format
- `results/figures/` — Visualization charts:
  - `class_distribution.png` — Original dataset imbalance
  - `imbalance_comparison.png` — Method comparison chart
  - `model_comparison.png` — Multi-model benchmark
  - `cm_<model_name>.png` — Confusion matrices per model
  - `adversarial_robustness.png` — Standard vs adversarial performance

## Configuration

All experiments use fixed random seeds (`--seed 42` by default) for full reproducibility. To explore different scenarios:

```bash
# High imbalance scenario (1% phishing)
python main.py --samples 10000 --ratio 0.01

# Moderate imbalance (10% phishing)
python main.py --samples 5000 --ratio 0.10

# Large-scale experiment
python main.py --samples 50000 --ratio 0.05 --adversarial-samples 500
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is intended for academic and research purposes.
