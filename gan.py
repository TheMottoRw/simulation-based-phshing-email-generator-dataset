"""
GAN-based Phishing Email Generator

This script trains a GAN on real phishing emails and generates synthetic phishing emails
to create a balanced dataset for training.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    classification_report
)

warnings.filterwarnings('ignore')

# -----------------------------------
# CONFIGURATION
# -----------------------------------
timestamp = str(datetime.now()).replace(" ", "").replace("-", "").replace(":", "")[:16]
CONFIG = {
    'dataset_path': 'src/dataset/SpamAssasin.csv',
    'output_path': 'src/dataset/generated_phishing_dataset_' + timestamp + '.csv',
    'max_features': 50,
    'latent_dim': 16,
    'epochs': 200,
    'batch_size': 8,
    'n_generated': 500,
    'seed': 42,
    "phishing_to_legitimate_ratio": (1, 5),
    "target_dataset_size": 10000,

}

# Set random seeds for reproducibility
np.random.seed(CONFIG['seed'])

# -----------------------------------
# LOAD DATASET
# -----------------------------------

print("=" * 70)
print("GAN Phishing Email Generator")
print("=" * 70)

try:
    dataset = pd.read_csv(CONFIG['dataset_path'])
    print(f"✓ Loaded dataset: {len(dataset)} emails")
except FileNotFoundError:
    print(f"✗ Dataset not found at {CONFIG['dataset_path']}")
    print("Creating a small sample dataset for testing...")

    # Create sample data if file doesn't exist
    sample_data = {
        'body': [
            "Your account has been compromised! Click here to verify.",
            "Dear customer, please update your payment information.",
            "You've won a prize! Claim now.",
            "Security alert: unusual login detected.",
            "Your subscription will expire soon. Renew now.",
            "Important: verify your account details.",
            "Congratulations! You're eligible for a special offer.",
            "Please confirm your identity to avoid account suspension.",
            "Your order has been processed. Track it here.",
            "Thank you for your purchase. Your receipt is attached."
        ],
        'label': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 5 phishing, 5 legitimate
    }
    dataset = pd.DataFrame(sample_data)
    os.makedirs("src/dataset", exist_ok=True)
    dataset.to_csv(CONFIG['dataset_path'], index=False)
    print(f"✓ Created sample dataset with {len(dataset)} emails")

# -----------------------------------
# CLEAN DATA
# -----------------------------------

print("\nCleaning data...")
dataset = dataset.dropna(subset=["body"])
dataset["body"] = dataset["body"].astype(str)
dataset = dataset[dataset["body"].str.strip() != ""]
dataset = dataset.reset_index(drop=True)

print(f"✓ Cleaned dataset: {len(dataset)} emails")

# -----------------------------------
# FILTER DATA
# -----------------------------------

phishing_data = dataset[dataset["label"] == 1]["body"]
legitimate_data = dataset[dataset["label"] == 0]["body"]

print(f"\nPhishing emails: {len(phishing_data)}")
print(f"Legitimate emails: {len(legitimate_data)}")

if len(phishing_data) < 10:
    print(f"⚠️  Warning: Only {len(phishing_data)} phishing emails found.")
    print("   Training GAN may not work well with so few samples.")
    print("   Consider adding more phishing emails to the dataset.")

# -----------------------------------
# TF-IDF FEATURE EXTRACTION
# -----------------------------------

print("\nExtracting TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=CONFIG['max_features'],
    stop_words='english',
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_df=0.95  # Ignore terms that appear in more than 95% of documents
)

try:
    X_phishing = vectorizer.fit_transform(phishing_data).toarray()
    print(f"✓ TF-IDF Shape: {X_phishing.shape}")
    print(f"✓ Feature names: {vectorizer.get_feature_names_out()}")
except ValueError as e:
    print(f"✗ TF-IDF failed: {e}")
    print("   Using fallback: creating simple feature vectors...")
    # Fallback: create simple binary features
    words = ["account", "click", "verify", "update", "prize", "claim", "security", "urgent"]
    X_phishing = np.zeros((len(phishing_data), len(words)))
    for i, email in enumerate(phishing_data):
        for j, word in enumerate(words):
            if word in email.lower():
                X_phishing[i, j] = 1
    vectorizer = None

if len(X_phishing) == 0:
    print("✗ No phishing data available. Exiting.")
    exit(1)

# -----------------------------------
# MODEL PARAMETERS
# -----------------------------------

input_dim = X_phishing.shape[1]
latent_dim = CONFIG['latent_dim']

print(f"\nModel Parameters:")
print(f"  Input dimension: {input_dim}")
print(f"  Latent dimension: {latent_dim}")

# -----------------------------------
# BUILD GENERATOR
# -----------------------------------

print("\nBuilding Generator...")
generator = Sequential([
    Dense(64, activation='relu', input_dim=latent_dim),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(input_dim, activation='sigmoid')
])

generator.summary()

# -----------------------------------
# BUILD DISCRIMINATOR
# -----------------------------------

print("\nBuilding Discriminator...")
discriminator = Sequential([
    Dense(128, activation='relu', input_dim=input_dim),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

discriminator.compile(
    optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

discriminator.summary()

# -----------------------------------
# BUILD GAN
# -----------------------------------

print("\nBuilding GAN...")
discriminator.trainable = False

gan = Sequential([
    generator,
    discriminator
])

gan.compile(
    optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy'
)

# -----------------------------------
# TRAINING
# -----------------------------------

print("\n" + "=" * 70)
print("Training GAN...")
print("=" * 70)

epochs = CONFIG['epochs']
batch_size = min(CONFIG['batch_size'], len(X_phishing) // 2)

print(f"Epochs: {epochs}")
print(f"Batch size: {batch_size}")
print(f"Total phishing samples: {len(X_phishing)}")

# Track losses for monitoring
d_loss_real_history = []
d_loss_fake_history = []
g_loss_history = []

for epoch in range(epochs):
    try:
        # Select random real emails
        idx = np.random.randint(0, X_phishing.shape[0], batch_size)
        real_emails = X_phishing[idx]

        # Generate fake emails
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_emails = generator.predict(noise, verbose=0)

        # Labels
        real_labels = np.ones((batch_size, 1)) * (1 - 0.1 * np.random.random())  # Label smoothing
        fake_labels = np.zeros((batch_size, 1)) + 0.1 * np.random.random()  # Label smoothing

        # Train Discriminator
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real_emails, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_emails, fake_labels)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        misleading_labels = np.ones((batch_size, 1))

        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, misleading_labels)

        # Store losses
        d_loss_real_history.append(d_loss_real[0])
        d_loss_fake_history.append(d_loss_fake[0])
        g_loss_history.append(g_loss)

        # Print progress
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"  D Loss Real: {d_loss_real[0]:.4f} (Acc: {d_loss_real[1]:.4f})")
            print(f"  D Loss Fake: {d_loss_fake[0]:.4f} (Acc: {d_loss_fake[1]:.4f})")
            print(f"  G Loss: {g_loss:.4f}")

    except Exception as e:
        print(f"Error in epoch {epoch}: {e}")
        continue

# Create results directory if it doesn't exist
os.makedirs("src/results/figures", exist_ok=True)

plt.figure(figsize=(10, 6))

plt.plot(d_loss_real_history, label="Discriminator Loss (Real)")
plt.plot(d_loss_fake_history, label="Discriminator Loss (Fake)")
plt.plot(g_loss_history, label="Generator Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN Training Loss")

plt.legend()
plt.grid(True)

# Save the figure
plt.savefig(
    "src/results/figures/gan_training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
print("\n✓ Training complete!")

# -----------------------------------
# GENERATE SYNTHETIC EMAILS
# -----------------------------------

print("\n" + "=" * 70)
print("Generating Synthetic Emails...")
print("=" * 70)

num_generated_samples = CONFIG['n_generated']

# Generate vectors
noise = np.random.normal(0, 1, (num_generated_samples, latent_dim))
generated_vectors = generator.predict(noise, verbose=0)

# Convert vectors back to email text
synthetic_emails = []

if vectorizer is not None:
    feature_names = vectorizer.get_feature_names_out()

    for vector in generated_vectors:
        # Get top N words
        n_words = min(10, len(feature_names))
        top_indices = vector.argsort()[-n_words:]

        # Build sentence from top words
        words = [feature_names[i] for i in top_indices if vector[i] > 0.1]
        if words:
            sentence = " ".join(words)
        else:
            # If no words above threshold, use top words regardless
            words = [feature_names[i] for i in top_indices]
            sentence = " ".join(words)

        # Make it look more like an email
        if len(sentence) > 0:
            sentence = sentence.capitalize() + "."
        else:
            sentence = "Your account requires verification. Click here to confirm your identity."

        synthetic_emails.append(sentence)
else:
    # Fallback if vectorizer failed
    templates = [
        "Your account has been compromised. Click here to verify.",
        "Important: Please update your payment information.",
        "Security alert: Unusual login detected on your account.",
        "Your subscription will expire soon. Renew now to continue service.",
        "Congratulations! You have won a special prize. Claim it now.",
        "Verify your identity to avoid account suspension.",
        "Your order has been shipped. Track it here.",
        "Please confirm your email address to activate your account.",
        "Your account has been flagged for suspicious activity.",
        "We need you to confirm your recent purchase."
    ]

    for i in range(num_generated_samples):
        synthetic_emails.append(np.random.choice(templates))

print(f"✓ Generated {len(synthetic_emails)} synthetic phishing emails")

# -----------------------------------
# CREATE DATAFRAMES
# -----------------------------------

print("\nCreating final dataset...")

# Generated phishing emails
generated_df = pd.DataFrame({
    "body": synthetic_emails,
    "label": 1
})

# Real legitimate emails (use all if available, or sample)
if len(legitimate_data) > 0:
    # -----------------------------------
    # CONTROL CLASS IMBALANCE RATIO
    # -----------------------------------

    phishing_ratio, legitimate_ratio = CONFIG[
        "phishing_to_legitimate_ratio"
    ]
    # ============================================================
    # TARGET DATASET SIZE CONFIGURATION
    # ============================================================

    target_dataset_size = CONFIG["target_dataset_size"]

    print("\n" + "=" * 60)
    print("TARGET DATASET CONFIGURATION")
    print("=" * 60)

    print(f"Target dataset size: {target_dataset_size}")
    print(f"Existing legitimate emails: {len(legitimate_data)}")

    # ============================================================
    # CALCULATE REQUIRED GENERATED PHISHING SAMPLES
    # ============================================================

    # Keep all existing legitimate emails
    n_legitimate = len(legitimate_data)

    # Calculate the number of rows still needed
    num_generated_samples = target_dataset_size - n_legitimate

    # Validate target size
    if num_generated_samples <= 0:
        raise ValueError(
            f"Target dataset size ({target_dataset_size}) must be "
            f"greater than the number of existing legitimate emails "
            f"({n_legitimate})."
        )

    print(f"Synthetic phishing emails required: {num_generated_samples}")

    # ============================================================
    # GENERATE SYNTHETIC PHISHING EMAIL VECTORS
    # ============================================================

    print("\nGenerating synthetic phishing emails...")

    noise = np.random.normal(
        0,
        1,
        (num_generated_samples, latent_dim)
    )

    generated_vectors = generator.predict(
        noise,
        batch_size=32,
        verbose=1
    )

    # ============================================================
    # CONVERT GENERATED VECTORS TO TEXT
    # ============================================================

    feature_names = vectorizer.get_feature_names_out()

    synthetic_emails = []

    for vector in generated_vectors:
        # Select the most important words
        top_indices = vector.argsort()[-10:]

        # Sort words by their generated score
        top_indices = top_indices[
            np.argsort(vector[top_indices])[::-1]
        ]

        words = [
            feature_names[i]
            for i in top_indices
        ]

        # Create synthetic email text
        synthetic_email = " ".join(words)

        synthetic_emails.append(synthetic_email)

    # ============================================================
    # CREATE SYNTHETIC PHISHING DATAFRAME
    # ============================================================

    generated_df = pd.DataFrame({
        "body": synthetic_emails,
        "label": 1
    })

    print(f"\nGenerated phishing emails: {len(generated_df)}")

    # ============================================================
    # CREATE LEGITIMATE DATAFRAME FROM EXISTING DATASET
    # ============================================================

    # Keep all existing legitimate emails
    legitimate_df = pd.DataFrame({
        "body": legitimate_data.astype(str).values,
        "label": 0
    })

    print(f"Existing legitimate emails used: {len(legitimate_df)}")

    # ============================================================
    # COMBINE EXISTING AND GENERATED DATA
    # ============================================================

    final_dataset = pd.concat(
        [
            legitimate_df,
            generated_df
        ],
        ignore_index=True
    )

    # ============================================================
    # SHUFFLE FINAL DATASET
    # ============================================================

    final_dataset = final_dataset.sample(
        frac=1,
        random_state=CONFIG["seed"]
    ).reset_index(drop=True)

    # ============================================================
    # VERIFY TARGET DATASET SIZE
    # ============================================================

    if len(final_dataset) != target_dataset_size:
        raise ValueError(
            f"Dataset size error: expected {target_dataset_size}, "
            f"but got {len(final_dataset)}"
        )

    # ============================================================
    # DISPLAY FINAL DATASET INFORMATION
    # ============================================================

    print("\n" + "=" * 60)
    print("FINAL GENERATED DATASET")
    print("=" * 60)

    print(f"Total rows: {len(final_dataset)}")

    print("\nClass distribution:")
    print(
        final_dataset["label"].value_counts()
    )

    print("\nClass distribution percentage:")
    print(
        final_dataset["label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nFirst five shuffled rows:")
    print(final_dataset.head())

    # ============================================================
    # EXPORT FINAL DATASET
    # ============================================================

    final_dataset.to_csv(
        CONFIG["output_path"],
        index=False
    )

    print("\n" + "=" * 60)
    print("DATASET EXPORTED SUCCESSFULLY")
    print("=" * 60)

    print(f"Output file: {CONFIG['output_path']}")
    print(f"Total rows exported: {len(final_dataset)}")
# -----------------------------------
# MACHINE LEARNING MODEL PERFORMANCE EVALUATION
# ============================================================

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------------------
# CREATE RESULTS DIRECTORY
# ------------------------------------------------------------

os.makedirs("src/results", exist_ok=True)

# ============================================================
# STEP 1: PREPARE ORIGINAL DATASET
# ============================================================

# Ensure email body is text
dataset["body"] = dataset["body"].fillna("").astype(str)

# Ensure labels are numeric
dataset["label"] = pd.to_numeric(
    dataset["label"],
    errors="coerce"
)

# Keep only valid labels: 0 = legitimate, 1 = phishing
dataset = dataset[
    dataset["label"].isin([0, 1])
].copy()

dataset["label"] = dataset["label"].astype(int)

# Remove empty email bodies
dataset = dataset[
    dataset["body"].str.strip() != ""
    ].copy()

print("\n" + "=" * 70)
print("ORIGINAL DATASET")
print("=" * 70)

print(f"Total emails: {len(dataset)}")
print(f"Phishing emails: {(dataset['label'] == 1).sum()}")
print(f"Legitimate emails: {(dataset['label'] == 0).sum()}")

# ============================================================
# STEP 2: SPLIT ORIGINAL DATA BEFORE EVALUATION
# ============================================================

# This test set remains untouched.
# Do NOT add synthetic data to this test set.

X_original = dataset["body"]
y_original = dataset["label"]

X_train_original, X_test, y_train_original, y_test = train_test_split(
    X_original,
    y_original,
    test_size=0.20,
    random_state=CONFIG["seed"],
    stratify=y_original
)

# Create training DataFrame
train_original_df = pd.DataFrame({
    "body": X_train_original.values,
    "label": y_train_original.values
})

# Create test DataFrame
test_original_df = pd.DataFrame({
    "body": X_test.values,
    "label": y_test.values
})

print("\n" + "=" * 70)
print("ORIGINAL DATA SPLIT")
print("=" * 70)

print(f"Training samples: {len(train_original_df)}")
print(f"Testing samples: {len(test_original_df)}")

print("\nTraining class distribution:")
print(train_original_df["label"].value_counts())

print("\nTesting class distribution:")
print(test_original_df["label"].value_counts())

# ============================================================
# STEP 3: PREPARE SYNTHETIC PHISHING DATA
# ============================================================

# generated_df should already be created by your GAN
# Expected columns:
# body
# label

generated_df = generated_df.copy()

generated_df["body"] = generated_df["body"].fillna("").astype(str)
generated_df["label"] = 1

# Remove empty synthetic emails
generated_df = generated_df[
    generated_df["body"].str.strip() != ""
    ].copy()

print("\n" + "=" * 70)
print("SYNTHETIC PHISHING DATA")
print("=" * 70)

print(f"Generated phishing emails: {len(generated_df)}")

# ============================================================
# STEP 4: GET LEGITIMATE EMAILS FROM TRAINING DATA ONLY
# ============================================================

# IMPORTANT:
# Legitimate emails are selected ONLY from the training set.
# The test set remains completely untouched.

legitimate_train_df = train_original_df[
    train_original_df["label"] == 0
    ].copy()

print(f"\nAvailable legitimate training emails: {len(legitimate_train_df)}")

# ============================================================
# STEP 5: SET CLASS IMBALANCE RATIO
# ============================================================

# Read ratio from CONFIG
# Example:
# (1, 1)   = 1 phishing : 1 legitimate
# (1, 2)   = 1 phishing : 2 legitimate
# (1, 5)   = 1 phishing : 5 legitimate
# (1, 10)  = 1 phishing : 10 legitimate

phishing_ratio, legitimate_ratio = CONFIG[
    "phishing_to_legitimate_ratio"
]

# Number of synthetic phishing samples
n_phishing = len(generated_df)

# Calculate required number of legitimate samples
n_legitimate_required = int(
    n_phishing * legitimate_ratio / phishing_ratio
)

# Cannot use more legitimate samples than available
n_legitimate = min(
    n_legitimate_required,
    len(legitimate_train_df)
)

print("\n" + "=" * 70)
print("TARGET CLASS DISTRIBUTION")
print("=" * 70)

print(
    f"Target ratio: "
    f"{phishing_ratio} phishing : "
    f"{legitimate_ratio} legitimate"
)

print(f"Synthetic phishing used: {n_phishing}")
print(f"Legitimate required: {n_legitimate_required}")
print(f"Legitimate available and used: {n_legitimate}")

# ============================================================
# STEP 6: SELECT LEGITIMATE TRAINING EMAILS
# ============================================================

selected_legitimate_df = legitimate_train_df.sample(
    n=n_legitimate,
    random_state=CONFIG["seed"]
)

# ============================================================
# STEP 7: CREATE FINAL TRAINING DATASET
# ============================================================

# Combine synthetic phishing emails with legitimate training emails
training_dataset = pd.concat(
    [
        generated_df[["body", "label"]],
        selected_legitimate_df[["body", "label"]]
    ],
    ignore_index=True
)

# Shuffle training dataset
training_dataset = training_dataset.sample(
    frac=1,
    random_state=CONFIG["seed"]
).reset_index(drop=True)

print("\n" + "=" * 70)
print("FINAL TRAINING DATASET")
print("=" * 70)

print(f"Total training samples: {len(training_dataset)}")

print("\nClass distribution:")
print(training_dataset["label"].value_counts())

print("\nClass percentage:")
print(
    training_dataset["label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ============================================================
# STEP 8: PREPARE TRAINING AND TESTING DATA
# ============================================================

# Training data:
# Synthetic phishing + original legitimate training emails

X_train = training_dataset["body"].astype(str)
y_train = training_dataset["label"].astype(int)

# Testing data:
# ONLY untouched original emails

X_test = test_original_df["body"].astype(str)
y_test = test_original_df["label"].astype(int)

# ============================================================
# STEP 9: CREATE TF-IDF + LOGISTIC REGRESSION MODEL
# ============================================================

evaluation_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=CONFIG["max_features"],
            stop_words="english",
            min_df=2,
            max_df=0.95
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=CONFIG["seed"],
            solver="liblinear"
        )
    )
])

# ============================================================
# STEP 10: TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION MODEL")
print("=" * 70)

evaluation_model.fit(
    X_train,
    y_train
)

print("Model training completed successfully.")

# ============================================================
# STEP 11: MAKE PREDICTIONS ON UNTOUCHED TEST DATA
# ============================================================

y_pred = evaluation_model.predict(X_test)

# Probability of phishing class = 1
y_scores = evaluation_model.predict_proba(X_test)[:, 1]

# ============================================================
# STEP 12: CALCULATE PERFORMANCE METRICS
# ============================================================

precision = precision_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

# Area Under Precision-Recall Curve
auprc = average_precision_score(
    y_test,
    y_scores
)

# ============================================================
# STEP 13: DISPLAY PERFORMANCE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PHISHING EMAIL DETECTION PERFORMANCE")
print("=" * 70)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"AUPRC     : {auprc:.4f}")

# ============================================================
# STEP 14: DISPLAY CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Legitimate",
            "Phishing"
        ],
        zero_division=0
    )
)

# ============================================================
# STEP 15: CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("                 Predicted")
print("              Legitimate  Phishing")
print(
    f"Actual Legitimate    {cm[0][0]:<10} {cm[0][1]}"
)
print(
    f"Actual Phishing      {cm[1][0]:<10} {cm[1][1]}"
)

# ============================================================
# STEP 16: SAVE MODEL PERFORMANCE RESULTS
# ============================================================

performance_results = pd.DataFrame([{
    "Model": "Logistic Regression",

    "Target_Ratio":
        f"{phishing_ratio}:{legitimate_ratio}",

    "Training_Samples":
        len(X_train),

    "Testing_Samples":
        len(X_test),

    "Synthetic_Phishing":
        n_phishing,

    "Legitimate_Training":
        n_legitimate,

    "Test_Phishing":
        int((y_test == 1).sum()),

    "Test_Legitimate":
        int((y_test == 0).sum()),

    "Precision":
        round(precision, 4),

    "Recall":
        round(recall, 4),

    "F1_Score":
        round(f1, 4),

    "AUPRC":
        round(auprc, 4)
}])

performance_path = "src/results/model_performance.csv"

performance_results.to_csv(
    performance_path,
    index=False
)

print("\n" + "=" * 70)
print("RESULTS SAVED SUCCESSFULLY")
print("=" * 70)

print(performance_results.to_string(index=False))

print(
    f"\nModel performance saved to:\n{performance_path}"
)
# -----------------------------------
# DISPLAY RESULTS
# -----------------------------------

print("\n" + "-" * 50)
print("PHISHING DETECTION PERFORMANCE")
print("-" * 50)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUPRC:     {auprc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    )
)

# -----------------------------------
# SAVE PERFORMANCE RESULTS
# -----------------------------------

os.makedirs("src/results", exist_ok=True)

performance_results = pd.DataFrame([{
    "Model": "Logistic Regression",
    "Dataset": os.path.basename(CONFIG["output_path"]),
    "Total_Samples": len(final_dataset),
    "Training_Samples": len(X_train),
    "Testing_Samples": len(X_test),
    "Phishing_Samples": int(sum(final_dataset['label'] == 1)),
    "Legitimate_Samples": int(sum(final_dataset['label'] == 0)),
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "AUPRC": auprc
}])

performance_path = "src/results/model_performance.csv"

performance_results.to_csv(
    performance_path,
    index=False
)

print(f"\n✓ Model performance saved to: {performance_path}")

print("✓ Dataset exported successfully!")
print("\nSample data:")
print(final_dataset.head(10))

# -----------------------------------
# STATISTICS
# -----------------------------------

print("\n" + "=" * 70)
print("Summary Statistics")
print("=" * 70)
print(f"Total emails: {len(final_dataset)}")
print(f"Unique words per email (avg): {final_dataset['body'].str.split().str.len().mean():.2f}")
print(f"Email length (avg chars): {final_dataset['body'].str.len().mean():.2f}")

# Show some example generated emails
print("\nExample Generated Phishing Emails:")
for i in range(min(5, len(generated_df))):
    print(f"  {i + 1}. {generated_df.iloc[i]['body'][:100]}...")
