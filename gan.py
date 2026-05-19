import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# -----------------------------------
# LOAD DATASET
# -----------------------------------

dataset = pd.read_csv("src/dataset/SpamAssasin.csv")

# -----------------------------------
# CLEAN DATA
# -----------------------------------

dataset = dataset.dropna(subset=["body"])

dataset["body"] = dataset["body"].astype(str)

# Remove empty rows
dataset = dataset[dataset["body"].str.strip() != ""]

# -----------------------------------
# FILTER DATA
# Labels:
# 1 = phishing
# 0 = legitimate
# -----------------------------------

phishing_data = dataset[dataset["label"] == 1]["body"]

legitimate_data = dataset[dataset["label"] == 0]["body"]

print("Phishing emails:", len(phishing_data))
print("Legitimate emails:", len(legitimate_data))

# -----------------------------------
# TF-IDF FEATURE EXTRACTION
# -----------------------------------

vectorizer = TfidfVectorizer(
    max_features=50,
    stop_words='english'
)

X_phishing = vectorizer.fit_transform(phishing_data).toarray()

print("TF-IDF Shape:", X_phishing.shape)

# -----------------------------------
# MODEL PARAMETERS
# -----------------------------------

input_dim = X_phishing.shape[1]
latent_dim = 16

# -----------------------------------
# GENERATOR
# -----------------------------------

generator = Sequential([
    Dense(32, activation='relu', input_dim=latent_dim),
    Dense(64, activation='relu'),
    Dense(input_dim, activation='sigmoid')
])

# -----------------------------------
# DISCRIMINATOR
# -----------------------------------

discriminator = Sequential([
    Dense(64, activation='relu', input_dim=input_dim),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

discriminator.compile(
    optimizer=Adam(0.0002),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -----------------------------------
# GAN
# -----------------------------------

discriminator.trainable = False

gan = Sequential([
    generator,
    discriminator
])

gan.compile(
    optimizer=Adam(0.0002),
    loss='binary_crossentropy'
)

# -----------------------------------
# TRAINING
# -----------------------------------

epochs = 200
batch_size = 8

for epoch in range(epochs):

    idx = np.random.randint(
        0,
        X_phishing.shape[0],
        batch_size
    )

    real_emails = X_phishing[idx]

    noise = np.random.normal(
        0,
        1,
        (batch_size, latent_dim)
    )

    fake_emails = generator.predict(
        noise,
        verbose=0
    )

    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    discriminator.trainable = True

    d_loss_real = discriminator.train_on_batch(
        real_emails,
        real_labels
    )

    d_loss_fake = discriminator.train_on_batch(
        fake_emails,
        fake_labels
    )

    # Train Generator

    noise = np.random.normal(
        0,
        1,
        (batch_size, latent_dim)
    )

    misleading_labels = np.ones((batch_size, 1))

    discriminator.trainable = False

    g_loss = gan.train_on_batch(
        noise,
        misleading_labels
    )

    if epoch % 50 == 0:

        print(f"\nEpoch {epoch}")

        print(f"D Loss Real: {d_loss_real[0]:.4f}")

        print(f"D Loss Fake: {d_loss_fake[0]:.4f}")

        print(f"G Loss: {g_loss:.4f}")

# -----------------------------------
# GENERATE SYNTHETIC EMAILS
# -----------------------------------

num_generated_samples = 500

noise = np.random.normal(
    0,
    1,
    (num_generated_samples, latent_dim)
)

generated_vectors = generator.predict(noise)

feature_names = vectorizer.get_feature_names_out()

synthetic_emails = []

for vector in generated_vectors:

    top_indices = vector.argsort()[-10:]

    words = [
        feature_names[i]
        for i in top_indices
    ]

    sentence = " ".join(words)

    synthetic_emails.append(sentence)

# -----------------------------------
# GENERATED DATAFRAME
# -----------------------------------

generated_df = pd.DataFrame({
    "body": synthetic_emails,
    "label": 1
})

# -----------------------------------
# LEGITIMATE DATAFRAME
# -----------------------------------

legitimate_df = pd.DataFrame({
    "body": legitimate_data.values,
    "label": 0
})

# -----------------------------------
# FINAL DATASET
# -----------------------------------

final_dataset = pd.concat([
    legitimate_df,
    generated_df
])

final_dataset = final_dataset.sample(
    frac=1
).reset_index(drop=True)

# -----------------------------------
# EXPORT CSV
# -----------------------------------

final_dataset.to_csv(
    "src/dataset/generated_phishing_dataset.csv",
    index=False
)

print("\nDataset exported successfully.")
print(final_dataset.head())