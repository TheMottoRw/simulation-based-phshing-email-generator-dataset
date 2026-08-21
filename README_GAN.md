This GAN-based generator creates realistic synthetic phishing emails by learning patterns from real phishing samples. It addresses the critical challenge of class imbalance in phishing detection datasets, where legitimate emails vastly outnumber phishing attempts.
Key Features

    Deep Learning Generation: Uses GANs to create realistic phishing emails

    TF-IDF Feature Extraction: Converts text to numerical vectors

    Label Smoothing: Improves training stability

    Dropout Regularization: Prevents overfitting

    Balanced Dataset Creation: Combines synthetic phishing with real legitimate emails

    Reproducible Results: Fixed random seeds for consistent outputs

🤔 Why GANs for Phishing Data?
`Typical Phishing Dataset:
├── Legitimate Emails: 95% (9,500 emails)
└── Phishing Emails:   5%  (500 emails)`

### This severe imbalance causes:

` Models biased toward legitimate emails
    Poor detection of phishing attempts
    High false negative rates`

### The Solution

GANs generate synthetic phishing emails that:

    Augment the minority class
    Preserve statistical properties of real phishing
    Increase model performance on phishing detection
    Reduce false negatives

┌─────────────────────────────────────────────────────────────┐
│                      GAN Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Random Noise (Latent Space)                               │
│         │                                                   │
│         ▼                                                   │
│   ┌─────────────┐                                          │
│   │  GENERATOR  │  ──► Synthetic Phishing Vectors          │
│   └─────────────┘                                          │
│         │                                                   │
│         │         ┌──────────────┐                         │
│         └────────►│              │                         │
│                   │ DISCRIMINATOR │◄── Real Phishing Vectors│
│                   │              │                         │
│                   └──────────────┘                         │
│                         │                                   │
│                         ▼                                   │
│              Real or Fake? (Binary)                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘

## Generator
Role: Creates synthetic phishing email vectors from random noise
`Generator(
    Dense(64, activation='relu'),    # First hidden layer
    Dropout(0.2),                     # Prevent overfitting
    Dense(128, activation='relu'),   # Second hidden layer
    Dropout(0.2),                     # Prevent overfitting
    Dense(input_dim, activation='sigmoid')  # Output layer
)`
## Discriminator
Role: Distinguishes real from synthetic phishing emails
`Discriminator(
    Dense(128, activation='relu'),   # First hidden layer
    Dropout(0.3),                     # Prevent overfitting
    Dense(64, activation='relu'),    # Second hidden layer
    Dropout(0.3),                     # Prevent overfitting
    Dense(1, activation='sigmoid')   # Binary classification
)`

### Custom Configuration
`CONFIG = {
    'dataset_path': 'src/dataset/SpamAssasin.csv',
    'output_path': 'src/dataset/generated_phishing_dataset.csv',
    'max_features': 50,          # TF-IDF features
    'latent_dim': 16,            # GAN latent space
    'epochs': 200,               # Training iterations
    'batch_size': 8,             # Training batch size
    'n_generated': 500,          # Number of synthetic emails
    'seed': 42                   # Reproducibility
}`
