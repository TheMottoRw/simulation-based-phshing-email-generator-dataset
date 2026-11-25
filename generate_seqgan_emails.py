#!/usr/bin/env python3
"""
CPU-friendly SeqGAN-style phishing-email text generator.

Important:
- Trains only on phishing/spam emails from the REAL TRAINING partition.
- Splits the real dataset before training to avoid using held-out test records.
- Generates synthetic phishing text only; it does not generate legitimate emails.
- The output is intended for controlled defensive research and classifier evaluation.

Install:
    pip install pandas numpy scikit-learn torch

Example:
    python generate_seqgan_emails.py \
        --data src/dataset/SpamAssasin.csv \
        --output src/dataset/seqgan_generated_phishing.csv \
        --num-generate 2000 \
        --text-col body --label-col label

If column names are not supplied, the script attempts to detect them.
"""

import argparse
import html
import re
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


PAD, BOS, EOS, UNK = "<PAD>", "<BOS>", "<EOS>", "<UNK>"
SPECIALS = [PAD, BOS, EOS, UNK]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean_email(text):
    text = html.unescape(str(text))
    text = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"(?i)https?://\\S+|www\\.\\S+", " URL ", text)
    text = re.sub(r"\\b[\\w.+-]+@[\\w.-]+\\.[A-Za-z]{2,}\\b", " EMAIL ", text)
    text = re.sub(r"\\s+", " ", text).strip().lower()
    return text


def tokenize(text):
    # Keep words, numbers and common punctuation as discrete tokens.
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\\d+|[!?.,:;()$%'-]", text.lower())


def detect_column(columns, candidates):
    lower = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    return None


def normalize_label(value):
    s = str(value).strip().lower()
    positive = {"1", "spam", "phishing", "phish", "junk", "malicious", "1.0", "true"}
    negative = {"0", "ham", "legitimate", "legit", "normal", "benign", "0.0", "false"}
    if s in positive:
        return 1
    if s in negative:
        return 0
    try:
        n = float(s)
        if n == 1:
            return 1
        if n == 0:
            return 0
    except ValueError:
        pass
    return None


def load_data(args):
    df = pd.read_csv(args.data)
    text_col = args.text_col or detect_column(
        df.columns, ["body"]
    )
    label_col = args.label_col or detect_column(
        df.columns, ["label"]
    )
    if not text_col or not label_col:
        raise ValueError(
            f"Could not detect text/label columns. Found: {list(df.columns)}. "
            "Pass --text-col and --label-col explicitly."
        )

    work = df[[text_col, label_col]].copy()
    work.columns = ["text", "raw_label"]
    work["label"] = work["raw_label"].map(normalize_label)
    work["text"] = work["text"].fillna("").map(clean_email)
    work = work.dropna(subset=["label"])
    work = work[work["text"].str.len() > 0]
    work["label"] = work["label"].astype(int)

    if work["label"].nunique() != 2:
        raise ValueError(
            "The label column must contain both spam/phishing and ham/legitimate classes. "
            "Check your label values or extend normalize_label()."
        )
    return work, text_col, label_col


def build_vocab(texts, max_vocab):
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocab_tokens = [w for w, _ in counts.most_common(max(0, max_vocab - len(SPECIALS)))]
    itos = SPECIALS + [w for w in vocab_tokens if w not in SPECIALS]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def encode(text, stoi, max_len):
    ids = [stoi[BOS]]
    ids.extend(stoi.get(tok, stoi[UNK]) for tok in tokenize(text)[: max_len - 2])
    ids.append(stoi[EOS])
    ids = ids[:max_len]
    ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


def decode(ids, itos):
    words = []
    for idx in ids:
        token = itos[int(idx)]
        if token == EOS:
            break
        if token in (PAD, BOS):
            continue
        words.append(token)
    text = " ".join(words)
    text = re.sub(r"\\s+([!?.,:;%)])", r"\\1", text)
    text = re.sub(r"([(])\\s+", r"\\1", text)
    return text.strip()


class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens):
        x = self.embedding(tokens)
        out, _ = self.lstm(x)
        return self.output(out)

    def step(self, token, state=None):
        x = self.embedding(token)
        out, state = self.lstm(x, state)
        return self.output(out[:, -1, :]), state

    @torch.no_grad()
    def sample(self, batch_size, max_len, bos_idx, eos_idx, temperature=1.0):
        device = next(self.parameters()).device
        token = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
        state = None
        sequences = [token]
        log_probs = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits, state = self.step(token, state)
            logits = logits / max(temperature, 1e-4)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            nxt = dist.sample()
            log_probs.append(dist.log_prob(nxt))
            nxt = torch.where(finished, torch.full_like(nxt, eos_idx), nxt)
            token = nxt.unsqueeze(1)
            sequences.append(token)
            finished |= nxt.eq(eos_idx)
            if finished.all():
                break

        seq = torch.cat(sequences, dim=1)
        if seq.size(1) < max_len:
            pad = torch.full((batch_size, max_len - seq.size(1)), 0, dtype=torch.long, device=device)
            seq = torch.cat([seq, pad], dim=1)
        return seq[:, :max_len], log_probs


class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, tokens):
        mask = tokens.ne(PAD_IDX)
        x = self.embedding(tokens)
        out, _ = self.gru(x)
        lengths = mask.sum(dim=1).clamp(min=1)
        last = out[torch.arange(out.size(0), device=out.device), lengths - 1]
        return self.fc(last).squeeze(1)


def batches(array, batch_size, shuffle=True):
    idx = np.arange(len(array))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield array[idx[start : start + batch_size]]


def pretrain_generator(generator, real_ids, args, device):
    optimizer = torch.optim.Adam(generator.parameters(), lr=args.generator_lr)
    data = torch.tensor(real_ids, dtype=torch.long, device=device)
    for epoch in range(args.pretrain_epochs):
        generator.train()
        losses = []
        for batch in batches(data, args.batch_size):
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = generator(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=PAD_IDX,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        print(f"Generator pretraining {epoch + 1}/{args.pretrain_epochs}: loss={np.mean(losses):.4f}")


def train_discriminator(discriminator, generator, real_ids, args, device):
    generator.eval()
    with torch.no_grad():
        fake_parts = []
        for _ in range(max(1, int(np.ceil(len(real_ids) / args.batch_size)))):
            fake, _ = generator.sample(
                args.batch_size, args.max_len, BOS_IDX, EOS_IDX, args.temperature
            )
            fake_parts.append(fake)
        fake_ids = torch.cat(fake_parts, dim=0)[: len(real_ids)]

    real = torch.tensor(real_ids, dtype=torch.long, device=device)
    fake = fake_ids.to(device)
    x = torch.cat([real, fake], dim=0)
    y = torch.cat(
        [torch.ones(len(real), device=device), torch.zeros(len(fake), device=device)]
    )
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.discriminator_lr)
    discriminator.train()
    for _ in range(args.discriminator_epochs):
        perm = torch.randperm(len(x), device=device)
        for start in range(0, len(x), args.batch_size):
            ix = perm[start : start + args.batch_size]
            logits = discriminator(x[ix])
            loss = F.binary_cross_entropy_with_logits(logits, y[ix])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            optimizer.step()


def rollout_reward(generator, discriminator, sequences, args, device):
    """
    Monte Carlo completion estimates Q(s_t, a_t): for each prefix, sample
    several completions and average the discriminator's probability that
    the completed sequence is real.
    """
    generator.eval()
    discriminator.eval()
    batch_size, seq_len = sequences.shape
    rewards = torch.zeros(batch_size, seq_len - 1, device=device)

    with torch.no_grad():
        for t in range(1, seq_len):
            prefix = sequences[:, : t + 1]
            valid = sequences[:, t].ne(PAD_IDX)
            if not valid.any():
                continue
            prefix = prefix[valid]
            estimates = []
            for _ in range(args.rollouts):
                completed = []
                for row in prefix:
                    current = row.unsqueeze(0)
                    state = None
                    # Rebuild recurrent state from the prefix.
                    embedded = generator.embedding(current)
                    _, state = generator.lstm(embedded)
                    token = current[:, -1:]
                    generated = [current]
                    for _step in range(seq_len - current.size(1)):
                        logits, state = generator.step(token, state)
                        probs = F.softmax(logits / max(args.temperature, 1e-4), dim=-1)
                        nxt = torch.multinomial(probs, 1)
                        generated.append(nxt)
                        token = nxt
                        if int(nxt.item()) == EOS_IDX:
                            break
                    full = torch.cat(generated, dim=1)
                    if full.size(1) < seq_len:
                        full = torch.cat(
                            [full, torch.zeros(1, seq_len - full.size(1), dtype=torch.long, device=device)],
                            dim=1,
                        )
                    completed.append(full[:, :seq_len])
                completed = torch.cat(completed, dim=0)
                estimates.append(torch.sigmoid(discriminator(completed)))
            rewards[valid, t - 1] = torch.stack(estimates).mean(dim=0)
    return rewards


def adversarial_train(generator, discriminator, real_ids, args, device):
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=args.generator_lr)
    data = torch.tensor(real_ids, dtype=torch.long, device=device)

    for epoch in range(args.adversarial_epochs):
        generator.train()
        epoch_losses = []
        steps = max(1, int(np.ceil(len(data) / args.batch_size)))
        for _ in range(steps):
            sampled, _ = generator.sample(
                args.batch_size, args.max_len, BOS_IDX, EOS_IDX, args.temperature
            )
            rewards = rollout_reward(generator, discriminator, sampled, args, device)

            # Recompute differentiable log probabilities for sampled actions.
            logits = generator(sampled[:, :-1])
            log_probs = F.log_softmax(logits, dim=-1)
            chosen = log_probs.gather(2, sampled[:, 1:].unsqueeze(-1)).squeeze(-1)
            mask = sampled[:, 1:].ne(PAD_IDX).float()
            loss = -(chosen * rewards * mask).sum() / mask.sum().clamp(min=1)

            gen_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            gen_optimizer.step()
            epoch_losses.append(loss.item())

        train_discriminator(discriminator, generator, real_ids, args, device)
        print(
            f"Adversarial epoch {epoch + 1}/{args.adversarial_epochs}: "
            f"policy_loss={np.mean(epoch_losses):.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Train SeqGAN and generate synthetic phishing emails.")
    parser.add_argument("--data", required=True, help="Path to the original labeled email CSV.")
    parser.add_argument("--output", default="src/dataset/seqgan_generated_phishing.csv")
    parser.add_argument("--text-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--num-generate", type=int, default=1000)
    parser.add_argument("--max-vocab", type=int, default=1500)
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--adversarial-epochs", type=int, default=3)
    parser.add_argument("--discriminator-epochs", type=int, default=2)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--generator-lr", type=float, default=0.001)
    parser.add_argument("--discriminator-lr", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-tokens", type=int, default=5)
    parser.add_argument("--max-attempts-factor", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    if args.num_generate < 1:
        parser.error("--num-generate must be at least 1")

    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df, text_col, label_col = load_data(args)
    print(f"Loaded {len(df)} usable records; text='{text_col}', label='{label_col}'")
    print("Label counts:", df["label"].value_counts().sort_index().to_dict())

    # Split FIRST. Only phishing examples in the training partition train SeqGAN.
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.seed, stratify=df["label"]
    )
    phishing_train = train_df.loc[train_df["label"] == 1, "text"].tolist()
    if len(phishing_train) < 20:
        raise ValueError(
            f"Only {len(phishing_train)} phishing training emails found. "
            "SeqGAN needs a larger training set for a meaningful experiment."
        )
    print(f"Real phishing emails used for SeqGAN training: {len(phishing_train)}")
    print(f"Held-out real test records (not used for generation): {len(test_df)}")

    global PAD_IDX, BOS_IDX, EOS_IDX
    stoi, itos = build_vocab(phishing_train, args.max_vocab)
    PAD_IDX, BOS_IDX, EOS_IDX = stoi[PAD], stoi[BOS], stoi[EOS]
    print(f"Vocabulary size: {len(itos)}")

    encoded = np.array([encode(t, stoi, args.max_len) for t in phishing_train], dtype=np.int64)

    generator = Generator(len(itos), args.embedding_dim, args.hidden_dim, PAD_IDX).to(device)
    discriminator = Discriminator(len(itos), args.embedding_dim, args.hidden_dim, PAD_IDX).to(device)

    print("\\nStage 1: supervised generator pretraining")
    pretrain_generator(generator, encoded, args, device)

    print("\\nStage 2: SeqGAN adversarial training")
    adversarial_train(generator, discriminator, encoded, args, device)

    print("\\nGenerating synthetic phishing emails...")
    generated = []
    seen = set(phishing_train)
    attempts = 0
    max_attempts = args.num_generate * args.max_attempts_factor
    generator.eval()

    with torch.no_grad():
        while len(generated) < args.num_generate and attempts < max_attempts:
            remaining = min(args.batch_size, args.num_generate - len(generated))
            samples, _ = generator.sample(
                remaining, args.max_len, BOS_IDX, EOS_IDX, args.temperature
            )
            for row in samples.cpu().numpy():
                text = decode(row, itos)
                attempts += 1
                if len(tokenize(text)) < args.min_tokens:
                    continue
                if text in seen:
                    continue
                seen.add(text)
                generated.append(text)
                if len(generated) >= args.num_generate:
                    break
            if attempts >= max_attempts:
                break

    if not generated:
        raise RuntimeError(
            "The generator produced no usable unique samples. Try more pretraining, "
            "a larger vocabulary, a larger max length, or inspect the input data."
        )

    output = pd.DataFrame({
        "body": generated,
        "label": ["phishing"] * len(generated),
        "source": ["SeqGAN"] * len(generated),
    })
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\\nSaved {len(output)} synthetic phishing emails to: {out_path}")
    if len(output) < args.num_generate:
        print(
            f"NOTE: Requested {args.num_generate}, but generated {len(output)} unique "
            "samples before reaching the attempt limit."
        )
    print("Output columns: body, label, source")
    print("Reminder: evaluate only against a separate, fully real held-out test set.")


if __name__ == "__main__":
    main()
