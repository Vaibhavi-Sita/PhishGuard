"""
Author: Sita Vaibhavi Gunturi

This script implements URLNet, a deep learning model for phishing URL detection using character-level CNN features.
The model combines convolutional neural networks for URL text processing with a separate branch for numeric features.

Key Classes and Functions:
- URLNetDataset: PyTorch Dataset class for handling URL sequences, numeric features, and labels
- URLNetNoisy: Neural network model with:
  - Character embedding layer
  - CNN for sequence processing
  - Separate branch for numeric features
  - Combined classifier with dropout for regularization
- main(): Orchestrates the entire pipeline:
  - Data loading and preprocessing
  - Noise injection (20% label noise, feature noise)
  - Model training with Adam optimizer
  - Evaluation using accuracy, precision, recall, F1, and ROC AUC metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

class URLNetDataset(Dataset):
    def __init__(self, seqs, nums, labels):
        self.seqs = torch.LongTensor(seqs)
        self.nums = torch.FloatTensor(nums)
        self.labels = torch.FloatTensor(labels)
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        return self.seqs[idx], self.nums[idx], self.labels[idx]

class URLNetNoisy(nn.Module):
    def __init__(self, vocab_size, num_feats, max_len=200, embed_dim=16, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size+1, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        conv_flat_dim = (max_len // 2) * 32

        self.num_branch = nn.Sequential(
            nn.Linear(num_feats, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(conv_flat_dim + hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, seqs, nums):
        x = self.embedding(seqs)
        x = x.permute(0, 2, 1)
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        n = self.num_branch(nums)
        combined = torch.cat([x, n], dim=1)
        return self.classifier(combined).squeeze()

def main():
    df = pd.read_csv('./Datasets/PhiUSIIL_Phishing_URL_Dataset.csv')
    y = df['label'].values.astype(float)
    rng = np.random.RandomState(42)
    flip_idx = rng.choice(len(y), size=int(0.2 * len(y)), replace=False)
    y[flip_idx] = 1 - y[flip_idx]

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'label']
    X_num = df[numeric_cols].values.astype(float)
    noise = rng.normal(0, X_num.std(axis=0)*0.1, X_num.shape)
    X_num += noise

    max_len = 200
    url_col = 'URL'
    chars = sorted(set(''.join(df[url_col].astype(str))))
    char2idx = {c: i+1 for i, c in enumerate(chars)}
    def url_to_seq(u):
        seq = [char2idx.get(ch, 0) for ch in str(u)[:max_len]]
        return seq + [0] * (max_len - len(seq))
    seqs = np.vstack([url_to_seq(u) for u in df[url_col]])

    seqs_train, seqs_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        seqs, X_num, y, test_size=0.2, random_state=42, stratify=df['label']
    )

    train_ds = URLNetDataset(seqs_train, X_num_train, y_train)
    test_ds  = URLNetDataset(seqs_test,  X_num_test,  y_test)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=128)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = URLNetNoisy(vocab_size=len(char2idx), num_feats=X_num_train.shape[1], max_len=max_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.BCELoss()

    for epoch in range(1, 4):
        model.train()
        total_loss = 0
        for seqs_b, nums_b, labels_b in train_loader:
            seqs_b, nums_b, labels_b = seqs_b.to(device), nums_b.to(device), labels_b.to(device)
            optimizer.zero_grad()
            out = model(seqs_b, nums_b)
            loss = criterion(out, labels_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for seqs_b, nums_b, _ in test_loader:
            seqs_b, nums_b = seqs_b.to(device), nums_b.to(device)
            p = model(seqs_b, nums_b).cpu().numpy()
            preds.extend((p >= 0.5).astype(int))
            probs.extend(p)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
    accuracy  = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall    = recall_score(y_test, preds, zero_division=0)
    f1        = f1_score(y_test, preds, zero_division=0)
    roc_auc   = auc(*roc_curve(y_test, probs)[:2])

    print(f"URLNet metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f"(AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],'--', label='Chance')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC — URLNet")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
