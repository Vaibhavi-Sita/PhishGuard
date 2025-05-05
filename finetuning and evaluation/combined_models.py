"""
Author: Sita Vaibhavi Gunturi

This script trains and evaluates three phishing-URL classifiers
(Logistic Regression, URLNet, Gemini LLM) on the any/attached dataset.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# ------------------------------
# PyTorch for URLNet
# ------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------------------
# Gemini LLM Classifier
# ------------------------------
import google.generativeai as genai

# ====== Logistic Regression ======

def run_logistic_regression(df):
    y = df['status']
    X = df.drop(['status'], axis=1)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    text_col = 'url'
    categorical = [c for c in X.columns if c not in numeric_cols + [text_col]]
    pre = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('txt', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,7), max_features=2000), text_col),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical)
    ])
    pipe = Pipeline([('pre', pre), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:,1]
    return compute_metrics(y_test, y_pred, y_prob)

# ====== URLNet Model ======

class URLNetDataset(Dataset):
    def __init__(self, seqs, nums, labels):
        self.seqs = torch.LongTensor(seqs)
        self.nums = torch.FloatTensor(nums)
        self.labels = torch.FloatTensor(labels)
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        return self.seqs[idx], self.nums[idx], self.labels[idx]

class URLNetMulti(nn.Module):
    def __init__(self, vocab_size, num_feats, max_len=200):
        super().__init__()
        self.embed = nn.Embedding(vocab_size+1, 32, padding_idx=0)
        self.conv1 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        conv_dim = (max_len//4)*128
        self.num_branch = nn.Sequential(nn.Linear(num_feats, 128), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(conv_dim+128, 64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
    def forward(self, seqs, nums):
        x = self.embed(seqs).permute(0,2,1)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(seqs.size(0), -1)
        n = self.num_branch(nums)
        return self.classifier(torch.cat([x,n],1)).squeeze()

def run_urlnet(df):
    y = df['status'].map({'legitimate':0,'phishing':1}).astype(float).values
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c!='status']
    X_num = df[num_cols].values.astype(float)
    max_len=200
    chars = sorted(set(''.join(df['url'])))
    c2i = {c:i+1 for i,c in enumerate(chars)}
    seqs = np.vstack([[c2i.get(ch,0) for ch in u[:max_len]] + [0]*(max_len-len(u)) for u in df['url']])
    s_tr, s_te, n_tr, n_te, y_tr, y_te = train_test_split(seqs, X_num, y, test_size=0.2, stratify=y, random_state=42)
    tr_ds = URLNetDataset(s_tr, n_tr, y_tr)
    te_ds = URLNetDataset(s_te, n_te, y_te)
    tr_lo = DataLoader(tr_ds, batch_size=64, shuffle=True)
    te_lo = DataLoader(te_ds, batch_size=64)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = URLNetMulti(vocab_size=len(c2i), num_feats=n_tr.shape[1], max_len=max_len).to(dev)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCELoss()
    for _ in range(3):
        model.train()
        for seqs_b, nums_b, lbl_b in tr_lo:
            seqs_b, nums_b, lbl_b = seqs_b.to(dev), nums_b.to(dev), lbl_b.to(dev)
            opt.zero_grad()
            loss = crit(model(seqs_b, nums_b), lbl_b)
            loss.backward(); opt.step()
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for seqs_b, nums_b, _ in te_lo:
            seqs_b, nums_b = seqs_b.to(dev), nums_b.to(dev)
            p = model(seqs_b, nums_b).cpu().numpy()
            preds.extend((p>=0.5).astype(int)); probs.extend(p)
    return compute_metrics(y_te, np.array(preds), np.array(probs))

# ====== Gemini LLM ======

def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")

def run_gemini(df):
    df['status_mapped'] = df['status'].map({'legitimate':0,'phishing':1}).astype(int)
    tr, te = train_test_split(df, test_size=0.2, stratify=df['status_mapped'], random_state=42)
    gm = configure_gemini()
    preds_tr, probs_tr = classify_with_gemini(gm, tr)
    preds_te, probs_te = classify_with_gemini(gm, te)
    print("== Gemini Metrics ==")
    print("Train:", compute_metrics(tr['status_mapped'], preds_tr, probs_tr))
    print("Test :", compute_metrics(te['status_mapped'], preds_te, probs_te))

def classify_with_gemini(model, df_sub, batch_size=10):
    preds, probs = [], []
    feats = [c for c in df_sub.columns if c not in ['url','status','status_mapped']]
    for i in range(0,len(df_sub),batch_size):
        for _, row in df_sub.iloc[i:i+batch_size].iterrows():
            feat_str = ", ".join(f"{c}={row[c]}" for c in feats)
            prompt = f'URL: {row["url"]}. Features: {feat_str}. Respond with JSON {{"label":0,"confidence":0.00}}.'
            resp = model.generate_content(prompt)
            txt = resp.text.strip()
            try:
                js = json.loads(txt)
                lab, conf = int(js.get("label",0)), float(js.get("confidence",0))
            except:
                lab = 1 if "phishing" in txt.lower() else 0; conf=0.5
            preds.append(lab); probs.append(conf)
    return np.array(preds), np.array(probs)

# ====== Utilities ======

def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    aucv = auc(*roc_curve(y_true, y_prob)[:2])
    return {'accuracy':acc, 'precision':prec, 'recall':rec, 'f1':f1, 'roc_auc':aucv}

def main():
    df = pd.read_csv('./Datasets/dataset_phishing.csv')
    print("\n== Logistic Regression ==")
    print(run_logistic_regression(df))
    print("\n== URLNet ==")
    print(run_urlnet(df))
    print("\n== Gemini LLM ==")
    run_gemini(df)

if __name__ == "__main__":
    main()
