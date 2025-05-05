"""
Author: Sita Vaibhavi Gunturi

This script implements a phishing URL classifier using Logistic Regression with basic URL features.
The model is trained on the PhiUSIIL dataset with added noise to improve robustness.

Key Functions:
- extract_basic_features(): Extracts URL features (length, digit count, special character count, etc)
- main(): Orchestrates the data loading, feature extraction, model training, and evaluation process
  - Includes noise injection for both labels (10%) and features (20% chance per feature)
  - Uses a pipeline with StandardScaler and LogisticRegression
  - Evaluates using accuracy, precision, recall, F1, and ROC AUC metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

def extract_features(url):
    return {
        'url_length': len(url),
        'num_digits': sum(c.isdigit() for c in url),
        'num_special': sum(not c.isalnum() for c in url)
    }

def main():
    df = pd.read_csv('./Datasets/PhiUSIIL_Phishing_URL_Dataset.csv')
    y = df['label']
    
    basic_feats = df['URL'].apply(lambda u: pd.Series(extract_features(u)))
    X = basic_feats
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    np.random.seed(42)
    noise_rate = 0.1
    n_noise = int(noise_rate * len(y_train))
    noise_idx = np.random.choice(len(y_train), size=n_noise, replace=False)
    y_train_noisy = y_train.copy().to_numpy()
    y_train_noisy[noise_idx] = 1 - y_train_noisy[noise_idx]
    
    X_train_noisy = X_train.copy().astype(float)
    for col in X_train_noisy.columns:
        if np.random.rand() < 0.2:
            X_train_noisy[col] += np.random.normal(0, X_train_noisy[col].std()*0.1, size=len(X_train_noisy))

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced'))
    ])
    
    pipeline.fit(X_train_noisy, y_train_noisy)
    
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:,1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = auc(*roc_curve(y_test, y_prob)[:2])
    
    print("Logistic Regression:")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC AUC  : {roc_auc:.4f}")
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"LR (AUC={roc_auc:.2f})")
    plt.plot([0,1], [0,1], '--', label='Chance')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — LR")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
