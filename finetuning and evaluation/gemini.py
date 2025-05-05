"""
Author: Sita Vaibhavi Gunturi

This script implements a phishing URL classifier using Google's Gemini LLM model on the PhiUSIIL dataset.
The code has been modified to remove the API key for security reasons as this is now a public repository.

Key Functions:
- configure_gemini(): Sets up the Gemini model configuration (API key removed for security)
- classify_with_gemini(): Classifies URLs as phishing or legitimate using Gemini model with batch processing
- evaluate_and_plot(): Calculates and plots performance metrics (accuracy, precision, recall, F1, ROC)
- main(): Orchestrates the data loading, model training, and evaluation process

Note: The original API key has been removed to prevent misuse in this public repository.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

import google.generativeai as genai

def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the GEMINI_API_KEY environment variable.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")

def classify_with_gemini(model, df_subset, batch_size=10):
    preds, probs = [], []
    feature_cols = [c for c in df_subset.columns if c not in ['URL', 'label', 'label_mapped']]
    for i in range(0, len(df_subset), batch_size):
        batch = df_subset.iloc[i:i+batch_size]
        for _, row in batch.iterrows():
            feats = ", ".join(f"{col}={row[col]}" for col in feature_cols)
            prompt = (
                f"You are a cybersecurity expert. URL: {row['URL']}. Features: {feats}. "
                "Respond only with JSON {"label":0,"confidence":0.00} "
                "(label=1 means phishing, 0 means legitimate)."
            )
            resp = model.generate_content(prompt)
            text = resp.text.strip()
            try:
                res = json.loads(text)
                lab = int(res.get("label", 0))
                conf = float(res.get("confidence", 0.0))
            except:
                lab = 1 if "phishing" in text.lower() else 0
                conf = 0.5
            preds.append(lab)
            probs.append(conf)
    return np.array(preds), np.array(probs)

def evaluate_and_plot(name, y_true, y_pred, y_prob):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': auc(*roc_curve(y_true, y_prob)[:2])
    }
    print(f"{name} metrics:", metrics)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={metrics['roc_auc']:.2f})")

def main():
    df = pd.read_csv('./Datasets/PhiUSIIL_Phishing_URL_Dataset.csv')
    df['label_mapped'] = df['label'].astype(int)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label_mapped']
    )

    gem_model = configure_gemini()
    train_pred, train_prob = classify_with_gemini(gem_model, train_df)
    evaluate_and_plot("Gemini (train)", train_df['label_mapped'], train_pred, train_prob)
    test_pred, test_prob = classify_with_gemini(gem_model, test_df)
    evaluate_and_plot("Gemini (test)", test_df['label_mapped'], test_pred, test_prob)

    plt.plot([0,1], [0,1], '--', label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Gemini")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
