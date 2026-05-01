---
title: FraudGuard AI
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# FraudGuard AI — Fraud Investigation Copilot

FraudGuard AI is a production-style machine learning web application that detects suspicious financial transactions, assigns fraud risk scores, explains risky behaviour, and generates investigation reports.

## What It Does

- Upload transaction CSV files
- Predict fraud probability
- Classify transactions as Low, Medium, or High risk
- Display suspicious transactions in a dashboard
- Explain why transactions may be risky
- Generate an investigation summary
- Export suspicious transactions and reports

## Tech Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- FastAPI
- Streamlit
- Joblib
- Docker
- GitHub
- Hugging Face Spaces

## Machine Learning Approach

The model is trained on a public credit card fraud dataset. Because fraud is rare, the project handles class imbalance and evaluates performance using:

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## Architecture

```text
User Uploads CSV
        ↓
Streamlit Frontend
        ↓
Fraud Detection Model
        ↓
Risk Scores + Explanations
        ↓
Dashboard + Report Export