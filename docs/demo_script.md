# FraudGuard AI Demo Script

Hi, I’m Aditya. This is FraudGuard AI, a fraud investigation copilot built with Python, machine learning, and Streamlit.

The problem is that fraud teams often deal with large volumes of transaction data and need a fast way to identify suspicious activity.

In this demo, I upload a transaction dataset. The system scores each transaction using a trained machine learning model and assigns a fraud probability.

The dashboard shows total transactions, high-risk transactions, predicted fraud cases, and a risk distribution chart.

I can inspect the most suspicious transactions and view explanations showing why each case may require investigation.

The system also generates an investigation summary and allows the user to export suspicious transactions and a report.

Technically, the project uses Pandas for data processing, Scikit-learn and XGBoost for modelling, Joblib for model persistence, and Streamlit for the deployed web interface.

This project demonstrates applied machine learning, model deployment, explainability, and practical product thinking.