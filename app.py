import joblib
import pandas as pd
import streamlit as st
from datetime import datetime

model = joblib.load("models/fraud_model.joblib")

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide"
)

def assign_risk_level(probability):
    if probability >= 0.80:
        return "High"
    elif probability >= 0.40:
        return "Medium"
    else:
        return "Low"

def explain_transaction(row):
    reasons = []

    if "Amount" in row and row["Amount"] > 200:
        reasons.append("High transaction amount")

    if "Time" in row and row["Time"] < 21600:
        reasons.append("Unusual early-hour transaction timing")

    suspicious_features = []

    for col in row.index:
        if col.startswith("V"):
            try:
                if abs(float(row[col])) > 3:
                    suspicious_features.append(col)
            except Exception:
                pass

    if suspicious_features:
        reasons.append("Unusual anonymised behaviour pattern in: " + ", ".join(suspicious_features[:5]))

    if not reasons:
        reasons.append("No obvious rule-based trigger found; model score is based on learned fraud patterns")

    return "; ".join(reasons)

st.title("🛡️ FraudGuard AI")
st.caption("AI Fraud Investigation Copilot")

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    if st.button("Run Fraud Detection"):
        required_columns = model.feature_names_in_
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error("CSV is missing required columns.")
            st.write(missing_columns)
        else:
            X = df[required_columns]

            probabilities = model.predict_proba(X)[:, 1]
            predictions = model.predict(X)

            results = df.copy()
            results["fraud_probability"] = probabilities
            results["predicted_fraud"] = predictions
            results["risk_level"] = results["fraud_probability"].apply(assign_risk_level)
            results["risk_explanation"] = results.apply(explain_transaction, axis=1)

            st.success("Fraud analysis completed.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Transactions", len(results))
            col2.metric("High Risk", int((results["risk_level"] == "High").sum()))
            col3.metric("Predicted Fraud", int(results["predicted_fraud"].sum()))

            st.subheader("Risk Distribution")
            st.bar_chart(results["risk_level"].value_counts())

            st.subheader("Top Suspicious Transactions")
            top_suspicious = results.sort_values("fraud_probability", ascending=False).head(10)
            st.dataframe(top_suspicious, use_container_width=True)

            summary = f"""
FraudGuard AI Investigation Report
Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Total transactions analysed: {len(results)}
High-risk transactions: {int((results["risk_level"] == "High").sum())}
Medium-risk transactions: {int((results["risk_level"] == "Medium").sum())}
Low-risk transactions: {int((results["risk_level"] == "Low").sum())}
Predicted fraud count: {int(results["predicted_fraud"].sum())}
Average fraud probability: {results["fraud_probability"].mean():.4f}

Recommended next steps:
1. Review high-risk transactions manually.
2. Verify customer identity.
3. Check recent transaction history.
4. Escalate confirmed suspicious cases.
"""

            st.subheader("Investigation Summary")
            st.text_area("Report", summary, height=250)

            st.download_button(
                "Download Suspicious Transactions CSV",
                data=top_suspicious.to_csv(index=False),
                file_name="fraudguard_suspicious_transactions.csv",
                mime="text/csv"
            )

            st.download_button(
                "Download Investigation Report TXT",
                data=summary,
                file_name="fraudguard_report.txt",
                mime="text/plain"
            )
else:
    st.info("Upload a CSV file to begin.")