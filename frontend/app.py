import requests
import pandas as pd
import streamlit as st
from datetime import datetime

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ FraudGuard AI")
st.caption("AI Fraud Investigation Copilot")

st.markdown("""
Upload transaction data and FraudGuard AI will score each transaction for fraud risk,
explain suspicious cases, generate an investigation summary, and export reports.
""")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if "results_df" not in st.session_state:
    st.session_state.results_df = None

if "summary" not in st.session_state:
    st.session_state.summary = None

if uploaded_file is not None:
    df_preview = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df_preview.head(10), use_container_width=True)

    if st.button("Run Fraud Detection"):
        with st.spinner("Analyzing transactions..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "text/csv"
                )
            }

            response = requests.post(f"{API_URL}/predict", files=files)

        if response.status_code == 200:
            data = response.json()

            st.session_state.results_df = pd.DataFrame(data["results"])
            st.session_state.summary = data["investigation_summary"]

            st.success("Fraud analysis completed.")
        else:
            st.error("Something went wrong.")
            st.json(response.json())

if st.session_state.results_df is not None:
    results_df = st.session_state.results_df

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", len(results_df))
    col2.metric(
        "High Risk Transactions",
        int((results_df["risk_level"] == "High").sum())
    )
    col3.metric(
        "Predicted Fraud Count",
        int(results_df["predicted_fraud"].sum())
    )

    st.subheader("Risk Level Distribution")
    st.bar_chart(results_df["risk_level"].value_counts())

    st.subheader("Top Suspicious Transactions")

    top_suspicious = results_df.sort_values(
        by="fraud_probability",
        ascending=False
    ).head(10)

    st.dataframe(top_suspicious, use_container_width=True)

    st.subheader("Transaction Risk Explanations")

    selected_index = st.selectbox(
        "Select a transaction row to investigate",
        results_df.index.tolist()
    )

    selected_transaction = results_df.loc[selected_index]

    st.write("Fraud probability:", selected_transaction["fraud_probability"])
    st.write("Risk level:", selected_transaction["risk_level"])
    st.write("Explanation:", selected_transaction["risk_explanation"])

    st.subheader("AI Investigation Summary")
    st.text_area(
        "Generated summary",
        st.session_state.summary,
        height=220
    )

    st.subheader("Simple Copilot Chat")

    user_question = st.text_input(
        "Ask about this batch, example: how many high risk transactions?"
    )

    if user_question:
        question = user_question.lower()

        if "high" in question:
            answer = f"There are {int((results_df['risk_level'] == 'High').sum())} high-risk transactions."
        elif "medium" in question:
            answer = f"There are {int((results_df['risk_level'] == 'Medium').sum())} medium-risk transactions."
        elif "low" in question:
            answer = f"There are {int((results_df['risk_level'] == 'Low').sum())} low-risk transactions."
        elif "average" in question or "mean" in question:
            answer = f"The average fraud probability is {results_df['fraud_probability'].mean():.4f}."
        elif "top" in question or "suspicious" in question:
            top_row = results_df.sort_values(by="fraud_probability", ascending=False).iloc[0]
            answer = f"The most suspicious transaction has fraud probability {top_row['fraud_probability']:.4f} and risk level {top_row['risk_level']}."
        elif "summary" in question:
            answer = st.session_state.summary
        else:
            answer = "I can answer basic questions about high, medium, low risk counts, average fraud probability, top suspicious transaction, and summary."

        st.info(answer)

    st.subheader("Export Investigation Outputs")

    suspicious_df = results_df[
        results_df["risk_level"].isin(["High", "Medium"])
    ].sort_values(by="fraud_probability", ascending=False)

    suspicious_csv = suspicious_df.to_csv(index=False)

    st.download_button(
        label="Download Suspicious Transactions CSV",
        data=suspicious_csv,
        file_name="fraudguard_suspicious_transactions.csv",
        mime="text/csv"
    )

    report_text = f"""
FraudGuard AI Investigation Report
Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

========================
EXECUTIVE SUMMARY
========================

{st.session_state.summary}

========================
KEY METRICS
========================

Total transactions analysed: {len(results_df)}
High-risk transactions: {int((results_df["risk_level"] == "High").sum())}
Medium-risk transactions: {int((results_df["risk_level"] == "Medium").sum())}
Low-risk transactions: {int((results_df["risk_level"] == "Low").sum())}
Predicted fraud count: {int(results_df["predicted_fraud"].sum())}
Average fraud probability: {results_df["fraud_probability"].mean():.4f}

========================
TOP SUSPICIOUS TRANSACTIONS
========================

{top_suspicious[["fraud_probability", "risk_level", "risk_explanation"]].to_string(index=False)}

========================
RECOMMENDED NEXT STEPS
========================

1. Review high-risk transactions manually.
2. Verify customer identity and transaction context.
3. Check whether similar transactions occurred recently.
4. Escalate confirmed suspicious cases to the fraud investigation team.
5. Keep model predictions as decision support, not as final legal proof.
"""

    st.download_button(
        label="Download Investigation Report TXT",
        data=report_text,
        file_name="fraudguard_investigation_report.txt",
        mime="text/plain"
    )

else:
    st.info("Upload a CSV file to begin.")