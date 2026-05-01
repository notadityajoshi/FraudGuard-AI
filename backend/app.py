import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI(
    title="FraudGuard AI API",
    description="Backend API for fraud risk prediction and investigation support",
    version="1.0.0"
)

model = joblib.load("models/fraud_model.joblib")


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
            except:
                pass

    if suspicious_features:
        reasons.append(
            "Unusual anonymised behaviour pattern in features: "
            + ", ".join(suspicious_features[:5])
        )

    if not reasons:
        reasons.append("No obvious rule-based risk trigger found, model score is based on learned patterns")

    return "; ".join(reasons)


def generate_summary(results):
    total = len(results)
    high_risk = (results["risk_level"] == "High").sum()
    medium_risk = (results["risk_level"] == "Medium").sum()
    low_risk = (results["risk_level"] == "Low").sum()
    avg_risk = results["fraud_probability"].mean()

    summary = f"""
FraudGuard AI Investigation Summary

Total transactions analysed: {total}
High-risk transactions: {high_risk}
Medium-risk transactions: {medium_risk}
Low-risk transactions: {low_risk}
Average fraud probability: {avg_risk:.4f}

Key finding:
The system identified {high_risk} transactions requiring immediate fraud review.

Recommended action:
Prioritise high-risk transactions first, manually verify customer identity, review transaction context, and escalate cases with abnormal amount, timing, or anonymised behaviour patterns.
"""

    return summary.strip()


@app.get("/")
def home():
    return {"message": "FraudGuard AI backend is running"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        required_columns = model.feature_names_in_
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Missing required columns",
                    "missing_columns": missing_columns
                }
            )

        X = df[required_columns]

        fraud_probabilities = model.predict_proba(X)[:, 1]
        fraud_predictions = model.predict(X)

        results = df.copy()
        results["fraud_probability"] = fraud_probabilities
        results["predicted_fraud"] = fraud_predictions
        results["risk_level"] = results["fraud_probability"].apply(assign_risk_level)
        results["risk_explanation"] = results.apply(explain_transaction, axis=1)

        summary = generate_summary(results)

        return {
            "total_transactions": len(results),
            "high_risk_transactions": int((results["risk_level"] == "High").sum()),
            "predicted_fraud_count": int(results["predicted_fraud"].sum()),
            "investigation_summary": summary,
            "results": results.head(100).to_dict(orient="records")
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )