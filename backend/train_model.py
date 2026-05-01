import json
import joblib
import openml
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


print("Loading fraud dataset...")

dataset = openml.datasets.get_dataset(1597)
X, y, _, _ = dataset.get_data(target="Class")

y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training model...")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("classifier", XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    ))
])

model.fit(X_train, y_train)

print("Model trained.")

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "roc_auc": float(roc_auc_score(y_test, y_proba)),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "classification_report": classification_report(y_test, y_pred, output_dict=True)
}

joblib.dump(model, "models/fraud_model.joblib")

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

sample_data = X_test.head(20).copy()
sample_data["actual_fraud"] = y_test.head(20).values
sample_data["fraud_probability"] = y_proba[:20]
sample_data["predicted_fraud"] = y_pred[:20]

sample_data.to_csv("data/sample_predictions.csv", index=False)

print("Saved model to models/fraud_model.joblib")
print("Saved metrics to models/metrics.json")
print("Saved sample predictions to data/sample_predictions.csv")
print("ROC-AUC:", metrics["roc_auc"])