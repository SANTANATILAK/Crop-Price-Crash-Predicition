import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

df = pd.read_csv("crop_price_ml.csv")

df["price_date"] = pd.to_datetime(df["price_date"])

df = df.sort_values("price_date").reset_index(drop=True)

features = [
    "modal_price",
    "price_lag_1",
    "price_lag_3",
    "price_lag_7",
    "price_lag_14",
    "price_lag_30",
    "price_change_3d",
    "price_change_7d",
    "price_change_14d",
    "price_change_30d",
    "rolling_mean_7d",
    "rolling_mean_30d",
    "rolling_std_7d",
    "rolling_std_30d",
    "price_vs_ma7",
    "price_vs_ma30",
    "month",
    "day_of_week"
]

X = df[features]
y = df["crash"]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, probabilities)

print("\nROC-AUC:", round(roc_auc, 4))

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]

results = []

print("\nThreshold comparison")
print("-" * 75)

for threshold in thresholds:

    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append([
        threshold,
        accuracy,
        precision,
        recall,
        f1
    ])

    print(
        "Threshold:",
        threshold,
        "| Accuracy:",
        round(accuracy, 4),
        "| Precision:",
        round(precision, 4),
        "| Recall:",
        round(recall, 4),
        "| F1:",
        round(f1, 4)
    )

results_df = pd.DataFrame(
    results,
    columns=[
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]
)

best_row = results_df.loc[
    results_df["F1"].idxmax()
]

best_threshold = float(best_row["Threshold"])

print("\n" + "=" * 70)
print("BEST THRESHOLD")
print("=" * 70)

print(best_row)

final_predictions = (
    probabilities >= best_threshold
).astype(int)

print("\nFinal Classification Results")

print(
    "Accuracy :",
    round(accuracy_score(y_test, final_predictions), 4)
)

print(
    "Precision:",
    round(precision_score(y_test, final_predictions, zero_division=0), 4)
)

print(
    "Recall   :",
    round(recall_score(y_test, final_predictions, zero_division=0), 4)
)

print(
    "F1 Score :",
    round(f1_score(y_test, final_predictions, zero_division=0), 4)
)

results_df.to_csv(
    "threshold_comparison.csv",
    index=False
)

joblib.dump(
    model,
    "crop_crash_model.pkl"
)

joblib.dump(
    features,
    "model_features.pkl"
)

joblib.dump(
    best_threshold,
    "crash_threshold.pkl"
)

print("\nSaved files:")
print("crop_crash_model.pkl")
print("model_features.pkl")
print("crash_threshold.pkl")
print("threshold_comparison.csv")