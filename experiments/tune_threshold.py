import os
import json
import joblib

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from src.split_data import load_and_split_data


MODEL_PATH = "models/voting_classifier.joblib"
THRESHOLD_PATH = "reports/best_threshold.json"
GRAPH_PATH = "reports/figures/threshold_metrics.png"


print("=" * 70)
print("HOTEL BOOKING ML — THRESHOLD TUNING")
print("=" * 70)


# Load the same train, validation, and test split used by the project.
X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()

print("\nDataset split:")
print(f"Train      : {len(X_train):,}")
print(f"Validation : {len(X_val):,}")
print(f"Test       : {len(X_test):,}")


# Load the already trained Voting Classifier.
print("\nLoading trained Voting Classifier...")

model = joblib.load(MODEL_PATH)

print(f"Model loaded from: {MODEL_PATH}")


# Generate probabilities using only the validation set.
print("\nGenerating validation probabilities...")

y_probability = model.predict_proba(X_val)[:, 1]

print("Probabilities generated.")


# Test a range of classification thresholds.
thresholds = [
    round(value, 2)
    for value in [
        0.10 + i * 0.01
        for i in range(81)
    ]
]


results = []

print("\n" + "=" * 70)
print("THRESHOLD COMPARISON")
print("=" * 70)


for threshold in thresholds:

    # Convert probabilities into class predictions.
    y_pred = (
        y_probability >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_val,
        y_pred,
    )

    precision = precision_score(
        y_val,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        y_pred,
        zero_division=0,
    )

    results.append(
        {
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )


# Find the threshold with the highest F1 score.
best_result = max(
    results,
    key=lambda result: result["f1"],
)

best_threshold = best_result["threshold"]


print("\n" + "=" * 70)
print("BEST THRESHOLD")
print("=" * 70)

print(f"\nBest threshold: {best_threshold:.2f}")
print(f"Best F1 Score : {best_result['f1']:.4f}")

print("\nBest threshold validation results:")

print(f"Accuracy : {best_result['accuracy']:.4f}")
print(f"Precision: {best_result['precision']:.4f}")
print(f"Recall   : {best_result['recall']:.4f}")
print(f"F1 Score : {best_result['f1']:.4f}")

# ROC-AUC and PR-AUC do not depend on the classification threshold.
roc_auc = roc_auc_score(
    y_val,
    y_probability,
)

pr_auc = average_precision_score(
    y_val,
    y_probability,
)

print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")


# Create the reports/figures directory.
os.makedirs(
    "reports/figures",
    exist_ok=True,
)


print("\n" + "=" * 70)
print("CREATING THRESHOLD METRICS GRAPH")
print("=" * 70)


threshold_values = [
    result["threshold"]
    for result in results
]

precision_values = [
    result["precision"]
    for result in results
]

recall_values = [
    result["recall"]
    for result in results
]

f1_values = [
    result["f1"]
    for result in results
]


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    threshold_values,
    precision_values,
    label="Precision",
)

plt.plot(
    threshold_values,
    recall_values,
    label="Recall",
)

plt.plot(
    threshold_values,
    f1_values,
    label="F1 Score",
)

# Mark the best F1 threshold.
plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Best Threshold = {best_threshold:.2f}",
)

plt.scatter(
    best_threshold,
    best_result["f1"],
)

plt.annotate(
    f"Best F1 = {best_result['f1']:.4f}",
    (
        best_threshold,
        best_result["f1"],
    ),
    xytext=(10, 10),
    textcoords="offset points",
)

plt.title("Precision, Recall and F1 Score vs Classification Threshold")
plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.xlim(0.10,0.90,)
plt.ylim(0.0,1.0,)
plt.grid(True,alpha=0.3,)

plt.legend()
plt.tight_layout()

plt.savefig(GRAPH_PATH,dpi=150,)
plt.close()

print(f"Saved: {GRAPH_PATH}")


# Save the selected threshold.
threshold_data = {
    "threshold": best_threshold,
    "selection_metric": "f1",
    "validation_f1": best_result["f1"],
    "validation_accuracy": best_result["accuracy"],
    "validation_precision": best_result["precision"],
    "validation_recall": best_result["recall"],
    "validation_roc_auc": roc_auc,
    "validation_pr_auc": pr_auc,
}


with open(
    THRESHOLD_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        threshold_data,
        f,
        indent=4,
    )


print("\n" + "=" * 70)
print("THRESHOLD SAVED")
print("=" * 70)

print(f"\nSaved: {THRESHOLD_PATH}")

print("\nThreshold tuning complete.")