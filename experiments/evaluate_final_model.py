import os
import json
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from src.split_data import load_and_split_data


MODEL_PATH = "models/voting_classifier.joblib"
THRESHOLD_PATH = "reports/best_threshold.json"

RESULTS_PATH = "reports/final_test_results.txt"
CONFUSION_MATRIX_DEFAULT_PATH = "reports/figures/final_confusion_matrix_default.png"
CONFUSION_MATRIX_TUNED_PATH = "reports/figures/final_confusion_matrix_tuned.png"
THRESHOLD_COMPARISON_PATH = "reports/figures/final_threshold_comparison.png"


print("=" * 70)
print("HOTEL BOOKING ML — FINAL MODEL EVALUATION")
print("=" * 70)


# Load the same train, validation, and test split used throughout the project.
print("\nLoading test data...")

X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()

print(f"\nTest samples: {len(X_test):,}")


# Load the trained Voting Classifier.
print("\nLoading trained Voting Classifier...")

model = joblib.load(MODEL_PATH)

print(f"Model loaded from: {MODEL_PATH}")


# Load the threshold selected using the validation set.
print("\nLoading tuned threshold...")

with open(THRESHOLD_PATH,"r",encoding="utf-8",) as f:
    threshold_data = json.load(f)

tuned_threshold = threshold_data["threshold"]

print(f"Default threshold: 0.50")
print(f"Tuned threshold   : {tuned_threshold:.2f}")


# Generate probabilities for the completely untouched test set.
print("\nGenerating test probabilities...")

y_probability = model.predict_proba(X_test)[:, 1]

print("Test probabilities generated.")


def calculate_metrics(y_true, y_probability, threshold):
    """
    Convert probabilities into predictions using a threshold
    and calculate classification metrics.
    """

    y_pred = (
        y_probability >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        y_probability,
    )

    pr_auc = average_precision_score(
        y_true,
        y_probability,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    report = classification_report(
        y_true,
        y_pred,
        digits=4,
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": matrix,
        "classification_report": report,
    }


# Evaluate using the normal 0.50 threshold.
default_results = calculate_metrics(
    y_test,
    y_probability,
    0.50,
)


# Evaluate using the threshold selected from validation data.
tuned_results = calculate_metrics(
    y_test,
    y_probability,
    tuned_threshold,
)


print("\n" + "=" * 70)
print("FINAL TEST RESULTS — DEFAULT THRESHOLD")
print("=" * 70)

print(f"\nThreshold: {default_results['threshold']:.2f}")

print(f"\nAccuracy : {default_results['accuracy']:.4f}")
print(f"Precision: {default_results['precision']:.4f}")
print(f"Recall   : {default_results['recall']:.4f}")
print(f"F1 Score : {default_results['f1']:.4f}")
print(f"ROC-AUC  : {default_results['roc_auc']:.4f}")
print(f"PR-AUC   : {default_results['pr_auc']:.4f}")

print("\nConfusion Matrix:")
print(default_results["confusion_matrix"])

print("\nClassification Report:")
print(default_results["classification_report"])


print("\n" + "=" * 70)
print("FINAL TEST RESULTS — TUNED THRESHOLD")
print("=" * 70)

print(f"\nThreshold: {tuned_results['threshold']:.2f}")

print(f"\nAccuracy : {tuned_results['accuracy']:.4f}")
print(f"Precision: {tuned_results['precision']:.4f}")
print(f"Recall   : {tuned_results['recall']:.4f}")
print(f"F1 Score : {tuned_results['f1']:.4f}")
print(f"ROC-AUC  : {tuned_results['roc_auc']:.4f}")
print(f"PR-AUC   : {tuned_results['pr_auc']:.4f}")

print("\nConfusion Matrix:")
print(tuned_results["confusion_matrix"])

print("\nClassification Report:")
print(tuned_results["classification_report"])


print("\n" + "=" * 70)
print("THRESHOLD COMPARISON")
print("=" * 70)

print(
    f"\n{'Metric':<15}"
    f"{'Threshold 0.50':>18}"
    f"{'Threshold 0.37':>18}"
)

print("-" * 51)

print(
    f"{'Accuracy':<15}"
    f"{default_results['accuracy']:>18.4f}"
    f"{tuned_results['accuracy']:>18.4f}"
)

print(
    f"{'Precision':<15}"
    f"{default_results['precision']:>18.4f}"
    f"{tuned_results['precision']:>18.4f}"
)

print(
    f"{'Recall':<15}"
    f"{default_results['recall']:>18.4f}"
    f"{tuned_results['recall']:>18.4f}"
)

print(
    f"{'F1 Score':<15}"
    f"{default_results['f1']:>18.4f}"
    f"{tuned_results['f1']:>18.4f}"
)

print(
    f"{'ROC-AUC':<15}"
    f"{default_results['roc_auc']:>18.4f}"
    f"{tuned_results['roc_auc']:>18.4f}"
)

print(
    f"{'PR-AUC':<15}"
    f"{default_results['pr_auc']:>18.4f}"
    f"{tuned_results['pr_auc']:>18.4f}"
)


# Calculate improvements.
f1_change = (
    tuned_results["f1"]
    - default_results["f1"]
)

recall_change = (
    tuned_results["recall"]
    - default_results["recall"]
)

accuracy_change = (
    tuned_results["accuracy"]
    - default_results["accuracy"]
)

precision_change = (
    tuned_results["precision"]
    - default_results["precision"]
)


print("\n" + "=" * 70)
print("METRIC CHANGES")
print("=" * 70)

print(f"\nAccuracy change : {accuracy_change:+.4f}")
print(f"Precision change: {precision_change:+.4f}")
print(f"Recall change   : {recall_change:+.4f}")
print(f"F1 change       : {f1_change:+.4f}")


# Create the reports directories.
os.makedirs(
    "reports/figures",
    exist_ok=True,
)


# Save confusion matrices as images.
import matplotlib.pyplot as plt


def save_confusion_matrix(matrix, title, path):
    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(matrix)

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(
        [0, 1],
        ["Not Cancelled", "Cancelled"],
    )

    plt.yticks(
        [0, 1],
        ["Not Cancelled", "Cancelled"],
    )

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center",
            )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()




print("\n" + "=" * 70)
print("CREATING THRESHOLD COMPARISON GRAPH")
print("=" * 70)

import matplotlib.pyplot as plt

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
]

default_values = [
    default_results["accuracy"],
    default_results["precision"],
    default_results["recall"],
    default_results["f1"],
]

tuned_values = [
    tuned_results["accuracy"],
    tuned_results["precision"],
    tuned_results["recall"],
    tuned_results["f1"],
]

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    default_values,
    width=width,
    label="Default Threshold (0.50)",
)

plt.bar(
    [i + width / 2 for i in x],
    tuned_values,
    width=width,
    label=f"Tuned Threshold ({tuned_threshold:.2f})",
)

plt.xticks(
    list(x),
    metrics,
)

plt.ylabel("Score")
plt.ylim(0, 1)

plt.title(
    "Final Test Performance: Default vs Tuned Threshold"
)

plt.legend()
plt.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    THRESHOLD_COMPARISON_PATH,
    dpi=300,
)

plt.close()

print(
    f"Saved: {THRESHOLD_COMPARISON_PATH}"
)



print("\n" + "=" * 70)
print("SAVING CONFUSION MATRICES")
print("=" * 70)

save_confusion_matrix(
    default_results["confusion_matrix"],
    "Final Test Confusion Matrix — Threshold 0.50",
    CONFUSION_MATRIX_DEFAULT_PATH,
)

save_confusion_matrix(
    tuned_results["confusion_matrix"],
    f"Final Test Confusion Matrix — Threshold {tuned_threshold:.2f}",
    CONFUSION_MATRIX_TUNED_PATH,
)

print(f"\nSaved: {CONFUSION_MATRIX_DEFAULT_PATH}")
print(f"Saved: {CONFUSION_MATRIX_TUNED_PATH}")


# Save final text report.
with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write("HOTEL BOOKING ML — FINAL TEST RESULTS\n")
    f.write("=" * 70)
    f.write("\n\n")

    f.write("MODEL: Voting Classifier\n")
    f.write(f"Default threshold: 0.50\n")
    f.write(f"Tuned threshold: {tuned_threshold:.2f}\n\n")

    f.write("DEFAULT THRESHOLD — 0.50\n")
    f.write("-" * 40)
    f.write("\n")

    f.write(f"Accuracy : {default_results['accuracy']:.4f}\n")
    f.write(f"Precision: {default_results['precision']:.4f}\n")
    f.write(f"Recall   : {default_results['recall']:.4f}\n")
    f.write(f"F1 Score : {default_results['f1']:.4f}\n")
    f.write(f"ROC-AUC  : {default_results['roc_auc']:.4f}\n")
    f.write(f"PR-AUC   : {default_results['pr_auc']:.4f}\n")

    f.write("\nConfusion Matrix:\n")
    f.write(str(default_results["confusion_matrix"]))

    f.write("\n\n")
    f.write("TUNED THRESHOLD\n")
    f.write("-" * 40)
    f.write("\n")

    f.write(f"Threshold: {tuned_threshold:.2f}\n")
    f.write(f"Accuracy : {tuned_results['accuracy']:.4f}\n")
    f.write(f"Precision: {tuned_results['precision']:.4f}\n")
    f.write(f"Recall   : {tuned_results['recall']:.4f}\n")
    f.write(f"F1 Score : {tuned_results['f1']:.4f}\n")
    f.write(f"ROC-AUC  : {tuned_results['roc_auc']:.4f}\n")
    f.write(f"PR-AUC   : {tuned_results['pr_auc']:.4f}\n")

    f.write("\nConfusion Matrix:\n")
    f.write(str(tuned_results["confusion_matrix"]))

    f.write("\n\n")
    f.write("METRIC CHANGES\n")
    f.write("-" * 40)
    f.write("\n")

    f.write(f"Accuracy change : {accuracy_change:+.4f}\n")
    f.write(f"Precision change: {precision_change:+.4f}\n")
    f.write(f"Recall change   : {recall_change:+.4f}\n")
    f.write(f"F1 change       : {f1_change:+.4f}\n")


print("\n" + "=" * 70)
print("SAVING FINAL TEST RESULTS")
print("=" * 70)

print(f"\nSaved: {RESULTS_PATH}")

print("\n" + "=" * 70)
print("FINAL EVALUATION COMPLETE")
print("=" * 70)