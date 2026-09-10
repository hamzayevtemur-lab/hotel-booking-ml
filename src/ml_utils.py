import time
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


def train_model(model, X_train, y_train):

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print(f"\nTraining time: {training_time:.3f} seconds")

    return training_time


def evaluate_model(model, X_val, y_val, training_time=None):
   
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    # Prediction
    start_time = time.time()

    y_pred = model.predict(X_val)
    y_probability = model.predict_proba(X_val)[:, 1]

    prediction_time = time.time() - start_time

    # Metrics
    accuracy = accuracy_score(y_val, y_pred)

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

    roc_auc = roc_auc_score(
        y_val,
        y_probability,
    )

    pr_auc = average_precision_score(
        y_val,
        y_probability,
    )

    # Print metrics
    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    if training_time is not None:
        print(f"\nTraining time  : {training_time:.3f} seconds")

    print(f"Prediction time: {prediction_time:.3f} seconds")

    # Confusion Matrix
    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_val,
            y_pred,
        )
    )

    # Classification Report
    print("\nClassification Report:")

    print(
        classification_report(
            y_val,
            y_pred,
            digits=4,
        )
    )

    # Return results
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "training_time": training_time,
        "prediction_time": prediction_time,
    }

    return results


def save_model(model, path):
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    joblib.dump(
        model,
        path,
    )

    print(f"\nModel saved to: {path}")