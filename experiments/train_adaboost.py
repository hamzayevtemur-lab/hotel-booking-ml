import os
import json

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from src.split_data import load_and_split_data
from src.model_pipeline import build_preprocessor
from src.ml_utils import train_model, evaluate_model, save_model
from src.experiment_tracker import save_experiment_result


RANDOM_STATE = 42
MODEL_PATH = "models/adaboost_classifier.joblib"
PARAMS_PATH = "reports/tuning/adaboost_best_params.json"


print("=" * 70)
print("HOTEL BOOKING ML — ADABOOST (GRID SEARCH)")
print("=" * 70)


# Load train, validation, and test data.
X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()

print("\nDataset split:")
print(f"Train      : {len(X_train):,}")
print(f"Validation : {len(X_val):,}")
print(f"Test       : {len(X_test):,}")


# Build preprocessing pipeline.
preprocessor = build_preprocessor()


# Base estimator and model.
base_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)

adaboost_model = AdaBoostClassifier(
    estimator=base_tree,
    random_state=RANDOM_STATE,
)


# Complete pipeline.
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", adaboost_model),
    ]
)


# Grid of hyperparameters to search.
# The previous fixed setup (max_depth=2, learning_rate=0.05) collapsed to
# predicting a single class, so this grid deliberately covers a wider,
# stronger range of stump depth and learning rate.
param_grid = {
    "model__n_estimators": [50, 100],
    "model__learning_rate": [0.05, 0.1],
    "model__estimator__max_depth": [1, 2],
}


print("\n" + "=" * 70)
print("RUNNING GRID SEARCH")
print("=" * 70)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=2,
)

training_time = train_model(
    grid_search,
    X_train,
    y_train,
)

best_pipeline = grid_search.best_estimator_

print("\nBest parameters found:")
print(grid_search.best_params_)

print(f"\nBest cross-validated F1: {grid_search.best_score_:.4f}")


# Evaluate the best pipeline on validation data.
results = evaluate_model(
    best_pipeline,
    X_val,
    y_val,
    training_time,
)

save_experiment_result(
    model_name="AdaBoost Classifier",
    results=results,
)


# Save the best hyperparameters for reference.
os.makedirs("reports/tuning", exist_ok=True)

with open(PARAMS_PATH, "w", encoding="utf-8") as f:
    json.dump(grid_search.best_params_, f, indent=4)

print(f"\nBest parameters saved to: {PARAMS_PATH}")


# Save the tuned pipeline.
os.makedirs("models", exist_ok=True)
save_model(
    best_pipeline,
    MODEL_PATH,
)


print("\n" + "=" * 70)
print("ADABOOST SUMMARY")
print("=" * 70)

print(f"\nAccuracy : {results['accuracy']:.4f}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall   : {results['recall']:.4f}")
print(f"F1 Score : {results['f1']:.4f}")
print(f"ROC-AUC  : {results['roc_auc']:.4f}")
print(f"PR-AUC   : {results['pr_auc']:.4f}")

print("\nModel saved successfully.")