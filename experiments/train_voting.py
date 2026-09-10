import os
import json

from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from src.split_data import load_and_split_data
from src.model_pipeline import build_preprocessor
from src.ml_utils import train_model, evaluate_model, save_model
from src.experiment_tracker import save_experiment_result


RANDOM_STATE = 42

MODEL_PATH = "models/voting_classifier.joblib"
PARAMS_PATH = "reports/tuning/voting_best_params.json"


print("=" * 70)
print("HOTEL BOOKING ML — VOTING CLASSIFIER (GRID SEARCH)")
print("=" * 70)


# Load train, validation, and test data.
X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()

print("\nDataset split:")
print(f"Train      : {len(X_train):,}")
print(f"Validation : {len(X_val):,}")
print(f"Test       : {len(X_test):,}")


# Build preprocessing pipeline.
preprocessor = build_preprocessor()


# Base model 1: Logistic Regression.
logistic = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE,
)


# Base model 2: Extra Trees.
extra_trees = ExtraTreesClassifier(
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# Base model 3: K-Nearest Neighbors.
knn = KNeighborsClassifier(
    n_jobs=-1,
)


# Voting classifier.
#
# Soft voting combines the probability predictions
# produced by the three base models.
voting_model = VotingClassifier(
    estimators=[
        ("logistic", logistic),
        ("extra_trees", extra_trees),
        ("knn", knn),
    ],
    voting="soft",
)


# Complete preprocessing + model pipeline.
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", voting_model),
    ]
)


# Hyperparameter grid.
#
# We keep this reasonably small because each combination
# is evaluated using cross-validation.
param_grid = {
    "model__logistic__C": [0.1, 1.0],
    "model__extra_trees__n_estimators": [100, 200],
    "model__knn__n_neighbors": [10, 20],
}


print("\n" + "=" * 70)
print("RUNNING GRID SEARCH")
print("=" * 70)

print("\nScoring metric: F1")
print("Cross-validation: 3 folds")


grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=2,
)


# Train GridSearchCV.
training_time = train_model(
    grid_search,
    X_train,
    y_train,
)


# Extract the best complete pipeline.
best_pipeline = grid_search.best_estimator_


print("\n" + "=" * 70)
print("BEST PARAMETERS")
print("=" * 70)

print("\nBest parameters found:")

for parameter, value in grid_search.best_params_.items():
    print(f"{parameter}: {value}")

print(
    f"\nBest cross-validated F1: "
    f"{grid_search.best_score_:.4f}"
)


# Evaluate the best configuration on the validation set.
results = evaluate_model(
    best_pipeline,
    X_val,
    y_val,
    training_time,
)


# Save validation results.
save_experiment_result(
    model_name="Voting Classifier",
    results=results,
)


# Save best parameters.
os.makedirs(
    "reports/tuning",
    exist_ok=True,
)

with open(
    PARAMS_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        grid_search.best_params_,
        f,
        indent=4,
    )


print(
    f"\nBest parameters saved to: "
    f"{PARAMS_PATH}"
)


# Save the complete best pipeline.
os.makedirs(
    "models",
    exist_ok=True,
)

save_model(
    best_pipeline,
    MODEL_PATH,
)


# Final summary.
print("\n" + "=" * 70)
print("VOTING CLASSIFIER SUMMARY")
print("=" * 70)

print(f"\nAccuracy : {results['accuracy']:.4f}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall   : {results['recall']:.4f}")
print(f"F1 Score : {results['f1']:.4f}")
print(f"ROC-AUC  : {results['roc_auc']:.4f}")
print(f"PR-AUC   : {results['pr_auc']:.4f}")

print("\nModel saved successfully.")