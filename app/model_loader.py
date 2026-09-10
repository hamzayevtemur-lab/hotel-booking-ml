"""
Model loader for the FastAPI app.

Tries to load the saved voting_classifier.joblib first.
If not found (e.g. on Render), retrains with the Colab best params.
"""

import sys
import os
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.split_data import load_and_split_data
from src.model_pipeline import build_preprocessor

MODEL_PATH = PROJECT_ROOT / "models" / "voting_classifier.joblib"
RANDOM_STATE = 42

_model = None


def get_model():
    """Return the trained pipeline, loading or training it once."""
    global _model
    if _model is not None:
        return _model

    if MODEL_PATH.exists():
        print(f"[model_loader] Loading saved model from: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        print("[model_loader] Model loaded successfully.")
    else:
        print("[model_loader] No saved model found — retraining with best params...")
        _model = _train_model()

    return _model


def _train_model():
    """Train Voting Classifier with the best params found on Colab."""
    X_train, _, _, y_train, _, _ = load_and_split_data()

    preprocessor = build_preprocessor()

    logistic = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    extra_trees = ExtraTreesClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    knn = KNeighborsClassifier(
        n_neighbors=10,
        n_jobs=-1,
    )

    voting_model = VotingClassifier(
        estimators=[
            ("logistic", logistic),
            ("extra_trees", extra_trees),
            ("knn", knn),
        ],
        voting="soft",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", voting_model),
        ]
    )

    print("[model_loader] Training...")
    pipeline.fit(X_train, y_train)
    print("[model_loader] Training complete.")

    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[model_loader] Model saved to: {MODEL_PATH}")

    return pipeline
