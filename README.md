# Hotel Booking Cancellation Predictor 🏨

An end-to-end machine learning project that predicts whether a hotel booking will be cancelled before arrival — enabling hotels to implement proactive overbooking strategies, optimise revenue, and reduce empty rooms.

---

## 🎯 Business Problem

Hotel cancellations cost the hospitality industry billions annually. Every undetected cancellation leaves a room empty with zero revenue. This project builds a classification model that predicts upcoming cancellations so hotel managers can act in advance.

**Key trade-off:** Missing a cancellation (False Negative) = lost revenue. Wrongly flagging a booking (False Positive) = unnecessary overbooking. Threshold tuning is used to find the optimal balance.

---

## 📊 Dataset

- **Source:** [Hotel Booking Demand Dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **Size:** 119,390 hotel bookings (City Hotel & Resort Hotel)
- **Target:** `is_canceled` — whether the booking was cancelled (binary classification)
- **Features:** 28 original features covering lead time, room type, country, deposit type, market segment, special requests, and more

---

## ⚙️ Methodology

### 1. Data Engineering
- Audited 119K+ records for missing values, data types, and distributions
- Removed bookings with **zero guests**, **negative ADR**, and exact **duplicate rows**
- Filled missing `children` values with `0` (semantic fill)
- Treated `agent` and `company` IDs as categorical features
- **Eliminated target-leaking columns** (`reservation_status`, `reservation_status_date`) that would cause artificially perfect accuracy

### 2. Preprocessing Pipeline
Built a leak-free `sklearn` `ColumnTransformer`:
- **Numerical features**: Median imputation → Standard Scaling
- **Categorical features**: Mode imputation → One-Hot Encoding (`handle_unknown='ignore'`)

### 3. Train / Validation / Test Split
Stratified split maintaining class proportions:
| Split | Size |
|---|---|
| Train | 61,060 (70%) |
| Validation | 13,084 (15%) |
| Test | 13,085 (15%) |

### 4. Model Experiments
Five models were trained and tuned with `GridSearchCV` (F1 scoring, 3-fold CV):

| Model | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| Logistic Regression | 0.6751 | 0.8699 | 0.6842 |
| AdaBoost | 0.2028 | 0.8292 | 0.6088 |
| Bagging Classifier | 0.7173 | 0.9135 | 0.8038 |
| **Voting Classifier** ⭐ | **0.6825** | **0.9037** | **0.7918** |
| Stacking Classifier | 0.6487 | 0.8828 | 0.7423 |

The **Voting Classifier** (soft voting of Logistic Regression + ExtraTrees + KNN) was selected as the final model for its strong generalisation and ensemble diversity.

### 5. Threshold Tuning
Lowering the classification threshold from `0.50` → `0.37` on the validation set:

| Metric | Default (0.50) | Tuned (0.37) | Change |
|---|---|---|---|
| Precision | 0.7525 | 0.6576 | -0.0949 |
| **Recall** | 0.6019 | **0.7807** | **+0.1788** ⬆️ |
| **F1 Score** | 0.6688 | **0.7139** | **+0.0451** ⬆️ |
| ROC-AUC | 0.8981 | 0.8981 | — |

Tuning recall means the model catches **78% of actual cancellations**, allowing revenue managers to act proactively.

---

## 📈 Final Test Results (Held-Out Set)

**Model:** Voting Classifier | **Threshold:** 0.37

| Metric | Score |
|---|---|
| Accuracy | 0.8277 |
| Precision | 0.6576 |
| Recall | **0.7807** |
| F1 Score | **0.7139** |
| ROC-AUC | **0.8981** |
| PR-AUC | **0.7792** |

---

## 📁 Project Structure

```
hotel-booking-ml/
│
├── data/
│   └── raw/
│       └── hotel_bookings.csv          # Original dataset
│
├── src/                                # Core library
│   ├── data_audit.py                   # Dataset profiling & EDA
│   ├── data_quality_checks.py          # Quality validation checks
│   ├── preprocessing.py                # Data cleaning functions
│   ├── model_pipeline.py               # Sklearn preprocessing pipeline
│   ├── split_data.py                   # Stratified train/val/test split
│   ├── ml_utils.py                     # Train, evaluate, save helpers
│   ├── experiment_tracker.py           # Logs results to CSV
│   └── feature_analysis.py             # Feature importance analysis
│
├── experiments/                        # Model training scripts
│   ├── train_logistic.py
│   ├── train_adaboost.py
│   ├── train_bagging.py
│   ├── train_voting.py
│   ├── train_stacking.py
│   ├── tune_threshold.py               # Precision-Recall threshold search
│   ├── compare_models.py               # Generates comparison charts
│   └── evaluate_final_model.py         # Final test set evaluation
│
├── models/                             # Saved trained models (.gitignored)
│
├── reports/
│   ├── experiment_results.csv          # All model validation metrics
│   ├── best_threshold.json             # Selected classification threshold
│   ├── final_test_results.txt          # Final evaluation output
│   ├── tuning/                         # Best hyperparameters per model
│   └── figures/                        # All generated charts
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
# Data audit
python -m src.data_audit

# Train individual models
python -m experiments.train_logistic
python -m experiments.train_adaboost
python -m experiments.train_bagging
python -m experiments.train_voting      # Computationally heavy — use Colab

# Compare all models
python -m experiments.compare_models

# Tune classification threshold
python -m experiments.tune_threshold

# Final evaluation on test set
python -m experiments.evaluate_final_model
```

> **Note:** Training the Voting and Stacking classifiers requires significant compute. It is recommended to run them on [Google Colab](https://colab.research.google.com/) using a GPU/high-RAM runtime.

---

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-red)

- **Python** — Core language
- **Pandas / NumPy** — Data manipulation
- **Scikit-Learn** — Preprocessing pipelines, model training, GridSearchCV
- **Matplotlib / Seaborn** — Visualisations
- **Joblib** — Model serialisation

---

## 💡 Key Learnings

- **Data leakage prevention** is critical — `reservation_status` had to be removed to avoid near-perfect but meaningless accuracy
- **Threshold tuning** is often more impactful than model selection for imbalanced classification tasks
- **Ensemble diversity** matters — Voting Classifier (combining tree, linear, and instance-based models) generalised better than Stacking despite stacking's theoretical advantage
- **PR-AUC** is more informative than ROC-AUC for imbalanced datasets (37% cancellation rate)
