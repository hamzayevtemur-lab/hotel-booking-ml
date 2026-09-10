from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import load_raw_data, clean_data


## Configuration
TARGET="is_canceled"


# Feature definitions
# ================================

NUMERICAL_FEATURES = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

CATEGORICAL_FEATURES = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",  
    "deposit_type",
    "agent",
    "company",
    "customer_type",
]


FEATURES=NUMERICAL_FEATURES+CATEGORICAL_FEATURES



### Build preprocessing pipeline
def build_preprocessor():
    numerical_pipeline=Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy='median'),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )
    
    categorical_pipeline=Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy='most_frequent'),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )
    
    preprocessor=ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    
    return preprocessor



### Test preprocessing
# ========================

if __name__=="__main__":
    print("=" * 70)
    print("HOTEL BOOKING ML — PREPROCESSING PIPELINE")
    print("=" * 70)
    
    ### Load and clean data
    df=clean_data(
        load_raw_data()
    )
    print(f"\nCleaned dataset: {df.shape}")
    
    
    # Separate X and y
    # ---------------------------
    X = df[FEATURES]
    y = df[TARGET]

    print("\nInput shape:")
    print(X.shape)
    
    
    # Build preprocessor
    # ---------------------------------
    preprocessor = build_preprocessor()
    
    
    # Fit and transform------------------------
    X_transformed = preprocessor.fit_transform(X)

    print("\nTransformed shape:")
    print(X_transformed.shape)

    print("\nInput features:")
    print(len(FEATURES))

    print("\nNumerical features:")
    print(len(NUMERICAL_FEATURES))

    print("\nCategorical features:")
    print(len(CATEGORICAL_FEATURES))

    print("\nTransformed data type:")
    print(type(X_transformed))


    