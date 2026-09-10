from src.preprocessing import load_raw_data, clean_data

TARGET="is_canceled"


### Load cleaned data
df=clean_data(load_raw_data())

print("=" * 70)
print("HOTEL BOOKING ML — FEATURE ANALYSIS")
print("=" * 70)


### Feature Categories
numerical_features = [
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
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

categorical_features = [
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

# Check feature coverage
# ====================================

all_features = numerical_features + categorical_features

print("\nFeature counts:")

print(f"Numerical   : {len(numerical_features)}")
print(f"Categorical : {len(categorical_features)}")
print(f"Total       : {len(all_features)}")

print("\nFeature coverage:")
missing_from_analysis = set(df.columns) - set(all_features) - {TARGET}
extra_in_analysis = set(all_features) - set(df.columns)

print("Missing from analysis:", missing_from_analysis)
print("Not present in dataframe:", extra_in_analysis)

# Numerical features
# ====================================
print("\n" + "=" * 70)
print("NUMERICAL FEATURES")
print("=" * 70)

for column in numerical_features:
    print(
        f"{column:<40}"
        f"dtype={str(df[column].dtype):<10}"
        f"unique={df[column].nunique():>5}"
    )


# Categorical features
# ===================================
print("\n" + "=" * 70)
print("CATEGORICAL FEATURES")
print("=" * 70)

for column in categorical_features:
    print(
        f"{column:<40}"
        f"unique={df[column].nunique():>5}"
    )
    
    
# Cardinality
# ================================
print("\n" + "=" * 70)
print("CATEGORICAL CARDINALITY")
print("=" * 70)

cardinality = (
    df[categorical_features]
    .nunique()
    .sort_values(ascending=False)
)

print(cardinality)


# Missing values
# ===================================
print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = (
    df[all_features]
    .isna()
    .sum()
    .sort_values(ascending=False)
)

print(missing[missing > 0])


# Target relationship for numerical features
# =====================================
print("\n" + "=" * 70)
print("NUMERICAL FEATURES — TARGET MEANS")
print("=" * 70)

for column in numerical_features:
    result = (
        df.groupby(TARGET)[column]
        .mean()
        .round(3)
    )

    print(f"\n{column}")
    print(result)
    
    
# Target relationship for categorical features
# ====================================
print("\n" + "=" * 70)
print("CATEGORICAL FEATURES — CANCELLATION RATE")
print("=" * 70)

for column in categorical_features:
    cancellation_rate = (
        df.groupby(column, dropna=False)[TARGET]
        .mean()
        .sort_values(ascending=False)
    )

    print(f"\n--- {column} ---")
    
    print(
        cancellation_rate
        .head(15)
        .round(3)
    )
