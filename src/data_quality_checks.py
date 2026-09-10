from pathlib import Path
import pandas as pd

### Configuration
# =====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "hotel_bookings.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("HOTEL BOOKING ML — DATA QUALITY CHECKS")
print("=" * 70)

## Missing children
print("\n" + "=" * 70)
print("1. MISSING CHILDREN VALUES")
print("=" * 70)

missing_children=df[df["children"].isna()]

print(f"Rows with missing children:{len(missing_children)}")
if len(missing_children)>0:
    print(
        missing_children[
            [
                "hotel",
                "adults",
                "children",
                "babies",
                "meal",
                "is_canceled",
            ]
        ]
    )
    

## Zero guests
print("\n" + "=" * 70)
print("2. BOOKINGS WITH ZERO GUESTS")
print("=" * 70)

zero_guests=df[
    (df["adults"]==0)
    & (df["children"].fillna(0)==0)
    & (df["babies"]==0)
]

print(f"Rows with zero guests: {len(zero_guests)}")

if len(zero_guests)>0:
    print(
        zero_guests[
            [
                "hotel",
                "is_canceled",
                "adults",
                "children",
                "babies",
                "meal",
                "market_segment",
                "customer_type",
            ]
        ].head(20)
    )
    
    
### Extreme ADR
print("\n" + "=" * 70)
print("3. ADR ANALYSIS")
print("=" * 70)

print("ADR percentiles:")

print(
    df["adr"].quantile(
        [0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0]
    )
)

print("\nTop 20 ADR values:")
print(
    df[
        [
            "hotel",
            "is_canceled",
            "adults",
            "children",
            "babies",
            "reserved_room_type",
            "adr",
        ]
    ]
    .sort_values("adr", ascending=False)
    .head(20)
)


### Negative / suspicious numerical values

print("\n" + "=" * 70)
print("4. SUSPICIOUS NUMERICAL VALUES")
print("=" * 70)

numerical_columns = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

for column in numerical_columns:
    negative_count=(df[column]<0).sum()
    
    if negative_count>0:
        print(f"{column}:{negative_count} negative values")
        
        

### Duplicate analysis
print("\n" + "=" * 70)
print("5. DUPLICATE ANALYSIS")
print("=" * 70)

duplicate_mask=df.duplicated(keep=False)

duplicates=df[duplicate_mask].copy()

print(f"Rows belonging to duplicate groups: {len(duplicates)}")

if not duplicates.empty:
    print("\nTarget distribution among duplicate rows:")
    
    print(
        duplicates["is_canceled"]
        .value_counts()
        .sort_index()
    )
    print("\nDuplicate rows by hotel:")
    
    print(
        duplicates['hotel']
        .value_counts()
    )
    print("\nNumber of duplicate groups:")
    print(
        duplicates.groupby(list(df.columns)).size()
        .value_counts()
        .sort_index()
        .head(20)
    )
    

### Duplicate groups with conflicting targets
print("\n" + "=" * 70)
print("6. DUPLICATES WITH CONFLICTING TARGETS")
print("=" * 70)

feature_columns=[
    column
    for column in df.columns
    if column !="is_canceled"
]

duplicate_target_conflicts=(
    df.groupby(feature_columns, dropna=False)["is_canceled"]
    .nunique()
)

conflicting_groups = duplicate_target_conflicts[
    duplicate_target_conflicts > 1
]

print(
    f"Duplicate feature combinations with conflicting targets: "
    f"{len(conflicting_groups):,}"
)

### Agent analysis
print("\n" + "=" * 70)
print("7. AGENT ANALYSIS")
print("=" * 70)

print(f"Missing agent: {df['agent'].isna().sum():,}")
print(f"Unique agents: {df['agent'].nunique(dropna=True):,}")

print("\nMost common agents:")

print(
    df["agent"]
    .value_counts(dropna=False)
    .head(15)
)


### Company analysis
print("\n" + "=" * 70)
print("8. COMPANY ANALYSIS")
print("=" * 70)

print(f"Missing company: {df['company'].isna().sum():,}")
print(f"Unique companies: {df['company'].nunique(dropna=True):,}")

print("\nMost common companies:")

print(
    df["company"]
    .value_counts(dropna=False)
    .head(15)
)


### Potential leakage columns
print("\n" + "=" * 70)
print("9. POTENTIAL LEAKAGE FEATURES")
print("=" * 70)

leakage_candidates = [
    "reservation_status",
    "reservation_status_date",
]

for column in leakage_candidates:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False).head(10))
    
    
### Target relationship with suspicious columns
print("\n" + "=" * 70)
print("10. TARGET VS RESERVATION STATUS")
print("=" * 70)

print(
    pd.crosstab(
        df["reservation_status"],
        df["is_canceled"],
        margins=True
    )
)


# Inspect anomalous rows

print("\n" + "=" * 70)
print("11. ANOMALOUS ROWS")
print("=" * 70)

print("\n--- Negative ADR ---")

print(
    df[df["adr"] < 0].T
)

print("\n--- ADR = 5400 ---")

print(
    df[df["adr"] == 5400].T
)

print("\n--- Zero guest bookings: target distribution ---")

print(
    zero_guests["is_canceled"]
    .value_counts()
)

print("\n--- Zero guest bookings: hotel distribution ---")

print(
    zero_guests["hotel"]
    .value_counts()
)

print("\n--- Missing children rows ---")

print(
    missing_children.T
)