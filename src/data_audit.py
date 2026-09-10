from pathlib import Path
import pandas as pd


### Configuration
# =========================
PROJECT_ROOT=Path(__file__).resolve().parent.parent
DATA_PATH=PROJECT_ROOT/"data"/"raw"/"hotel_bookings.csv"


### Load Data
# ========================
print("=" * 70)
print("HOTEL BOOKING ML — DATA AUDIT")
print("=" * 70)

print(f"\nLoading dataset from:")
print(DATA_PATH)

df = pd.read_csv(DATA_PATH)


### Basic information
# ========================
print("\n" + "=" * 70)
print("1. DATASET SHAPE")
print("=" * 70)

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]:,}")

print("\n" + "=" * 70)
print("2. COLUMN NAMES")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")



print("\n" + "=" * 70)
print("3. DATA TYPES")
print("=" * 70)

print(df.dtypes)

print("\n" + "=" * 70)
print("4. FIRST 5 ROWS")
print("=" * 70)

print(df.head())



### Missing Values
print("\n" + "=" * 70)
print("5. MISSING VALUES")
print("=" * 70)

missing=df.isnull().sum()

missing=missing[missing>0].sort_values(ascending=False)

if missing.empty:
    print("No missing values.")
else:
    missing_percentage=(missing/len(df)*100).round(2)
    
    missing_table=pd.DataFrame({
        "missing_count":missing,
        "missing_percentage":missing_percentage
    })
    print(missing_table)
    

# Duplicate rows
# =========================
print("\n" + "=" * 70)
print("6. DUPLICATE ROWS")
print("=" * 70)

duplicate_count = df.duplicated().sum()
print(f"Duplicate rows: {duplicate_count:,}")


# Target distribution
# ============================
print("\n" + "=" * 70)
print("7. TARGET DISTRIBUTION")
print("=" * 70)

target = "is_canceled"

print(df[target].value_counts())
print("\nPercentage:")
print(
    (df[target].value_counts(normalize=True)*100)
)


# Numerical summary
# =================================
print("\n" + "=" * 70)
print("8. NUMERICAL SUMMARY")
print("=" * 70)

print(df.describe().T)

# Categorical summary
# ===================================
print("\n" + "=" * 70)
print("9. CATEGORICAL FEATURES")
print("=" * 70)

categorical_columns=df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:
    print(f"\n---------{column}-----------")
    print(f"Unique values: {df[column].nunique(dropna=False):,}")
    
    print(df[column].value_counts(dropna=False).head(10))
    

