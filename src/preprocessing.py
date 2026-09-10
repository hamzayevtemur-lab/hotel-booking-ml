from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "hotel_bookings.csv"

TARGET="is_canceled"

LEAKAGE_COLUMNS = [
    "reservation_status",
    "reservation_status_date",
]

def load_raw_data():
    return pd.read_csv(DATA_PATH)

def clean_data(df):
    df=df.copy()
    
    ##  Fill semantically missing children values
    df["children"]=df["children"].fillna(0)
    
    ## Remove bookings with zero guests
    guest_count=(
        df["adults"]+
        df["children"]
        +df["babies"]
    )
    
    df=df[guest_count>0].copy()
    
    ## Remove impossible negative ADR
    df=df[df["adr"]>=0].copy()
    
    ## Remove exact duplicate observations
    df=df.drop_duplicates().copy()
    
    ## Remove target-leaking columns
    df=df.drop(
        columns=LEAKAGE_COLUMNS
    )
    
    ## Treat Agent and Company IDS as categorical
    df["agent"] = df["agent"].fillna("Unknown").astype(str)
    df["company"] = df["company"].fillna("Unknown").astype(str)
    
    return df


if __name__=="__main__":
    print("=" * 70)
    print("HOTEL BOOKING ML — CLEANING CHECK")
    print("=" * 70)
    
    df=load_raw_data()
    
    print(f"\nRaw shape: {df.shape}")
    
    cleaned_df=clean_data(df)
    
    print(f"Cleaned shape:{cleaned_df.shape}")
    
    print("\nRemoved rows:")
    print(len(df) - len(cleaned_df))
    
    print("\nRemaining columns:")
    print(cleaned_df.columns.tolist())
    
    print("\nTarget distribution:")
    print(cleaned_df[TARGET].value_counts())
    
    print("\nMissing values:")
    print(
        cleaned_df.isnull()
        .sum()
        .sort_values(ascending=False)
        .head(20)
    )
    print("\nTarget percentage:")

    print(
        cleaned_df[TARGET]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )
    
    