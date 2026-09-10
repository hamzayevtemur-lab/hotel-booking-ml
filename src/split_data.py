from sklearn.model_selection import train_test_split

from src.preprocessing import load_raw_data, clean_data


RANDOM_STATE = 42
TARGET = "is_canceled"


def load_and_split_data():
    """
    Load the raw dataset, clean it, and create
    stratified train/validation/test splits.
    """

    df = clean_data(load_raw_data())

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


if __name__ == "__main__":

    print("=" * 70)
    print("HOTEL BOOKING ML — DATA SPLIT")
    print("=" * 70)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = load_and_split_data()

    print("\nSplit sizes:")

    print(f"Train      : {len(X_train):,}")
    print(f"Validation : {len(X_val):,}")
    print(f"Test       : {len(X_test):,}")

    print("\nTarget distribution:")

    print("\nTrain:")
    print(y_train.value_counts(normalize=True).round(4))

    print("\nValidation:")
    print(y_val.value_counts(normalize=True).round(4))

    print("\nTest:")
    print(y_test.value_counts(normalize=True).round(4))

    print("\nShapes:")

    print("X_train:", X_train.shape)
    print("X_val  :", X_val.shape)
    print("X_test :", X_test.shape)