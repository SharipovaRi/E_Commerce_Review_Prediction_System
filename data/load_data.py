import pandas as pd
from pathlib import Path

RAW_TRAIN = Path("data/raw/train.csv")
RAW_TEST = Path("data/raw/test.csv")

PROCESSED_TRAIN = Path("data/processed/train_clean.csv")
PROCESSED_TEST = Path("data/processed/test_clean.csv")


# Load raw data
def load_data(path):
    return pd.read_csv(path)


# Minimal cleaning function (same for train and test)
def clean_data(df):
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=["review_text"])
    return df


# Save processed data
def save_data(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    # Process train
    train_df = load_data(RAW_TRAIN)
    train_df = clean_data(train_df)
    save_data(train_df, PROCESSED_TRAIN)

    # Process test
    test_df = load_data(RAW_TEST)
    test_df = clean_data(test_df)
    save_data(test_df, PROCESSED_TEST)

    print("Saved processed train and test datasets!")


if __name__ == "__main__":
    main()